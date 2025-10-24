from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from .optimizer import create_optimizer, create_scheduler
from .prompting import build_prompts

class LLMTrainer:
    """Lightweight LLM trainer scaffold with DPO train_step."""

    def __init__(self, model_name: str, gpu_id: int, num_steps: int, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self.tokenizer.pad_token_id = eos_id
        # For causal LM batching, left padding often helps.
        self.tokenizer.padding_side = "left"

        dtype = torch.bfloat16

        # Primary (trainable) model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)
        for p in self.reference_model.parameters():
            p.requires_grad_(False)

        self.reference_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = create_optimizer(self, config=config)
        self.scheduler = create_scheduler(self.optimizer, num_steps)
        self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def _prompt_token_lengths(self, prompts: List[str]) -> List[int]:
        """Tokenize prompts only to know how many tokens to mask out from the completion likelihood."""
        toks = self.tokenizer(prompts, padding=False, truncation=True, return_tensors=None)
        # length per example
        return [len(ids) for ids in toks["input_ids"]]

    def _concat_tokenize(
            self,
            questions: List[str],
            completions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Use build_prompts() to construct system+user chat template,
        then append the assistant completion text for tokenization.
        """
        # Build chat-style prompts
        prompts = build_prompts(questions, self.tokenizer)

        # Join each prompt with its completion.
        # For chat templates, completions are usually just concatenated directly.
        texts = [p + c for p, c in zip(prompts, completions)]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}


    def _build_completion_mask(
        self,
        input_ids: torch.Tensor,          # (B, S)
        attention_mask: torch.Tensor,     # (B, S)
        prompt_lengths: List[int],        # len B, number of tokens belonging to prompt
    ) -> torch.Tensor:
        """
        Build a boolean mask (B, S-1) to select next-token log-probs that correspond
        to completion tokens only. We align with shifted labels (teacher forcing),
        so we drop the first position.
        """
        B, S = input_ids.shape
        # Positions that produce the token at index t are logits at index t-1.
        # We'll create a per-sample range mask for token indices [prompt_len, S-1)
        # Then we shift it left to match logits for labels[:, 1:].
        comp_mask = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        for i, plen in enumerate(prompt_lengths):
            # completion tokens start at position 'plen' (0-based)
            if plen < S:
                comp_mask[i, plen:] = True
        # Drop first position to align with labels[:, 1:]
        comp_mask = comp_mask[:, 1:]
        # Also respect padding (attention_mask for labels side is attention_mask[:, 1:])
        comp_mask = comp_mask & (attention_mask[:, 1:].bool())
        return comp_mask  # (B, S-1)

    def _sequence_logprobs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,          # (B, S)
        attention_mask: torch.Tensor,     # (B, S)
        completion_mask: torch.Tensor,    # (B, S-1) selecting completion tokens
    ) -> torch.Tensor:
        """
        Compute sequence-level log-prob over completion tokens only.
        Uses standard causal LM shift: logits[:, :-1, :] vs labels[:, 1:].
        Returns (B,) sequence log-probs (sum over selected positions).
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, S, V)
        # Shift for next-token prediction
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)              # (B, S-1, V)
        labels = input_ids[:, 1:]                                        # (B, S-1)
        token_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, S-1)

        # Mask to completion tokens only
        token_logprobs = token_logprobs.masked_fill(~completion_mask, 0.0)
        seq_logprob = token_logprobs.sum(dim=-1)  # sum over completion tokens
        return seq_logprob  # (B,)

    def _dpo_loss(
        self,
        pol_pos: torch.Tensor, pol_neg: torch.Tensor,
        ref_pos: torch.Tensor, ref_neg: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """
        pol_* and ref_* are sequence log-probs over completion tokens (shape: (B,))
        DPO loss = -log sigmoid( beta * [ (pol_pos - pol_neg) - (ref_pos - ref_neg) ] )
        """
        logits = beta * ((pol_pos - pol_neg) - (ref_pos - ref_neg))
        return -F.logsigmoid(logits).mean()


    def train_step(self, triplets: List[Tuple[str, str, str]]) -> float:
        """
        triplets: list of (prompt_question, chosen_completion, rejected_completion)
        Trains in mini-batches using config['batch_size'] (default 1).
        Returns the average DPO loss over all mini-batches.

        Strategy:
          - Accumulate gradients across all mini-batches.
          - Call optimizer.step() and scheduler.step() ONCE at the end.
          - Free per-batch CUDA tensors ASAP to keep memory low.
        """
        self.model.gradient_checkpointing_enable()
        self.model.train()
        if hasattr(self.reference_model, "eval"):
            self.reference_model.eval()  # make sure ref is not building grads

        beta = float(self.config.get("dpo_beta", 0.1))
        train_batch_size = int(self.config.get("batch_size", 1))
        max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        print(f'beta = {beta}, batch_size = {train_batch_size}, max_grad_norm = {max_grad_norm}')
        total_loss_val = 0.0
        num_batches = 0

        # start a fresh grad buffer
        self.optimizer.zero_grad(set_to_none=True)

        # loop mini-batches
        for start in range(0, len(triplets), train_batch_size):
            end = min(start + train_batch_size, len(triplets))
            batch = triplets[start:end]

            questions = [t[0] for t in batch]
            pos = [t[1] for t in batch]
            neg = [t[2] for t in batch]

            # ---- prompts via chat template (system+user), then lengths for masking ----
            templated_prompts = build_prompts(questions, self.tokenizer)
            prompt_lens = self._prompt_token_lengths(templated_prompts)

            # ---- tokenize (prompt+completion) for pos/neg; uses same chat template internally ----
            batch_pos = self._concat_tokenize(questions, pos)
            batch_neg = self._concat_tokenize(questions, neg)

            # ---- build completion masks aligned with shifted labels ----
            comp_mask_pos = self._build_completion_mask(
                batch_pos["input_ids"], batch_pos["attention_mask"], prompt_lens
            )
            comp_mask_neg = self._build_completion_mask(
                batch_neg["input_ids"], batch_neg["attention_mask"], prompt_lens
            )

            # ---- forward passes (policy) ----
            pol_pos = self._sequence_logprobs(
                self.model, batch_pos["input_ids"], batch_pos["attention_mask"], comp_mask_pos
            )
            pol_neg = self._sequence_logprobs(
                self.model, batch_neg["input_ids"], batch_neg["attention_mask"], comp_mask_neg
            )
            print(templated_prompts)
            print(self.tokenizer.decode(batch_pos["input_ids"][0]))
            print(self.tokenizer.decode(batch_neg["input_ids"][0]))
            

            # ---- forward passes (reference) - no grad ----
            with torch.no_grad():
                ref_pos = self._sequence_logprobs(
                    self.reference_model, batch_pos["input_ids"], batch_pos["attention_mask"], comp_mask_pos
                )
                ref_neg = self._sequence_logprobs(
                    self.reference_model, batch_neg["input_ids"], batch_neg["attention_mask"], comp_mask_neg
                )

            # ---- DPO loss ----
            # scale by total number of mini-batches so total gradient matches one big batch
            loss = self._dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta)
            total_loss_val += float(loss.detach().cpu().item())
            num_batches += 1

            # backprop for THIS batch only; free graph right after
            (loss / 1.0).backward()  # If you want exact “once at the end” magnitude, use: (loss / num_total_batches)
            # We'll divide later after counting batches (see below).

            # ---- per-batch cleanup: drop tensors and clear CUDA cache pressure ----
            del loss, pol_pos, pol_neg, ref_pos, ref_neg
            del batch_pos, batch_neg, comp_mask_pos, comp_mask_neg, templated_prompts, prompt_lens
            # NOTE: empty_cache() does not free reserved memory to the OS, but can reduce fragmentation
            torch.cuda.empty_cache()

        if num_batches > 0:
            # Optional: rescale accumulated grads if you didn't divide each batch’s loss earlier
            # Here we normalize to the mean loss so gradient magnitude matches a single big batch
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.div_(num_batches)

            # clip & step ONCE
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

        # clear grad buffers for next call
        self.optimizer.zero_grad(set_to_none=True)

        return total_loss_val / max(1, num_batches)

    def save_model(self, tmp_weights_path: str):
        self.model.save_pretrained(tmp_weights_path, safe_serialization=True)


def load_llm_trainer(model_name: str, gpu_id: int, num_steps: int, config: Optional[Dict[str, Any]] = None) -> LLMTrainer:
    return LLMTrainer(model_name, gpu_id, num_steps, config)

