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
        self.scheduler = create_scheduler(self.optimizer, num_steps, config=config)
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
        return {k: v for k, v in batch.items()}


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

    # def _sequence_logprobs(
    #     self,
    #     model: AutoModelForCausalLM,
    #     input_ids: torch.Tensor,          # (B, S)
    #     attention_mask: torch.Tensor,     # (B, S)
    #     completion_mask: torch.Tensor,    # (B, S-1) selecting completion tokens
    # ) -> torch.Tensor:
    #     """
    #     Compute length-normalized (mean) log-prob over completion tokens only.
    #     Uses standard causal LM shift: logits[:, :-1, :] vs labels[:, 1:].
    #     Returns (B,) mean per-token log-prob across selected completion positions.
    #     """
    #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    #     logits = outputs.logits  # (B, S, V)
    #     logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)              # (B, S-1, V)
    #     labels = input_ids[:, 1:]                                        # (B, S-1)
    #     token_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, S-1)
    #
    #     # Mask to completion tokens only (keep zeros for non-completion positions)
    #     masked = token_logprobs.masked_fill(~completion_mask, 0.0)
    #     lengths = completion_mask.sum(dim=-1).clamp(min=1)  # (B,)
    #     seq_logprob = masked.sum(dim=-1) / lengths          # mean per completion token
    #     return seq_logprob  # (B,)

    def _sequence_logprobs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,          # (B, S)
        attention_mask: torch.Tensor,     # (B, S)
        completion_mask: torch.Tensor,    # (B, S-1)
        debug: bool = True,
    ) -> torch.Tensor:

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, S, V)


        # Shift for causal LM
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, S-1, V)
        labels = input_ids[:, 1:]                             # (B, S-1)
        token_logprobs = torch.gather(
            logprobs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)                                         # (B, S-1)

        for i in range(150):
            tid = labels[0, i].item()
            tok = self.tokenizer.decode([tid])
            logp = token_logprobs[0, i].item()  # log P(token_i | prefix)
            prob = token_logprobs[0, i].exp().item()  # P(token_i | prefix)
            print(i, labels[0, i].item(), self.tokenizer.decode([labels[0, i].item()]))
            print(f"{i:3d}  id={tid:6d}  token={tok!r:15s}  logp={logp:8.4f}  p={prob: .4e}")

        idx = labels[0, 112].item()
        print("label token id:", idx)
        print("decoded:", self.tokenizer.decode([idx]))
        print("logprob for this index:", logprobs[0, 111, idx])
        print("token prob at this index:", torch.exp(logprobs[0, 111, idx]))

        step = 111  # the timestep before "To"

        step_logprobs = logprobs[0, step]  # (V,)
        topk_vals, topk_idx = torch.topk(step_logprobs, k=10)

        print("\nTop-10 tokens at this step:")
        for rank in range(10):
            tid = topk_idx[rank].item()
            print(
                f"#{rank + 1}: id={tid}, token={self.tokenizer.decode([tid])!r}, "
                f"logp={topk_vals[rank].item():.4f}, p={topk_vals[rank].exp().item():.4e}"
            )

        # Where is "To" in the ranking?
        to_id = idx  # 1249
        to_logp = step_logprobs[to_id]
        to_rank = (step_logprobs > to_logp).sum().item() + 1
        print(f"\n'To' rank: {to_rank}, logp={to_logp.item():.4f}, p={to_logp.exp().item():.4e}")

        # Apply mask
        masked = token_logprobs.masked_fill(~completion_mask, 0.0)
        lengths = completion_mask.sum(dim=-1).clamp(min=1)
        seq_logprob = masked.sum(dim=-1) / lengths

        # ---------------------------
        # DEBUG PRINTS
        # ---------------------------
        if debug:
            B, S = input_ids.shape
            print("\n" + "="*80)
            print("DEBUG `_sequence_logprobs`")
            print("="*80)

            print(f"Batch size: {B}, Sequence length: {S}")
            print(f"Completion lengths: {lengths.tolist()}")

            # Show which positions are marked as completion tokens
            print("\nCompletion mask (first 2 examples):")
            for i in range(min(2, B)):
                print(f"  Example {i}:")
                print(f"    mask sum:  {completion_mask[i].sum().item()}")
                print(f"    mask bool: {completion_mask[i].tolist()}")

            # Show the raw per-token logprobs
            print("\nToken log-probs (first example):")
            if B > 0:
                print(token_logprobs[0].tolist())

            # Show masked token log-probs
            print("\nMasked token log-probs (first example):")
            if B > 0:
                print(masked[0].tolist())

            # Show final aggregated log-probs
            print("\nFinal seq_logprob (per example):")
            print(seq_logprob.tolist())

            # Decode prompt + completion tokens for sanity check
            print("\nDecoded prompt + completion snapshots:")
            for i in range(min(2, B)):
                decoded_full = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                print(f"  Example {i} text:")
                print(decoded_full)

                # decode only completion region
                comp_positions = completion_mask[i].nonzero(as_tuple=True)[0].tolist()
                if comp_positions:
                    start = comp_positions[0]+1  # +1 due to labels shift
                    end   = comp_positions[-1]+2
                    decoded_completion = self.tokenizer.decode(
                        input_ids[i, start:end], skip_special_tokens=False
                    )
                    print(f"  Completion-only text: {decoded_completion}")
                else:
                    print("  Completion-only text: <EMPTY>")

            print("="*80 + "\n")
        exit(0)
        return seq_logprob

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
                self.model, (batch_pos["input_ids"]).to(self.model.device), (batch_pos["attention_mask"]).to(self.model.device), comp_mask_pos.to(self.model.device)
            )
            pol_neg = self._sequence_logprobs(
                self.model, (batch_neg["input_ids"]).to(self.model.device), (batch_neg["attention_mask"]).to(self.model.device), comp_mask_neg.to(self.model.device)
            )



            # ---- forward passes (reference) - no grad ---
            try:
                with torch.no_grad():
                    ref_pos = self._sequence_logprobs(
                        self.reference_model, (batch_pos["input_ids"]).to(self.reference_model.device), (batch_pos["attention_mask"]).to(self.reference_model.device), comp_mask_pos.to(self.reference_model.device)
                    ).to(self.device)
                    ref_neg = self._sequence_logprobs(
                        self.reference_model, (batch_neg["input_ids"]).to(self.reference_model.device), (batch_neg["attention_mask"]).to(self.reference_model.device), comp_mask_neg.to(self.reference_model.device)
                    ).to(self.device)
            except RuntimeError as e:
                print(f"Reference model exception: {e} will skip this batch.")
                torch.cuda.empty_cache()
                continue
            raise

            # ---- DPO loss ----
            # scale by total number of mini-batches so total gradient matches one big batch
            loss = self._dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta)
            total_loss_val += float(loss.detach().cpu().item())
            num_batches += 1

            # backprop for THIS batch only; free graph right after
            (loss / 1.0).backward()  # If you want exact “once at the end” magnitude, use: (loss / num_total_batches)
            # We'll divide later after counting batches (see below).

            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.div_(num_batches)

            # clip & step ONCE
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            del loss, pol_pos, pol_neg, ref_pos, ref_neg
            del batch_pos, batch_neg, comp_mask_pos, comp_mask_neg, templated_prompts, prompt_lens
            # NOTE: empty_cache() does not free reserved memory to the OS, but can reduce fragmentation
            torch.cuda.empty_cache()
        self.scheduler.step()
        return total_loss_val / max(1, num_batches)

    def save_model(self, tmp_weights_path: str):
        self.model.save_pretrained(tmp_weights_path, safe_serialization=True)

    def update_ref_model(self):
        self.reference_model.load_state_dict(self.model.state_dict())
        print('updated reference model')


def load_llm_trainer(model_name: str, gpu_id: int, num_steps: int, config: Optional[Dict[str, Any]] = None) -> LLMTrainer:
    return LLMTrainer(model_name, gpu_id, num_steps, config)
