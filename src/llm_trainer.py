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
    ) -> torch.Tensor:

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, S, V)


        # Shift for causal LM
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, S-1, V)
        labels = input_ids[:, 1:]                             # (B, S-1)
        token_logprobs = torch.gather(
            logprobs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)                                         # (B, S-1)

        masked = token_logprobs.masked_fill(~completion_mask, 0.0)
        lengths = completion_mask.sum(dim=-1).clamp(min=1)
        seq_logprob = masked.sum(dim=-1) / lengths

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


            # ---- DPO loss ----
            # scale by total number of mini-batches so total gradient matches one big batch
            loss = self._dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta)
            total_loss_val += float(loss.detach().cpu().item())
            num_batches += 1

            # backprop for THIS batch only; free graph right after
            (loss / 1.0).backward()  # If you want exact “once at the end” magnitude, use: (loss / num_total_batches)
            # We'll divide later after counting batches (see below).
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

    def load_model(self, path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)


    def compute_kl_scores(
        self,
        questions: List[str],
        candidates_by_q: List[List[str]],
        batch_size: int,
    ) -> List[List[float]]:
        """
        Compute mean per-token KL divergence KL(pol || ref) over completion tokens for each candidate.
        Returns nested List[List[float]].
        """
        self.model.eval()
        if hasattr(self.reference_model, "eval"):
            self.reference_model.eval()
        # Flatten
        flat_questions: List[str] = []
        flat_candidates: List[str] = []
        sizes: List[int] = []
        for qi, cands in enumerate(candidates_by_q):
            sizes.append(len(cands))
            for cand in cands:
                flat_questions.append(questions[qi])
                flat_candidates.append(cand)
        templated_prompts = build_prompts(flat_questions, self.tokenizer)
        prompt_lens = self._prompt_token_lengths(templated_prompts)
        flat_kls: List[float] = [0.0] * len(flat_candidates)
        for start in range(0, len(flat_candidates), batch_size):
            end = min(start + batch_size, len(flat_candidates))
            batch_qs = flat_questions[start:end]
            batch_cands = flat_candidates[start:end]
            batch = self._concat_tokenize(batch_qs, batch_cands)
            comp_mask = self._build_completion_mask(
                batch["input_ids"], batch["attention_mask"], prompt_lens[start:end]
            )
            try:
                with torch.no_grad():
                    # policy logits
                    pol_outputs = self.model(
                        input_ids=batch["input_ids"].to(self.model.device),
                        attention_mask=batch["attention_mask"].to(self.model.device),
                    )
                    pol_logits = pol_outputs.logits  # (B,S,V)
                    pol_logprobs = F.log_softmax(pol_logits[:, :-1, :], dim=-1)  # (B,S-1,V)
                    pol_probs = pol_logprobs.exp()  # (B,S-1,V)

                    # reference logits
                    ref_outputs = self.reference_model(
                        input_ids=batch["input_ids"].to(self.reference_model.device),
                        attention_mask=batch["attention_mask"].to(self.reference_model.device),
                    )
                    ref_logits = ref_outputs.logits
                    ref_logprobs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)

                    # KL per position: sum p_pol * (log p_pol - log p_ref)
                    kl_pos = (pol_probs * (pol_logprobs - ref_logprobs.to(pol_probs.device))).sum(dim=-1)  # (B,S-1)

                    # mask to completion tokens
                    comp_mask_dev = comp_mask.to(pol_probs.device)
                    masked = kl_pos.masked_fill(~comp_mask_dev, 0.0)
                    lengths = comp_mask.sum(dim=-1).clamp(min=1).to(pol_probs.device)
                    mean_kl = (masked.sum(dim=-1) / lengths).detach().cpu().tolist()
            except RuntimeError as e:
                # On any ref model failure, skip this batch and leave zeros (or could set None)
                print(f"KL batch exception: {e} — skipping these items.")
                mean_kl = [0.0] * (end - start)
            for i, val in enumerate(mean_kl):
                flat_kls[start + i] = float(val)
            # cleanup
            del batch, comp_mask, pol_outputs, pol_logits, pol_logprobs, pol_probs
            if 'ref_outputs' in locals():
                del ref_outputs, ref_logits, ref_logprobs
            del kl_pos, masked, lengths, mean_kl
            torch.cuda.empty_cache()
        # Unflatten
        result: List[List[float]] = []
        idx = 0
        for n in sizes:
            result.append(flat_kls[idx:idx + n])
            idx += n
        return result

    def compute_explore_and_entropy_scores(
        self,
        questions: List[str],
        candidates_by_q: List[List[str]],
        batch_size: int,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Returns:
            explore_scores: List[List[float]]  # 1 - softmax(logprob) per candidate
            entropy_scores: List[List[float]]  # avg token entropy per candidate
        Combines explore and entropy computations in one forward pass per batch.
        """
        self.model.eval()
        # Flatten inputs
        sizes: List[int] = []
        flat_questions: List[str] = []
        flat_candidates: List[str] = []
        for qi, cands in enumerate(candidates_by_q):
            sizes.append(len(cands))
            for cand in cands:
                flat_questions.append(questions[qi])
                flat_candidates.append(cand)
        total = len(flat_candidates)
        if total == 0:
            return [], []
        # Prompts and lengths
        templated_prompts = build_prompts(flat_questions, self.tokenizer)
        prompt_lens = self._prompt_token_lengths(templated_prompts)
        # Buffers
        flat_logprobs: List[float] = [0.0] * total
        flat_entropies: List[float] = [0.0] * total
        # Batched loop
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_qs = flat_questions[start:end]
            batch_cands = flat_candidates[start:end]
            # tokenize
            batch = self._concat_tokenize(batch_qs, batch_cands)
            comp_mask = self._build_completion_mask(
                batch["input_ids"], batch["attention_mask"], prompt_lens[start:end]
            )
            # forward once
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.model.device),
                    attention_mask=batch["attention_mask"].to(self.model.device),
                )
                logits = outputs.logits  # (B,S,V)
                # Shift for causal LM (align to labels)
                logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B,S-1,V)
                labels = batch["input_ids"].to(self.model.device)[:, 1:]  # (B,S-1)
                token_logprobs = torch.gather(
                    logprobs, dim=-1, index=labels.unsqueeze(-1)
                ).squeeze(-1)  # (B,S-1)
                comp_mask_dev = comp_mask.to(self.model.device)
                # sequence mean logprob over completion tokens
                masked_lp = token_logprobs.masked_fill(~comp_mask_dev, 0.0)
                lengths = comp_mask_dev.sum(dim=-1).clamp(min=1)
                seq_lp = (masked_lp.sum(dim=-1) / lengths).detach().cpu().tolist()
                # entropy per position: -sum p * log p
                probs = logprobs.exp()
                ent = -(probs * logprobs).sum(dim=-1)  # (B,S-1)
                masked_ent = ent.masked_fill(~comp_mask_dev, 0.0)
                avg_ent = (masked_ent.sum(dim=-1) / lengths).detach().cpu().tolist()
            # store
            for i in range(end - start):
                flat_logprobs[start + i] = float(seq_lp[i])
                flat_entropies[start + i] = float(avg_ent[i])
            # cleanup
            del batch, comp_mask, outputs, logits, logprobs, labels, token_logprobs, comp_mask_dev, masked_lp, lengths, seq_lp, probs, ent, masked_ent, avg_ent
            torch.cuda.empty_cache()
        # Per-question explore via softmax of logprobs
        explore_scores: List[List[float]] = []
        entropy_scores: List[List[float]] = []
        offset = 0
        for n in sizes:
            group_lp = flat_logprobs[offset:offset + n]
            group_ent = flat_entropies[offset:offset + n]
            offset += n
            if n == 0:
                explore_scores.append([])
                entropy_scores.append([])
                continue
            g_tensor = torch.tensor(group_lp, dtype=torch.float32)
            probs = torch.softmax(g_tensor, dim=0).cpu().tolist()
            explore = [float(1.0 - p) for p in probs]
            explore_scores.append(explore)
            entropy_scores.append(group_ent)
        return explore_scores, entropy_scores

def load_llm_trainer(model_name: str, gpu_id: int, num_steps: int, config: Optional[Dict[str, Any]] = None) -> LLMTrainer:
    return LLMTrainer(model_name, gpu_id, num_steps, config)
