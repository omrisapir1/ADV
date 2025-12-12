from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

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

    def _ipo_loss(
            self,
            pol_pos: torch.Tensor,
            pol_neg: torch.Tensor,
            beta: float,
    ) -> torch.Tensor:
        """
        IPO (reference-free preference loss):
        L = -log sigmoid( beta * (pol_pos - pol_neg) )
        """
        logits = beta * (pol_pos - pol_neg)
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

        beta = self.config["dpo_beta"]
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
            # try:
            #     with torch.no_grad():
            #         ref_pos = self._sequence_logprobs(
            #             self.reference_model, (batch_pos["input_ids"]).to(self.reference_model.device), (batch_pos["attention_mask"]).to(self.reference_model.device), comp_mask_pos.to(self.reference_model.device)
            #         ).to(self.device)
            #         ref_neg = self._sequence_logprobs(
            #             self.reference_model, (batch_neg["input_ids"]).to(self.reference_model.device), (batch_neg["attention_mask"]).to(self.reference_model.device), comp_mask_neg.to(self.reference_model.device)
            #         ).to(self.device)
            # except RuntimeError as e:
            #     print(f"Reference model exception: {e} will skip this batch.")
            #     torch.cuda.empty_cache()
            #     continue


            # ---- DPO loss ----
            # scale by total number of mini-batches so total gradient matches one big batch
            # loss = self._dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta)
            loss = self._ipo_loss(pol_pos, pol_neg, beta)

            total_loss_val += float(loss.detach().cpu().item())
            num_batches += 1

            # backprop for THIS batch only; free graph right after
            (loss / 1.0).backward()  # If you want exact “once at the end” magnitude, use: (loss / num_total_batches)
            # We'll divide later after counting batches (see below).
            # clip & step ONCE

            del loss, pol_pos, pol_neg
            del batch_pos, batch_neg, comp_mask_pos, comp_mask_neg, templated_prompts, prompt_lens
            # NOTE: empty_cache() does not free reserved memory to the OS, but can reduce fragmentation
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
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

    def save_state(self, path: str):
        """Save model along with optimizer and scheduler states."""
        # Save model weights (HF format)
        self.save_model(path)
        # Save optimizer/scheduler as a single checkpoint file
        state = {
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
        }
        torch.save(state, os.path.join(path, "trainer_optim.pt"))

    def load_state(self, path: str):
        """Load model plus optimizer and scheduler states if present."""
        # Load model from directory
        self.load_model(path)
        # Try load optimizer/scheduler
        ckpt_path = os.path.join(path, "trainer_optim.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            if self.optimizer is not None and state.get("optimizer") is not None:
                self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])
        # Ensure reference model config matches pad token and device
        if hasattr(self.reference_model, "eval"):
            self.reference_model.eval()
        # keep tokenizer padding side
        return

    def compute_kl_scores(
        self,
        questions: List[str],
        candidates_by_q: List[List[str]],
        batch_size: int,
    ) -> List[List[float]]:
        """
        Compute reference surprisal over completion tokens for each candidate:
        explore_score = - (1 / T_i) * sum_t log pi_ref(x_t | q)
        Returns nested List[List[float]] aligned with candidates_by_q.
        """
        # Put models in eval mode; we only use reference_model here
        if hasattr(self.reference_model, "eval"):
            self.reference_model.eval()
        # ------------------------------
        # Flatten inputs
        # ------------------------------
        flat_questions: List[str] = []
        flat_candidates: List[str] = []
        sizes: List[int] = []
        for qi, cands in enumerate(candidates_by_q):
            sizes.append(len(cands))
            for cand in cands:
                flat_questions.append(questions[qi])
                flat_candidates.append(self.clear_solution(cand))
        # Early return if nothing to score
        if len(flat_candidates) == 0:
            return [[] for _ in sizes]
        # Build prompts and prompt token lengths (for completion masking)
        templated_prompts = build_prompts(flat_questions, self.tokenizer)
        prompt_lens = self._prompt_token_lengths(templated_prompts)
        # Output buffer for explore scores (reference surprisal)
        flat_scores: List[float] = [0.0] * len(flat_candidates)
        # ------------------------------
        # Batched loop
        # ------------------------------
        for start in range(0, len(flat_candidates), batch_size):
            end = min(start + batch_size, len(flat_candidates))
            batch_qs = flat_questions[start:end]
            batch_cands = flat_candidates[start:end]
            # Tokenize prompt+completion
            batch = self._concat_tokenize(batch_qs, batch_cands)
            # Build completion mask (B, S-1)
            comp_mask = self._build_completion_mask(
                batch["input_ids"], batch["attention_mask"], prompt_lens[start:end]
            )
            try:
                with torch.no_grad():
                    # Reference forward only
                    ref_outputs = self.reference_model(
                        input_ids=batch["input_ids"].to(self.reference_model.device),
                        attention_mask=batch["attention_mask"].to(self.reference_model.device),
                    )
                    ref_logits = ref_outputs.logits  # (B, S, V)
                    # Shift for causal LM: use logits at positions [:, :-1, :]
                    ref_logprobs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)  # (B, S-1, V)
                    labels = batch["input_ids"].to(self.reference_model.device)[:, 1:]  # (B, S-1)
                    # Gather logprobs of actual generated tokens
                    gathered = torch.gather(ref_logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, S-1)
                    # Mask to completion tokens only and length-normalize
                    comp_mask_dev = comp_mask.to(self.reference_model.device)
                    masked = gathered.masked_fill(~comp_mask_dev, 0.0)
                    lengths = comp_mask.sum(dim=-1).clamp(min=1).to(self.reference_model.device)
                    mean_ref_logprob = (masked.sum(dim=-1) / lengths)  # (B,)
                    explore_scores = (-mean_ref_logprob).detach().cpu().tolist()
            except RuntimeError as e:
                # On any ref model failure, skip this batch and leave zeros
                print(f"Reference surprisal batch exception: {e} — skipping these items.")
                explore_scores = [0.0] * (end - start)
            # Store
            for i, val in enumerate(explore_scores):
                flat_scores[start + i] = float(val)
            # cleanup
            del batch, comp_mask, ref_outputs, ref_logits, ref_logprobs, labels, gathered
            del masked, lengths, mean_ref_logprob, explore_scores
            torch.cuda.empty_cache()
        # ------------------------------
        # Unflatten back to per-question lists
        # ------------------------------
        result: List[List[float]] = []
        idx = 0
        for n in sizes:
            result.append(flat_scores[idx:idx + n])
            idx += n
        return result

    def compute_avg_entropy(
            self,
            questions: List[str],
            candidates_by_q: List[List[str]],
            batch_size: int,
            top_k: int = 20,
            sample_ratio: float = 0.05,
    ) -> float:
        """
        Returns a single float: avg model token entropy over a random 5% sample of candidate solutions.
        """
        self.model.eval()

        # Flatten candidates same as before
        flat_questions: List[str] = []
        flat_candidates: List[str] = []
        for qi, cands in enumerate(candidates_by_q):
            for cand in cands:
                flat_questions.append(questions[qi])
                flat_candidates.append(self.clear_solution(cand))

        total = len(flat_candidates)
        if total == 0:
            return 0.0

        # Sample indices: 5% uniform random sample, but at least 1 if any candidates exist
        sample_size = max(1, int(total * sample_ratio))
        sample_size = min(sample_size, total)
        # Use torch for device-agnostic randomness, then move to CPU
        perm = torch.randperm(total).tolist()
        sampled_idx = perm[:sample_size]

        # Build sampled lists
        sampled_questions = [flat_questions[i] for i in sampled_idx]
        sampled_candidates = [flat_candidates[i] for i in sampled_idx]
        # Explicit cleanup of sampling buffers to reduce memory pressure
        del perm, sampled_idx

        # Build prompts and get prompt lengths (to mask completions)
        templated_prompts = build_prompts(sampled_questions, self.tokenizer)
        prompt_lens = self._prompt_token_lengths(templated_prompts)

        device = self.model.device
        per_sample_avg_ent: List[float] = []

        # Batched loop over sampled subset only
        for start in range(0, len(sampled_candidates), batch_size):
            end = min(start + batch_size, len(sampled_candidates))

            batch_qs = sampled_questions[start:end]
            batch_cands = sampled_candidates[start:end]

            batch = self._concat_tokenize(batch_qs, batch_cands)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            comp_mask = self._build_completion_mask(
                batch["input_ids"], batch["attention_mask"], prompt_lens[start:end]
            ).to(device)  # (B, S-1)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]  # (B, S-1, V)

                # Top-k entropy per step
                topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)  # (B, S-1, k)
                topk_logprobs = F.log_softmax(topk_vals, dim=-1)    # (B, S-1, k)
                topk_probs = torch.exp(topk_logprobs)
                ent_t = -(topk_probs * topk_logprobs).sum(dim=-1)   # (B, S-1)

                masked_ent = ent_t.masked_fill(~comp_mask, 0.0)
                lengths = comp_mask.sum(dim=-1).clamp(min=1)        # (B,)
                avg_ent_tensor = masked_ent.sum(dim=-1) / lengths   # (B,)
                avg_ent_list = avg_ent_tensor.detach().cpu().tolist()

            per_sample_avg_ent.extend(float(x) for x in avg_ent_list)

            # cleanup
            del batch, input_ids, attention_mask, comp_mask
            del outputs, logits, topk_vals, topk_logprobs, topk_probs, ent_t, masked_ent, lengths, avg_ent_tensor
        torch.cuda.empty_cache()

        # Final average over sampled candidates
        if len(per_sample_avg_ent) == 0:
            return 0.0
        return float(sum(per_sample_avg_ent) / len(per_sample_avg_ent))

    @staticmethod
    def clear_solution(full_solution: str) -> str:
        if '</think>' in full_solution:
            return full_solution[:full_solution.rfind('</think>')] + '</think>'
        if len(full_solution) < 50:
            return '</think>'
        ind = full_solution[:-50].rfind("\n")
        if ind == -1:
            return '</think>'
        return full_solution[:ind] + '</think>'

def load_llm_trainer(model_name: str, gpu_id: int, num_steps: int, config: Optional[Dict[str, Any]] = None) -> LLMTrainer:
    return LLMTrainer(model_name, gpu_id, num_steps, config)
