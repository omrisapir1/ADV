from __future__ import annotations

from typing import List, Tuple, Optional, Iterable
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator  # added for type hints
from .optimizer import create_optimizer, create_scheduler
from .losses import pairwise_rm_loss


class AceMathRewardModel:
    """
    Reward model with optional integrated training setup.
    If rm_config, num_steps, accelerator are provided at init, optimizer & scheduler are created and prepared.
    """

    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        rm_config: Optional[dict] = None,
        num_steps: Optional[int] = None,
        accelerator: Optional[Accelerator] = None,  # type hint updated
    ):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.rm_config = rm_config or {}
        self.train_config = self.rm_config.get("train", {}) if isinstance(self.rm_config.get("train", {}), dict) else {}

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        self.tokenizer.padding_side = "right"

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Training artifacts (optional)
        self.optimizer = None
        self.scheduler = None
        self.grad_accum = None
        self.grad_clip = None
        self.pair_batch_size = None

        if num_steps is not None and accelerator is not None:
            self._setup_training(num_steps, accelerator)

    def _setup_training(self, num_steps: int, accelerator: Accelerator):  # type hint
        """Initialize optimizer, scheduler, and training hyperparams; prepare via accelerator."""
        self.grad_accum = self.train_config.get("grad_accum")
        self.grad_clip = self.train_config.get("grad_clip")
        self.pair_batch_size = self.train_config.get("batch_size")
        self.optimizer = create_optimizer(self, self.rm_config)
        self.scheduler = create_scheduler(self.optimizer, self.rm_config, num_steps)
        # Prepare objects with accelerator
        self.model, self.optimizer, self.scheduler = accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

    # ------------------------- Chat formatting & tokenize -------------------------
    def _chat(self, question: str, solution: str):
        return [
            {
                "role": "system",
                "content": "Please reason step by step, and check your final answer within \\boxed{}.",
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]

    def _ids_only(self, question: str, solution: str) -> Tuple[list, int]:
        ids = self.tokenizer.apply_chat_template(
            self._chat(question, solution),
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            return_tensors=None,
        )
        return ids, len(ids)

    # ------------------------- Batching utilities -------------------------
    @staticmethod
    def _plan_batches_by_items(n_items: int, batch_size_items: int) -> Iterable[range]:
        for start in range(0, n_items, batch_size_items):
            yield range(start, min(start + batch_size_items, n_items))

    @staticmethod
    def _pad_to_multiple_of_8(batch: dict) -> dict:
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        L = input_ids.size(1)
        L8 = (L + 7) // 8 * 8
        if L8 == L:
            return batch
        pad_len = L8 - L
        pad_id = batch.get("pad_token_id", 0)
        pad_ids = torch.full(
            (input_ids.size(0), pad_len), pad_id, dtype=input_ids.dtype, device=input_ids.device
        )
        pad_mask = torch.zeros(
            (attn.size(0), pad_len), dtype=attn.dtype, device=attn.device
        )
        batch["input_ids"] = torch.cat([input_ids, pad_ids], dim=1)
        batch["attention_mask"] = torch.cat([attn, pad_mask], dim=1)
        return batch

    def _collate_and_forward(
        self,
        ids_lists: List[list],
        *,
        use_amp_bf16: bool,
        grad_enabled: bool,
        pad_to_mult8: bool,
    ) -> torch.Tensor:
        item_dicts = [{"input_ids": ids} for ids in ids_lists]
        batch = self.tokenizer.pad(item_dicts, padding=True, return_tensors="pt")
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        batch["pad_token_id"] = self.tokenizer.pad_token_id
        if pad_to_mult8:
            batch = self._pad_to_multiple_of_8(batch)
        ctx_amp = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp_bf16 else torch.nullcontext()
        ctx_grad = torch.enable_grad() if grad_enabled else torch.inference_mode()
        with ctx_grad, ctx_amp:
            out = self.model(**batch)
            logits = out.logits.squeeze(-1).to(dtype=torch.float32)
        return logits

    @staticmethod
    def pack_by_tokens(lengths: List[int], max_tokens_per_batch: int, max_seqs_per_batch: int) -> List[List[int]]:
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_tokens = 0
        for i in order:
            L = lengths[i]
            if cur and (cur_tokens + L > max_tokens_per_batch or len(cur) >= max_seqs_per_batch):
                batches.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(i)
            cur_tokens += L
        if cur:
            batches.append(cur)
        return batches

    # ------------------------- Public inference scoring -------------------------
    def score_reference(
        self,
        questions: List[str],
        candidates_by_q: List[List[str]],
        rm_config: Optional[dict] = None,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            rm_config = rm_config or self.rm_config
            pad_to_mult8 = bool(rm_config.get("pad_to_multiple_of_8", False))
            max_tokens = int(rm_config.get("max_tokens_per_batch_infer", 64000))
            max_seqs = int(rm_config.get("max_seqs_per_infer_batch", 80))
            skip_batching = bool(rm_config.get("skip_batching", False))
            meta = []  # (qi, kj, ids, length)
            max_k = 0
            for qi, (q, cand_list) in enumerate(zip(questions, candidates_by_q)):
                max_k = max(max_k, len(cand_list))
                for kj, s in enumerate(cand_list):
                    ids, L = self._ids_only(q, s)
                    meta.append((qi, kj, ids, L))
            if not meta:
                return torch.empty(len(questions), max_k, dtype=torch.float32)
            lengths = [m[3] for m in meta]
            scores = torch.empty(len(questions), max_k, dtype=torch.float32)
            scores.fill_(float("nan"))

            if skip_batching:
                # Single large batch inference (may OOM if too big)
                ids_lists = [m[2] for m in meta]
                print(f"[score_reference] skip_batching=True total_seqs={len(ids_lists)} total_tokens={sum(lengths)}")
                logits = self._collate_and_forward(
                    ids_lists,
                    use_amp_bf16=True,
                    grad_enabled=False,
                    pad_to_mult8=pad_to_mult8,
                )
                logits_cpu = logits.detach().to(dtype=torch.float32, device="cpu")
                for local_i, (qi, kj, _ids, _L) in enumerate(meta):
                    scores[qi, kj] = logits_cpu[local_i]
                del logits, logits_cpu
                torch.cuda.empty_cache()
                return scores

            # Packed batching path (default)
            batches = self.pack_by_tokens(lengths, max_tokens, max_seqs)
            for idxs in batches:
                batch_meta = [meta[i] for i in idxs]
                ids_lists = [m[2] for m in batch_meta]
                # logging batch size and token usage
                total_tokens = sum(m[3] for m in batch_meta)
                print(f"[score_reference] batch_size={len(batch_meta)} tokens={total_tokens}")
                logits = self._collate_and_forward(
                    ids_lists,
                    use_amp_bf16=True,
                    grad_enabled=False,
                    pad_to_mult8=pad_to_mult8,
                )
                logits_cpu = logits.detach().to(dtype=torch.float32, device="cpu")
                for local_i, (qi, kj, _ids, _L) in enumerate(batch_meta):
                    scores[qi, kj] = logits_cpu[local_i]
                del logits, logits_cpu, batch_meta, ids_lists
                torch.cuda.current_stream().synchronize()
                torch.cuda.empty_cache()
            return scores

    # ------------------------- Pair scoring (grad-enabled) -------------------------
    def score_pairs(
        self,
        questions: List[str],
        solutions_pos: List[str],
        solutions_neg: List[str],
        rm_config: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (r_pos, r_neg) score tensors for aligned pos/neg solutions.
        Interleaves pos/neg for efficient batching, length-sorts to reduce padding.
        """
        assert len(questions) == len(solutions_pos) == len(solutions_neg)
        if len(questions) == 0:
            return (torch.empty(0, device=self.device), torch.empty(0, device=self.device))
        rm_config = rm_config or self.rm_config
        pad_to_mult8 = bool(rm_config.get("pad_to_multiple_of_8", False))
        # Build interleaved ids and lengths
        inter_ids: List[list] = []
        lengths: List[int] = []
        for q, p, n in zip(questions, solutions_pos, solutions_neg):
            ids_p, Lp = self._ids_only(q, p)
            ids_n, Ln = self._ids_only(q, n)
            inter_ids.extend([ids_p, ids_n])
            lengths.extend([Lp, Ln])
        # Sort by length desc
        order = sorted(range(len(inter_ids)), key=lambda i: lengths[i], reverse=True)
        inter_ids_sorted = [inter_ids[i] for i in order]

        logits = self._collate_and_forward(
            inter_ids_sorted,
            use_amp_bf16=True,
            grad_enabled=True,
            pad_to_mult8=pad_to_mult8,
        )  # (2*B,)
        # Undo sort
        inv = torch.empty(len(order), dtype=torch.long, device=logits.device)
        inv[torch.tensor(order, device=logits.device)] = torch.arange(len(order), device=logits.device)
        original_logits = logits[inv]
        r_pos = original_logits[0::2]
        r_neg = original_logits[1::2]
        return r_pos, r_neg

    def train_step(self, triplets: List[Tuple[str, str, str]], accelerator) -> Tuple[float, float]:
        """Train over provided (question, pos_solution, neg_solution) triplets.
        For each batch (size = self.pair_batch_size or configured train batch_size):
          - Forward to obtain r_pos, r_neg
          - Compute pairwise loss
          - Backprop immediately
          - Delete per-batch tensors and empty CUDA cache
        After all batches, apply optimizer & scheduler step once, clip grads, zero grads.
        Returns (avg_loss, current_lr).
        """
        assert self.optimizer is not None and self.scheduler is not None, "Optimizer/scheduler not initialized."

        self.model.train()
        batch_size = self.pair_batch_size or self.train_config.get("batch_size")
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(triplets), batch_size):
            end = min(start + batch_size, len(triplets))
            batch = triplets[start:end]
            batch_q = [t[0] for t in batch]
            batch_pos = [t[1] for t in batch]
            batch_neg = [t[2] for t in batch]

            # Forward + loss + backward
            r_pos, r_neg = self.score_pairs(batch_q, batch_pos, batch_neg, self.rm_config)
            assert r_pos.shape == r_neg.shape, "r_pos and r_neg shape mismatch"
            loss = pairwise_rm_loss(r_pos, r_neg)
            accelerator.backward(loss)
            total_loss += loss.detach().item()
            num_batches += 1
            # Free per-batch tensors explicitly
            del r_pos, r_neg, loss, batch_q, batch_pos, batch_neg, batch
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches else 0.0

        if num_batches:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip or 1.0)
            self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        current_lr = self.scheduler.get_last_lr()[0]
        return avg_loss, current_lr


def load_reward_model(
    model_name: str,
    gpu_id: int,
    rm_config: Optional[dict] = None,
    num_steps: Optional[int] = None,
    accelerator: Optional[object] = None,
) -> AceMathRewardModel:
    return AceMathRewardModel(model_name, gpu_id, rm_config, num_steps, accelerator)
