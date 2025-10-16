from __future__ import annotations

from typing import List, Tuple, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator
from .optimizer import create_optimizer, create_scheduler
from .losses import pairwise_rm_loss


class AceMathRewardModel:
    """Reward model with optional integrated training setup.
    Uses batch (vectorized) tokenization, optional double-buffered GPU prefetch, pinned memory.
    """
    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        rm_config: Optional[dict] = None,
        num_steps: Optional[int] = None,
        accelerator: Optional[Accelerator] = None,
    ):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.rm_config = rm_config or {}
        self.train_config = self.rm_config.get("train", {}) if isinstance(self.rm_config.get("train", {}), dict) else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
            # attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = None
        self.scheduler = None
        self.grad_accum = None
        self.grad_clip = None
        self.pair_batch_size = None

        if num_steps is not None and accelerator is not None:
            self._setup_training(num_steps, accelerator)

    def _setup_training(self, num_steps: int, accelerator: Accelerator):
        self.grad_accum = self.train_config.get("grad_accum")
        self.grad_clip = self.train_config.get("grad_clip")
        self.pair_batch_size = self.train_config.get("batch_size")
        self.optimizer = create_optimizer(self, self.rm_config)
        self.scheduler = create_scheduler(self.optimizer, self.rm_config, num_steps)
        self.model, self.optimizer, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.scheduler)

    def _chat(self, question: str, solution: str):
        return [
            {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]

    # -------------------- Packing utility (token-count based) --------------------
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

    # -------------------- Core forward (expects already tokenized batch on device) --------------------
    def _forward_logits(self, enc: dict, *, grad_enabled: bool) -> torch.Tensor:
        ctx_amp = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.device.startswith("cuda") else torch.nullcontext()
        ctx_grad = torch.enable_grad() if grad_enabled else torch.inference_mode()
        with ctx_grad, ctx_amp:
            out = self.model(**enc)
            return out.logits.squeeze(-1).to(dtype=torch.float32)

    def _apply_padding_and_move(self, batch_texts: List[str], pad_to_mult8: bool, grad_enabled: bool) -> torch.Tensor:
        raise NotImplementedError("Deprecated path; not used after refactor")

    # -------------------- Reference scoring (batch tokenization + optional double buffering) --------------------
    def score_reference(self, questions: List[str], candidates_by_q: List[List[str]], rm_config: Optional[dict] = None) -> torch.Tensor:
        self.model.eval()
        rm_config = rm_config or self.rm_config
        pad_to_mult8 = bool(rm_config.get("pad_to_multiple_of_8", False))
        max_tokens = int(rm_config.get("max_tokens_per_batch_infer", 64000))
        max_seqs = int(rm_config.get("max_seqs_per_infer_batch", 80))

        texts: List[str] = []
        meta: List[Tuple[int, int]] = []  # (qi, kj)
        max_k = 0
        for qi, (q, cand_list) in enumerate(zip(questions, candidates_by_q)):
            max_k = max(max_k, len(cand_list))
            for kj, sol in enumerate(cand_list):
                chat = self._chat(q, sol)
                text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                texts.append(text)
                meta.append((qi, kj))
        if not texts:
            return torch.empty(len(questions), max_k, dtype=torch.float32)

        # Single global tokenization to obtain lengths (no padding) for packing
        prelim = self.tokenizer(texts, padding=False, truncation=True)
        lengths = [len(ids) for ids in prelim["input_ids"]]
        scores = torch.empty(len(questions), max_k, dtype=torch.float32).fill_(float("nan"))

        batches = self.pack_by_tokens(lengths, max_tokens, max_seqs)
        if not batches:
            return scores

        use_double_buffer = torch.cuda.is_available() and len(batches) > 1
        prefetch_stream = torch.cuda.Stream(device=torch.device(self.device)) if use_double_buffer else None
        next_enc = None

        def prepare_batch(idxs: List[int]):
            batch_texts = [texts[i] for i in idxs]
            enc_local = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                pad_to_multiple_of=8 if pad_to_mult8 else None,
            )
            if torch.cuda.is_available():
                for k, v in enc_local.items():
                    if v.device.type == "cpu":
                        enc_local[k] = v.pin_memory()
                enc_local = {k: v.to(self.device, non_blocking=True) for k, v in enc_local.items()}
            return enc_local

        # Prefetch first batch synchronously (or on stream then sync)
        if use_double_buffer:
            with torch.cuda.stream(prefetch_stream):
                next_enc = prepare_batch(batches[0])
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        else:
            next_enc = prepare_batch(batches[0])

        for bi, idxs in enumerate(batches):
            current_enc = next_enc
            # Launch prefetch for next batch if exists
            if use_double_buffer and bi + 1 < len(batches):
                with torch.cuda.stream(prefetch_stream):
                    next_enc = prepare_batch(batches[bi + 1])
            # Ensure current batch ready
            if use_double_buffer:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
            logits = self._forward_logits(current_enc, grad_enabled=False)
            logits_cpu = logits.detach().to(dtype=torch.float32, device="cpu")
            for local_i, global_i in enumerate(idxs):
                qi, kj = meta[global_i]
                scores[qi, kj] = logits_cpu[local_i]
            del logits, logits_cpu, current_enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores

    # -------------------- Pair scoring (pos/neg) --------------------
    def score_pairs(self, questions: List[str], solutions_pos: List[str], solutions_neg: List[str], rm_config: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(questions) == len(solutions_pos) == len(solutions_neg)
        if len(questions) == 0:
            return (torch.empty(0, device=self.device), torch.empty(0, device=self.device))
        rm_config = rm_config or self.rm_config
        pad_to_mult8 = bool(rm_config.get("pad_to_multiple_of_8", False))

        # Build interleaved texts (pos, neg)
        texts: List[str] = []
        for q, p, n in zip(questions, solutions_pos, solutions_neg):
            texts.append(self.tokenizer.apply_chat_template(self._chat(q, p), tokenize=False, add_generation_prompt=False))
            texts.append(self.tokenizer.apply_chat_template(self._chat(q, n), tokenize=False, add_generation_prompt=False))
        # Initial global tokenization (no padding) to get lengths for sorting
        prelim = self.tokenizer(texts, padding=False, truncation=True)
        lengths = [len(ids) for ids in prelim["input_ids"]]
        order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
        texts_sorted = [texts[i] for i in order]

        enc = self.tokenizer(
            texts_sorted,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8 if pad_to_mult8 else None,
        )
        if torch.cuda.is_available():
            for k, v in enc.items():
                if v.device.type == "cpu":
                    enc[k] = v.pin_memory()
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        logits_sorted = self._forward_logits(enc, grad_enabled=True)
        # Undo sort correctly: place each sorted logit back at original index
        original_logits = torch.empty_like(logits_sorted)
        original_order_tensor = torch.tensor(order)
        original_logits[original_order_tensor] = logits_sorted
        r_pos = original_logits[0::2]
        r_neg = original_logits[1::2]
        return r_pos, r_neg

    def train_step(self, triplets: List[Tuple[str, str, str]], accelerator, forced_batch_size=None) -> Tuple[float, float]:
        assert self.optimizer is not None and self.scheduler is not None, "Optimizer/scheduler not initialized."
        self.model.train()
        batch_size = forced_batch_size or self.pair_batch_size
        total_loss = 0.0
        num_batches = 0
        for start in range(0, len(triplets), batch_size):
            end = min(start + batch_size, len(triplets))
            batch = triplets[start:end]
            batch_q = [t[0] for t in batch]
            batch_pos = [t[1] for t in batch]
            batch_neg = [t[2] for t in batch]
            r_pos, r_neg = self.score_pairs(batch_q, batch_pos, batch_neg, self.rm_config)
            assert r_pos.shape == r_neg.shape
            loss = pairwise_rm_loss(r_pos, r_neg)
            accelerator.backward(loss)
            total_loss += loss.detach().item()
            num_batches += 1
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


def load_reward_model(model_name: str, gpu_id: int, rm_config: Optional[dict] = None, num_steps: Optional[int] = None, accelerator: Optional[object] = None) -> AceMathRewardModel:
    return AceMathRewardModel(model_name, gpu_id, rm_config, num_steps, accelerator)
