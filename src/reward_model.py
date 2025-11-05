from __future__ import annotations

from typing import List, Tuple, Optional
import os
import torch
from transformers import AutoModel, AutoTokenizer

from .prompting import SYSTEM_PROMPT

try:
    from .optimizer import create_optimizer, create_scheduler
    from .losses import pairwise_rm_loss
except:
    from optimizer import create_optimizer, create_scheduler
    from losses import pairwise_rm_loss


# Utility to find last occurrence index of a token id per sequence
class AceMathRewardModel:
    """Reward model adapted to PRM (Process Reward Model).
    Uses built-in PRM head accessed via model.score; pools final hidden state at last </think> token.
    Reward is positive class logit (index 1). Keeps original batching & loss logic.
    """
    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        rm_config: Optional[dict] = None,
        num_steps: Optional[int] = None,
    ):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.rm_config = rm_config  # no fallback
        self.train_config = self.rm_config.get("train")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        self.tokenizer.padding_side = "right"

        # Load PRM base model (with built-in head: model.score)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        # Removed strict .score validation to allow StudentPRM models without this attr
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        # Create frozen reference PRM model
        self.ref_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        # Removed strict .score validation
        self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.pool_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.model.pool_id = self.pool_token_id
        self.ref_model.pool_id = self.pool_token_id

        self.optimizer = None
        self.scheduler = None
        self.grad_accum = None
        self.grad_clip = None
        self.pair_batch_size = None

        if num_steps is not None:
            self._setup_training(num_steps)

    def _setup_training(self, num_steps: int):
        self.grad_accum = int(self.train_config.get("grad_accum"))
        self.grad_clip = self.train_config.get("grad_clip")
        self.pair_batch_size = int(self.train_config.get("batch_size"))
        self.optimizer = create_optimizer(self, self.rm_config)
        self.scheduler = create_scheduler(self.optimizer, num_steps)
        # if hasattr(self.model, "gradient_checkpointing_enable"):
        #     self.model.gradient_checkpointing_enable()

    def save_state(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "ref_model_state": self.ref_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.rm_config,
            },
            os.path.join(path, "reward_model.pt"),
        )

    def _chat(self, question: str, solution: str):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,continue_final_message=True)

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

    def extract_last_think(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        pos = last_token_index(input_ids, self.pool_token_id)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden_states[batch_idx, pos]

    def _forward_logits(self, enc: dict, *, grad_enabled: bool, model=None) -> torch.Tensor:
        m = model if model is not None else self.model
        ctx_amp = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.device.startswith("cuda") else torch.nullcontext()
        ctx_grad = torch.enable_grad() if grad_enabled else torch.inference_mode()
        print(enc)
        with ctx_grad, ctx_amp:
            out = m(**enc)
            logits = out.logits
            print(logits)
            return logits.to(dtype=torch.float32)

    def _apply_padding_and_move(self, batch_texts: List[str], pad_to_mult8: bool, grad_enabled: bool) -> torch.Tensor:
        raise NotImplementedError("Deprecated path; not used after refactor")

    # -------------------- Reference scoring --------------------
    def score_reference(self, questions: List[str], candidates_by_q: List[List[str]], rm_config: Optional[dict] = None, forced_small_batch_size=False) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.ref_model.eval()
        pad_to_mult8 = bool(rm_config.get("pad_to_multiple_of_8"))
        max_tokens = int(rm_config.get("max_tokens_per_batch_infer"))
        max_seqs = int(rm_config.get("max_seqs_per_infer_batch"))
        if forced_small_batch_size:
            max_tokens = int(max_tokens * 0.25)
            max_seqs = int(max_seqs * 0.25)

        texts: List[str] = []
        meta: List[Tuple[int, int]] = []  # (qi, kj)
        max_k = 0
        for qi, (q, cand_list) in enumerate(zip(questions, candidates_by_q)):
            cand_list = [self.clear_solution(s) for s in cand_list]
            max_k = max(max_k, len(cand_list))
            for kj, sol in enumerate(cand_list):
                text = self._chat(q, sol)
                print(text)
                texts.append(text)
                meta.append((qi, kj))
        if not texts:
            empty = torch.empty(len(questions), max_k, dtype=torch.float32)
            return empty, empty.clone()

        prelim = self.tokenizer(texts, padding=False, truncation=True)
        lengths = [len(ids) for ids in prelim["input_ids"]]
        scores_model = torch.empty(len(questions), max_k, dtype=torch.float32).fill_(float("nan"))
        scores_ref = torch.empty(len(questions), max_k, dtype=torch.float32).fill_(float("nan"))

        batches = self.pack_by_tokens(lengths, max_tokens, max_seqs)
        if not batches:
            return scores_model, scores_ref

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
        if use_double_buffer:
            with torch.cuda.stream(prefetch_stream):
                next_enc = prepare_batch(batches[0])
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        else:
            next_enc = prepare_batch(batches[0])

        for bi, idxs in enumerate(batches):
            current_enc = next_enc
            if use_double_buffer and bi + 1 < len(batches):
                with torch.cuda.stream(prefetch_stream):
                    next_enc = prepare_batch(batches[bi + 1])
            if use_double_buffer:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
            logits_model = self._forward_logits(current_enc, grad_enabled=False, model=self.model)
            logits_ref = self._forward_logits(current_enc, grad_enabled=False, model=self.ref_model)
            r_model = logits_model[:, 1].detach().to(dtype=torch.float32, device="cpu")
            r_ref = logits_ref[:, 1].detach().to(dtype=torch.float32, device="cpu")
            for local_i, global_i in enumerate(idxs):
                qi, kj = meta[global_i]
                scores_model[qi, kj] = r_model[local_i]
                scores_ref[qi, kj] = r_ref[local_i]
            del logits_model, logits_ref, r_model, r_ref, current_enc
        return scores_model, scores_ref

    # -------------------- Pair scoring (pos/neg) --------------------
    def score_pairs(self, questions: List[str], solutions_pos: List[str], solutions_neg: List[str], rm_config: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        rm_config = rm_config if rm_config is not None else self.rm_config
        pad_to_mult8 = bool(rm_config.get("pad_to_multiple_of_8"))
        solutions_pos = [self.clear_solution(s) for s in solutions_pos]
        solutions_neg = [self.clear_solution(s) for s in solutions_neg]
        texts: List[str] = []
        for q, p, n in zip(questions, solutions_pos, solutions_neg):
            texts.append(self._chat(q, p))
            texts.append(self._chat(q, n))
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
        logits_sorted = self._forward_logits(enc, grad_enabled=True, model=self.model)  # (2B, C)
        order_tensor = torch.tensor(order, device=logits_sorted.device, dtype=torch.long)
        inv = torch.argsort(order_tensor)
        original_logits = logits_sorted[inv]
        r_pos = original_logits[0::2, 1]
        r_neg = original_logits[1::2, 1]
        return r_pos, r_neg

    def train_step(self, triplets: List[Tuple[str, str, str]]) -> float:
        # if hasattr(self.model, "gradient_checkpointing_enable"):
        #     self.model.gradient_checkpointing_enable()
        self.model.train()
        batch_size = self.pair_batch_size
        total_loss = 0.0
        num_batches = 0
        accum_steps = self.grad_accum
        self.optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(triplets), batch_size):
            end = min(start + batch_size, len(triplets))
            batch = triplets[start:end]
            batch_q = [t[0] for t in batch]
            batch_pos = [self.clear_solution(t[1]) for t in batch]
            batch_neg = [self.clear_solution(t[2]) for t in batch]
            r_pos, r_neg = self.score_pairs(batch_q, batch_pos, batch_neg, self.rm_config)
            loss_full = pairwise_rm_loss(r_pos, r_neg)
            total_loss += loss_full.detach().item()
            num_batches += 1
            (loss_full / accum_steps).backward()
            if num_batches % accum_steps == 0 or end == len(triplets):
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            del r_pos, r_neg, loss_full, batch_q, batch_pos, batch_neg, batch
        torch.cuda.empty_cache()
        return total_loss / num_batches if num_batches else 0.0

    @staticmethod
    def clear_solution(full_solution: str) -> str:
        if '</think>' in full_solution:
            return full_solution[:full_solution.index('</think>')] + '</think>'
        if len(full_solution) < 50:
            return '</think>'
        ind = full_solution[:-50].rfind("\n")
        if ind == -1:
            return '</think>'
        return full_solution[:ind] + '</think>'


def load_reward_model(model_name: str, gpu_id: int, rm_config: Optional[dict] = None, num_steps: Optional[int] = None) -> AceMathRewardModel:
    return AceMathRewardModel(model_name, gpu_id, rm_config, num_steps)


def last_token_index(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    mask = (input_ids == token_id)
    flipped = torch.flip(mask, dims=[1]).int().argmax(dim=1)
    return (input_ids.shape[1] - 1) - flipped
