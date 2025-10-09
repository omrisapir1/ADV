from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class AceMathRewardModel:
    def __init__(self, model_name: str, gpu_id: int):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device

        # Fast(er) tokenizer behavior
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            # fall back to eos if pad is missing
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        # Right padding tends to be friendlier for many kernels
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        # Move the entire model to the specific device
        self.model = self.model.to(self.device)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Optional: can help a bit when shapes repeat
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)  # PyTorch 2.x
        except Exception:
            pass

    def _chat_of(self, question: str, solution: str):
        return [
            {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]

    def build_chat_inputs_ids_only(self, question: str, solution: str) -> Tuple[list, int]:
        """
        Returns (input_ids_list, length) without padding.
        We tokenize via apply_chat_template(tokenize=True) to avoid re-parsing prompts later.
        """
        ids = self.tokenizer.apply_chat_template(
            self._chat_of(question, solution),
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,        # keep if your RM has a max length
            return_tensors=None,    # -> list[int]
        )
        return ids, len(ids)


def load_reward_model(model_name: str, gpu_id: int) -> AceMathRewardModel:
    return AceMathRewardModel(model_name=model_name, gpu_id=gpu_id)


def _pad_to_multiple_of_8(batch):
    # Optional: round sequence length up to multiple-of-8 to improve tensor-core utilization
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    L = input_ids.size(1)
    L8 = (L + 7) // 8 * 8
    if L8 == L:
        return batch
    pad_len = L8 - L
    pad_id = batch.get("pad_token_id", 0)
    pad_ids = torch.full((input_ids.size(0), pad_len), pad_id, dtype=input_ids.dtype, device=input_ids.device)
    pad_mask = torch.zeros((attn.size(0), pad_len), dtype=attn.dtype, device=attn.device)
    batch["input_ids"] = torch.cat([input_ids, pad_ids], dim=1)
    batch["attention_mask"] = torch.cat([attn, pad_mask], dim=1)
    return batch


def score_solutions(
    questions: List[str],
    solutions: List[str],
    model: AceMathRewardModel,
    n_candidates: int,
    rm_config: dict = None
) -> torch.Tensor:
    if len(solutions) != len(questions) * n_candidates:
        raise ValueError("Mismatch between flattened solutions and expected shape")

    rm_config = rm_config or {}
    # You can keep your old "batch size by items" or switch to a token budget:
    batch_size_items = rm_config.get("reference_batch_size")
    tokens_per_batch = rm_config.get("rm_reference_tokens_per_batch", None)  # e.g., 80000 on 48–80GB GPUs

    # ---- 1) Pre-tokenize ALL pairs once (fast & avoids per-step overhead) ----
    ids_and_lens = []
    ids_and_meta = []
    for qi, q in enumerate(questions):
        base = qi * n_candidates
        for k in range(n_candidates):
            ids, L = model.build_chat_inputs_ids_only(q, solutions[base + k])
            ids_and_lens.append(L)
            ids_and_meta.append((qi, k, ids))

    if not ids_and_meta:
        return torch.empty(len(questions), n_candidates)

    # ---- 2) Length-aware ordering (descending) to reduce padding ----
    order = sorted(range(len(ids_and_meta)), key=lambda i: ids_and_lens[i], reverse=True)
    ids_and_meta = [ids_and_meta[i] for i in order]
    lengths = [ids_and_lens[i] for i in order]

    # Helper to yield batches either by item count or token budget
    def batch_indices():
        if tokens_per_batch is None:
            for start in range(0, len(ids_and_meta), batch_size_items):
                yield range(start, min(start + batch_size_items, len(ids_and_meta)))
        else:
            start = 0
            N = len(ids_and_meta)
            while start < N:
                budget = 0
                end = start
                while end < N and budget + lengths[end] <= tokens_per_batch:
                    budget += lengths[end]
                    end += 1
                if end == start:  # single very long item
                    end = start + 1
                yield range(start, end)
                start = end

    device = next(model.model.parameters()).device
    logits_out = torch.empty(len(ids_and_meta), dtype=torch.float32, device=device)

    # ---- 3) Minimize host<->device stalls; use AMP + inference_mode ----
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for idx_range in batch_indices():
            # Build a list of dicts with unpadded ids so tokenizer.pad can pad efficiently
            item_dicts = []
            for idx in idx_range:
                _qi, _k, ids = ids_and_meta[idx]
                item_dicts.append({"input_ids": ids})

            batch = model.tokenizer.pad(
                item_dicts,
                padding=True,                # pad to longest in this batch
                return_tensors="pt"
            )
            # HF pad returns attention_mask automatically
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["pad_token_id"] = model.tokenizer.pad_token_id

            # Optional: pad length up to multiple-of-8 tokens
            batch = _pad_to_multiple_of_8(batch)

            out = model.model(**batch)
            logits = out.logits.squeeze(-1).to(dtype=torch.float32)  # (B,)

            # Write back into the global output in the length-sorted order
            b_indices = list(idx_range)
            logits_out[b_indices] = logits

    # ---- 4) Undo the sort to original (questions, candidates) layout ----
    # 'order' maps sorted positions -> original positions; build inverse permutation
    inv = torch.empty(len(order), dtype=torch.long, device=logits_out.device)
    inv[torch.tensor(order, device=inv.device)] = torch.arange(len(order), device=inv.device)
    logits_out = logits_out[inv]

    # Return shape (num_questions, n_candidates)
    return logits_out.view(len(questions), n_candidates).cpu()


def score_question_solution_list(
    questions: List[str],
    solutions: List[str],
    model: AceMathRewardModel,
    rm_config: dict | None = None,
) -> torch.Tensor:
    """Score aligned (question, solution) pairs with grad enabled.

    Returns tensor on model device retaining gradient history.
    """
    if len(questions) != len(solutions):
        raise ValueError("questions and solutions must be same length")

    if len(questions) == 0:
        return torch.empty(0)

    rm_config = rm_config or {}
    batch_size_items = rm_config.get("reference_batch_size") or 8
    tokens_per_batch = rm_config.get("rm_reference_tokens_per_batch", None)

    ids_and_lens = []
    ids_and_meta = []  # (idx, ids)
    for i, (q, s) in enumerate(zip(questions, solutions)):
        ids, L = model.build_chat_inputs_ids_only(q, s)
        ids_and_lens.append(L)
        ids_and_meta.append((i, ids))

    order = sorted(range(len(ids_and_meta)), key=lambda i: ids_and_lens[i], reverse=True)
    ids_and_meta = [ids_and_meta[i] for i in order]
    lengths = [ids_and_lens[i] for i in order]

    def batch_indices():
        if tokens_per_batch is None:
            for start in range(0, len(ids_and_meta), batch_size_items):
                yield range(start, min(start + batch_size_items, len(ids_and_meta)))
        else:
            start = 0
            N = len(ids_and_meta)
            while start < N:
                budget = 0
                end = start
                while end < N and budget + lengths[end] <= tokens_per_batch:
                    budget += lengths[end]
                    end += 1
                if end == start:
                    end = start + 1
                yield range(start, end)
                start = end

    device = next(model.model.parameters()).device
    logits_out = torch.empty(len(ids_and_meta), dtype=torch.float32, device=device)

    # Grad enabled pass (uses autocast for efficiency)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for idx_range in batch_indices():
            item_dicts = []
            for idx in idx_range:
                _orig_i, ids = ids_and_meta[idx]
                item_dicts.append({"input_ids": ids})
            batch = model.tokenizer.pad(item_dicts, padding=True, return_tensors="pt")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["pad_token_id"] = model.tokenizer.pad_token_id
            batch = _pad_to_multiple_of_8(batch)
            out = model.model(**batch)
            logits = out.logits.squeeze(-1).to(dtype=torch.float32)
            b_indices = list(idx_range)
            logits_out[b_indices] = logits

    # Invert ordering
    inv = torch.empty(len(order), dtype=torch.long, device=logits_out.device)
    inv[torch.tensor(order, device=inv.device)] = torch.arange(len(order), device=inv.device)
    logits_out = logits_out[inv]
    return logits_out  # keep on device with gradients
