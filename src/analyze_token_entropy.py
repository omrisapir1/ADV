"""
Analyze per-token entropy of the model's reasoning ("think" trace) comparing:
  1. SGLang server sampling path (using build_sglang_engine + generate_candidates).
  2. Direct HuggingFace forward logits (raw distribution; no sampling transforms).

Output: CSV + printed pandas DataFrame with columns:
  question | model_answer | entropy_sglang | entropy_trainer | num_tokens | raw_think_section

Notes:
- We only process the "thinking" phase up to the stop tag '</think>'.
- Engine wrapper does not expose raw server logprobs; we reconstruct per-step distributions
  locally with the HF model applying generation config (temperature, top-p, top-k, repetition penalty).
- Direct HF path computes entropy from unmodified logits (pre temperature / filtering).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd

# Absolute imports (repo root on PYTHONPATH); friendlier for notebook copy-paste
from src.train import load_config, load_dataset_handle, get_batch_records
from src.prompting import build_prompts
from src.llm_trainer import load_llm_trainer
from src.generation import THINK_STOP, build_sglang_engine

# ---------------------------------------------------------------------------
@dataclass
class EntropyAnalysisConfig:
    config_yaml_path: str = "configs/config.yaml"
    batches: int = 2            # number of batches to sample
    batch_size: int = 2         # questions per batch
    save_csv_path: str = "token_entropy_analysis.csv"
    device: Optional[str] = None  # override device for HF model

# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy H_t = -sum_v p_t(v) log p_t(v) from logits (seq, vocab)."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=-1)  # (seq,)

@torch.no_grad()
def apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], penalty: float) -> None:
    if penalty is None or penalty == 1.0 or not generated_ids:
        return
    for token_id in set(generated_ids):
        val = logits[token_id]
        if val > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

@torch.no_grad()
def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if not top_k or top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    masked = torch.full_like(logits, float('-inf'))
    masked[indices] = values
    return masked

@torch.no_grad()
def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not top_p or top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    keep_mask = cum <= top_p
    if not torch.any(keep_mask):
        keep_mask[0] = True
    keep_indices = sorted_indices[keep_mask]
    new_logits = torch.full_like(logits, float('-inf'))
    new_logits[keep_indices] = logits[keep_indices]
    return new_logits

# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_sglang_style_entropies(think_text: str, prompt: str, tokenizer, model, gen_cfg: Dict[str, Any]) -> Tuple[float, List[float], int]:
    """Reconstruct per-step sampling distribution for think_text tokens using HF model + generation cfg."""
    full_text = prompt + think_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    gen_ids = full_ids[len(prompt_ids):]
    if not gen_ids:
        return float('nan'), [], 0

    temperature = gen_cfg.get("think_temperature", 1.0) or 1.0
    top_p = gen_cfg.get("think_top_p", 1.0)
    top_k = gen_cfg.get("think_top_k", 0)
    repetition_penalty = gen_cfg.get("think_repetition_penalty", 1.0)

    entropies: List[float] = []
    generated_so_far: List[int] = []
    device = next(model.parameters()).device
    for next_id in gen_ids:
        prefix_ids = prompt_ids + generated_so_far
        inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp, device=device)
        logits = model(input_ids=inp, attention_mask=attn).logits[0, -1, :].float()
        logits = logits / temperature
        apply_repetition_penalty(logits, generated_so_far, repetition_penalty)
        if top_k and top_k > 0:
            logits = apply_top_k(logits, top_k)
        logits = apply_top_p(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        entropy_val = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        entropies.append(entropy_val)
        generated_so_far.append(next_id)
    mean_entropy = float(sum(entropies) / len(entropies)) if entropies else float('nan')
    return mean_entropy, entropies, len(gen_ids)

@torch.no_grad()
def compute_hf_raw_entropies_for_think_text(think_text: str, prompt: str, tokenizer, model) -> Tuple[float, List[float], int]:
    """Compute raw (untransformed) HF per-token entropies for the provided think_text tokens.
    Uses the same token sequence produced by SGLang for fair comparison.
    """
    # Tokenize full sequence (prompt + think_text) to extract generated token ids
    full_text = prompt + think_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    gen_ids = full_ids[len(prompt_ids):]
    if not gen_ids:
        return float('nan'), [], 0
    entropies: List[float] = []
    device = next(model.parameters()).device
    # Iteratively feed prefixes to get next-token distribution entropy
    prefix_ids = prompt_ids.copy()
    for token_id in gen_ids:
        inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp, device=device)
        logits = model(input_ids=inp, attention_mask=attn).logits[0, -1, :].float()
        entropy_val = float(compute_token_entropy(logits.unsqueeze(0))[0].item())
        entropies.append(entropy_val)
        prefix_ids.append(token_id)
    mean_entropy = float(sum(entropies) / len(entropies)) if entropies else float('nan')
    return mean_entropy, entropies, len(gen_ids)

# ---------------------------------------------------------------------------
def run_entropy_analysis(cfg: EntropyAnalysisConfig):
    base_cfg = load_config(cfg.config_yaml_path)
    llm_name = base_cfg["model"]["llm_name"]
    generation_cfg = base_cfg.get("generation", {})
    n_samples_per_problem = base_cfg.get("train", {}).get("n_samples_per_problem", 1)

    # Dataset
    train_ds, _test_ds, q_field, _a_field = load_dataset_handle(base_cfg)

    # HF model via trainer (reuse tokenizer & weights)
    trainer = load_llm_trainer(llm_name, base_cfg["hardware"].get("llm_trainer_gpu_id", 0), 0, base_cfg.get("llm_trainer"))
    model = trainer.model
    tokenizer = trainer.tokenizer
    if cfg.device:
        model.to(cfg.device)
    model.eval()

    # SGLang engine for actual generation of think phase (one sample each)
    engine = build_sglang_engine(llm_name, generation_cfg)

    rows: List[Dict[str, Any]] = []
    total = 0
    for batch_idx in range(cfg.batches):
        batch_records = get_batch_records(train_ds, cfg.batch_size, batch_idx)
        questions = [r[q_field] for r in batch_records]
        prompts = build_prompts(questions, tokenizer)

        # Use same generation signature as train.py (multiple samples per problem)
        raw_candidates_nested = None
        try:
            import asyncio
            raw_candidates_nested = asyncio.run(engine.generate_candidates(prompts, n_samples=n_samples_per_problem, **generation_cfg))
        except RuntimeError:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            raw_candidates_nested = loop.run_until_complete(engine.generate_candidates(prompts, n_samples=n_samples_per_problem, **generation_cfg))
            loop.close()

        # Iterate per question and per sample
        for q_idx, (q, prompt, cand_list) in enumerate(zip(questions, prompts, raw_candidates_nested)):
            if not cand_list:
                rows.append({
                    "question": q,
                    "sample_idx": -1,
                    "model_answer": "",
                    "entropy_sglang": float('nan'),
                    "entropy_trainer": float('nan'),
                    "num_tokens": 0,
                    "raw_think_section": "",
                    "sglang_per_token_entropies": [],
                    "trainer_per_token_entropies": [],
                    "top_logprobs_sequence": [],
                })
                total += 1
                continue
            for sample_idx, candidate in enumerate(cand_list):
                # candidate structure: (full_text, phase_flag, top_logprobs_sequence)
                try:
                    full_text, phase_flag, top_lp_seq = candidate
                except Exception:
                    # Backward compatibility fallback
                    full_text = candidate[0]
                    phase_flag = candidate[1] if len(candidate) > 1 else 0
                    top_lp_seq = []
                think_part = full_text.split(THINK_STOP, 1)[0] if THINK_STOP in full_text else full_text
                try:
                    mean_ent_sg, entropies_sg, num_tokens_sg = compute_sglang_style_entropies(think_part, prompt, tokenizer, model, generation_cfg)
                except Exception as e:
                    mean_ent_sg, entropies_sg, num_tokens_sg = float('nan'), [], 0
                    print(f"[Batch {batch_idx} Q{q_idx} Sample{sample_idx}] SGLang-style entropy error: {e}")
                try:
                    mean_ent_hf, entropies_hf, num_tokens_hf = compute_hf_raw_entropies_for_think_text(think_part, prompt, tokenizer, model)
                except Exception as e:
                    mean_ent_hf, entropies_hf, num_tokens_hf = float('nan'), [], 0
                    print(f"[Batch {batch_idx} Q{q_idx} Sample{sample_idx}] HF raw entropy error: {e}")
                rows.append({
                    "question": q,
                    "sample_idx": sample_idx,
                    "model_answer": full_text,
                    "entropy_sglang": mean_ent_sg,
                    "entropy_trainer": mean_ent_hf,
                    "num_tokens": num_tokens_sg,
                    "raw_think_section": think_part,
                    "sglang_per_token_entropies": entropies_sg,
                    "trainer_per_token_entropies": entropies_hf,
                    "top_logprobs_sequence": top_lp_seq,
                })
                total += 1
        print(f"[Batch {batch_idx}] Processed {len(questions)} questions with {n_samples_per_problem} samples each. Total rows: {total}")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = EntropyAnalysisConfig()
    df = run_entropy_analysis(cfg)
