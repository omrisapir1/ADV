from __future__ import annotations

from typing import List, Dict, Any, Optional

# Optional imports with fallbacks (mirror pattern used in generation_sync)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from sklearn.metrics import roc_auc_score  # type: ignore
except Exception:  # pragma: no cover
    def roc_auc_score(*_args, **_kwargs):  # type: ignore
        return 0.0
try:
    from .prompting import build_prompts
    from .answer_parse import compute_final_correctness
    from .evaluation import merge_eval_results, log_evaluation  # reuse existing helpers
except:
    from prompting import build_prompts
    from answer_parse import compute_final_correctness
    from evaluation import merge_eval_results, log_evaluation  # reuse existing helpers

# --- Utility helpers duplicated (kept private) to avoid relying on async versions ---

def _normalize_correctness(row: List[int]) -> List[int]:
    return [v for v in row if v in (0, 1)]


def _per_question_accuracy(correctness: List[List[int]]) -> float:
    per_q: List[float] = []
    for row in correctness:
        filt = _normalize_correctness(row)
        if not filt:
            continue
        per_q.append(sum(filt) / len(filt))
    return sum(per_q) / len(per_q) if per_q else 0.0


def _average_auc(rm_scores, correctness: List[List[int]]) -> float:
    if torch is None:  # fallback if torch absent
        return 0.0
    aucs: List[float] = []
    for qi in range(rm_scores.shape[0]):
        scores_row = rm_scores[qi].detach().cpu().tolist()
        labels_row = correctness[qi]
        scores: List[float] = []
        labels: List[int] = []
        for s, l in zip(scores_row, labels_row):
            if l in (0, 1) and s == s:
                scores.append(float(s))
                labels.append(int(l))
        if len(set(labels)) < 2:
            continue
        try:
            auc = roc_auc_score(labels, scores)
            aucs.append(float(auc))
        except Exception:
            continue
    return sum(aucs) / len(aucs) if aucs else 0.0


def _percent_ambiguous(correctness: List[List[int]]) -> float:
    total = sum(len(row) for row in correctness)
    if total == 0:
        return 0.0
    ambiguous = sum(1 for row in correctness for v in row if v == -1)
    return ambiguous / total

# --- Greedy evaluation (serial) ---

def evaluate_greedy_sync(engine, test_ds, q_field: str, a_field: str, tokenizer, generation_config: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    total = len(test_ds)
    questions = [test_ds[i][q_field] for i in range(total)]
    gold_answers = [test_ds[i][a_field] for i in range(total)]
    prompts = build_prompts(questions, tokenizer)
    greedy_cfg = dict(generation_config)
    greedy_cfg['think_temperature'] = 0.0
    greedy_cfg['think_top_p'] = 1.0
    greedy_cfg['think_top_k'] = 1
    greedy_cfg['think_repetition_penalty'] = 1.0

    raw_candidates = engine.generate_candidates(prompts, n_samples=1, **greedy_cfg)
    candidate_texts = [[c[0] for c in row] for row in raw_candidates]
    correctness = compute_final_correctness(candidate_texts, gold_answers)
    acc = _per_question_accuracy(correctness)
    amb_pct = _percent_ambiguous(correctness)
    return {
        'mode': 'greedy',
        'num_questions': len(questions),
        'accuracy': acc,
        'percent_minus_one': amb_pct
    }

# --- Sampling evaluation (serial) ---

def evaluate_sampling_sync(engine, rm_model, test_ds, q_field: str, a_field: str, tokenizer, generation_config: Dict[str, Any], evaluation_config: Dict[str, Any], rm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    n_samples = int(evaluation_config.get('n_samples_per_problem'))
    total = len(test_ds)
    batch_size = int(evaluation_config.get('sampling_batch_size', total))
    if batch_size <= 0:
        batch_size = total

    all_questions: List[str] = []
    all_gold_answers: List[str] = []
    all_candidate_texts: List[List[str]] = []
    if torch is not None:
        rm_scores = torch.empty(total, n_samples, dtype=torch.float32).fill_(float('nan'))
    else:
        # Fallback simple 2D list placeholder for shape compatibility in _average_auc
        rm_scores = [[float('nan')] * n_samples for _ in range(total)]
        class _ShapeWrapper:
            def __init__(self, data):
                self._data = data
            @property
            def shape(self):
                return (len(self._data), len(self._data[0]) if self._data else 0)
            def __getitem__(self, idx):
                return self._data[idx]
        rm_scores = _ShapeWrapper(rm_scores)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_questions = [test_ds[i][q_field] for i in range(start, end)]
        batch_gold = [test_ds[i][a_field] for i in range(start, end)]
        prompts = build_prompts(batch_questions, tokenizer)
        raw_candidates = engine.generate_candidates(prompts, n_samples=n_samples, **generation_config)
        batch_candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        all_questions.extend(batch_questions)
        all_gold_answers.extend(batch_gold)
        all_candidate_texts.extend(batch_candidate_texts)
        if torch is not None:
            try:
                batch_rm_scores = rm_model.score_reference(batch_questions, batch_candidate_texts, rm_config)
            except Exception as e:
                print(f"[Eval Sampling Sync] RM scoring exception on batch {start}:{end}: {e}; retry small batch.")
                torch.cuda.empty_cache()
                batch_rm_scores = rm_model.score_reference(batch_questions, batch_candidate_texts, rm_config, forced_small_batch_size=True)
            torch.cuda.empty_cache()
            b_rows, b_cols = batch_rm_scores.shape
            rm_scores[start:start + b_rows, :b_cols] = batch_rm_scores.detach().to(dtype=torch.float32, device='cpu')
            del batch_rm_scores, raw_candidates, batch_candidate_texts

    correctness = compute_final_correctness(all_candidate_texts, all_gold_answers)
    avg_acc = _per_question_accuracy(correctness)
    avg_auc = _average_auc(rm_scores, correctness)
    amb_pct = _percent_ambiguous(correctness)
    return {
        'mode': 'sampling',
        'num_questions': len(all_questions),
        'n_samples_per_question': n_samples,
        'avg_accuracy': avg_acc,
        'avg_auc': avg_auc,
        'percent_minus_one': amb_pct,
        'sampling_batch_size': batch_size
    }

# --- Public main entry ---

def run_full_evaluation_sync(engine, rm_model, test_ds, q_field: str, a_field: str, tokenizer, generation_config: Dict[str, Any], evaluation_config: Dict[str, Any], rm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Serial (non-async) counterpart to run_full_evaluation.

    Returns identical structure: {'timestamp': ..., 'results': (greedy_res, sampling_res)}
    and logs each evaluation JSON to evaluation_logs/ like the async version.
    """
    greedy_res = evaluate_greedy_sync(engine, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config)
    sampling_res = evaluate_sampling_sync(engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config, rm_config)
    combined = merge_eval_results(greedy_res, sampling_res)
    log_evaluation(greedy_res, out_dir='evaluation_logs')
    log_evaluation(sampling_res, out_dir='evaluation_logs')
    return combined
