import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import torch
from sklearn.metrics import roc_auc_score
import pandas as pd  # added for DataFrame export

try:
    from .prompting import build_prompts
    from .answer_parse import compute_final_correctness
except:
    from prompting import build_prompts
    from answer_parse import compute_final_correctness


def _select_test_subset(test_ds, q_field: str, a_field: str, max_questions: int) -> Tuple[List[str], List[str]]:
    total = len(test_ds)
    use_n = min(total, max_questions)
    records = [test_ds[i] for i in range(use_n)]
    questions = [r[q_field] for r in records]
    gold = [r[a_field] for r in records]
    return questions, gold


def _normalize_correctness(row: List[int]) -> List[int]:
    # Remove -1 ambiguous items for accuracy calculations
    return [v for v in row if v in (0, 1)]


def _per_question_accuracy(correctness: List[List[int]]) -> float:
    per_q: List[float] = []
    for row in correctness:
        filt = _normalize_correctness(row)
        if not filt:
            continue
        per_q.append(sum(filt) / len(filt))
    return sum(per_q) / len(per_q) if per_q else 0.0


def _average_auc(rm_scores: torch.Tensor, correctness: List[List[int]]) -> float:
    aucs: List[float] = []
    for qi in range(rm_scores.shape[0]):
        scores_row = rm_scores[qi].detach().cpu().tolist()
        labels_row = correctness[qi]
        scores: List[float] = []
        labels: List[int] = []
        for s, l in zip(scores_row, labels_row):
            if l in (0, 1) and s == s:  # filter ambiguous and NaN
                scores.append(float(s))
                labels.append(int(l))
        if len(set(labels)) < 2:
            continue  # need both classes for ROC AUC
        try:
            auc = roc_auc_score(labels, scores)
            aucs.append(float(auc))
        except Exception:
            continue
    return sum(aucs) / len(aucs) if aucs else 0.0


def _percent_ambiguous(correctness: List[List[int]]) -> float:
    """Return percentage (0-100) of ambiguous (-1) correctness entries across all samples."""
    total = sum(len(row) for row in correctness)
    if total == 0:
        return 0.0
    ambiguous = sum(1 for row in correctness for v in row if v == -1)
    return (ambiguous / total)


def filter_and_select_mixed(
    questions: List[str],
    gold_answers: List[str],
    candidate_texts: List[List[str]],
    candidate_valid_flags: List[List[int]],
    correctness: List[List[int]],
) -> Tuple[List[str], List[str], List[List[str]], List[List[int]]]:
    """Replicated from train.py. Remove invalid-but-correct candidates and retain only questions
    that have mixed correctness (contain both 0 and 1). Returns filtered sets or empty lists if none remain."""
    filtered_candidate_texts: List[List[str]] = []
    filtered_correctness: List[List[int]] = []
    for texts_row, flags_row, corr_row in zip(candidate_texts, candidate_valid_flags, correctness):
        new_texts: List[str] = []
        new_corr: List[int] = []
        for t, f, corr in zip(texts_row, flags_row, corr_row):
            if corr == -1 or (f == 0 and corr == 1):
                continue
            new_texts.append(t)
            new_corr.append(corr)
        filtered_candidate_texts.append(new_texts)
        filtered_correctness.append(new_corr)

    mixed_indices: List[int] = []
    for i, row in enumerate(filtered_correctness):
        vals = set(row)
        if 1 in vals and 0 in vals:
            mixed_indices.append(i)
    if not mixed_indices:
        return [], [], [], []

    questions_f = [questions[i] for i in mixed_indices]
    gold_answers_f = [gold_answers[i] for i in mixed_indices]
    candidates_f = [filtered_candidate_texts[i] for i in mixed_indices]
    correctness_f = [filtered_correctness[i] for i in mixed_indices]
    return questions_f, gold_answers_f, candidates_f, correctness_f


async def evaluate_greedy(engine, test_ds, q_field: str, a_field: str, tokenizer, generation_config: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    total = len(test_ds)
    questions = [test_ds[i][q_field] for i in range(total)]
    gold_answers = [test_ds[i][a_field] for i in range(total)]
    prompts = build_prompts(questions, tokenizer)
    greedy_cfg = dict(generation_config)
    # Override for greedy decoding
    greedy_cfg['think_temperature'] = 0.0
    greedy_cfg['think_top_p'] = 1.0
    greedy_cfg['think_top_k'] = 1
    greedy_cfg['think_repetition_penalty'] = 1.0

    raw_candidates = await engine.generate_candidates(prompts, n_samples=1, **greedy_cfg)
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


async def evaluate_sampling(engine, rm_model, test_ds, q_field: str, a_field: str, tokenizer, generation_config: Dict[str, Any], evaluation_config: Dict[str, Any], rm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    n_samples = int(evaluation_config.get('n_samples_per_problem'))
    total = len(test_ds)
    batch_size = int(evaluation_config.get('sampling_batch_size', total))
    if batch_size <= 0:
        batch_size = total

    # Accumulate dynamically after filtering rather than preallocating tensors (since counts shrink)
    all_questions: List[str] = []
    all_gold_answers: List[str] = []
    all_candidate_texts: List[List[str]] = []
    all_correctness: List[List[int]] = []
    all_pass1: List[int] = []
    rm_scores_rows: List[torch.Tensor] = []
    rm_scores_ref_rows: List[torch.Tensor] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_questions = [test_ds[i][q_field] for i in range(start, end)]
        batch_gold = [test_ds[i][a_field] for i in range(start, end)]
        prompts = build_prompts(batch_questions, tokenizer)
        raw_candidates = await engine.generate_candidates(prompts, n_samples=n_samples, **generation_config)
        batch_candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        batch_valid_flags = [[c[1] for c in row] for row in raw_candidates]
        batch_correctness = compute_final_correctness(batch_candidate_texts, batch_gold)
        batch_pass1 = [any(c == 1 for c in cs) for cs in batch_correctness]

        # Apply filtering + mixed selection (new)
        batch_questions_f, batch_gold_f, batch_candidate_texts_f, batch_correctness_f = filter_and_select_mixed(
            batch_questions, batch_gold, batch_candidate_texts, batch_valid_flags, batch_correctness
        )
        # Reward scoring on filtered batch
        try:
            batch_rm_scores_model, batch_rm_scores_ref = rm_model.score_reference(batch_questions_f, batch_candidate_texts_f, rm_config)
        except Exception as e:
            print(f"[Eval Sampling] RM scoring exception on filtered batch {start}:{end}: {e}; retry small batch.")
            torch.cuda.empty_cache()
            batch_rm_scores_model, batch_rm_scores_ref = rm_model.score_reference(batch_questions_f, batch_candidate_texts_f, rm_config, forced_small_batch_size=True)
        torch.cuda.empty_cache()

        # Accumulate
        all_questions.extend(batch_questions_f)
        all_gold_answers.extend(batch_gold_f)
        all_candidate_texts.extend(batch_candidate_texts_f)
        all_correctness.extend(batch_correctness_f)
        all_pass1.extend(batch_pass1)
        rm_scores_rows.extend([row.detach().cpu() for row in batch_rm_scores_model])
        rm_scores_ref_rows.extend([row.detach().cpu() for row in batch_rm_scores_ref])
        del raw_candidates, batch_candidate_texts, batch_valid_flags, batch_rm_scores_model, batch_rm_scores_ref

    if not all_questions:
        # Return empty-style result if no mixed questions found at all
        return {
            'mode': 'sampling',
            'num_questions': 0,
            'n_samples_per_question': n_samples,
            'avg_accuracy': 0.0,
            'avg_auc': 0.0,
            'avg_auc_ref': 0.0,
            'percent_minus_one': 0.0,
            'note': 'No mixed correctness questions after filtering.'
        }

    # Build padded tensors for AUC computations
    max_k = max(len(row) for row in all_candidate_texts)
    rm_scores = torch.empty(len(all_questions), max_k, dtype=torch.float32).fill_(float('nan'))
    rm_scores_ref = torch.empty(len(all_questions), max_k, dtype=torch.float32).fill_(float('nan'))
    for i, (scores_row, scores_ref_row) in enumerate(zip(rm_scores_rows, rm_scores_ref_rows)):
        k = min(max_k, scores_row.shape[0])
        rm_scores[i, :k] = scores_row[:k]
        rm_scores_ref[i, :k] = scores_ref_row[:k]

    avg_acc = _per_question_accuracy(all_correctness)
    avg_auc = _average_auc(rm_scores, all_correctness)
    avg_auc_ref = _average_auc(rm_scores_ref, all_correctness)
    amb_pct = _percent_ambiguous(all_correctness)

    # Build and persist a DataFrame with per-question details
    details_rows: List[Dict[str, Any]] = []
    for i, q in enumerate(all_questions):
        bool_correctness = [v == 1 for v in all_correctness[i]]
        details_rows.append({
            'question': q,
            'candidates': all_candidate_texts[i],
            'correct_inccorect': bool_correctness,  # keeping original column name spelling
            'rm_scores': [float(x) if x == x else None for x in rm_scores[i].tolist()],
            'rm_rf_scores': [float(x) if x == x else None for x in rm_scores_ref[i].tolist()],
        })

    df = pd.DataFrame(details_rows, columns=[
        'question', 'candidates', 'correct_inccorect', 'rm_scores', 'rm_rf_scores',
    ])
    os.makedirs('evaluation_logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_path = os.path.join('evaluation_logs', f'sampling_details_{ts}.parquet')
    try:
        df.to_parquet(df_path, index=False)
    except Exception:
        df_path = os.path.join('evaluation_logs', f'sampling_details_{ts}.csv')
        df.to_csv(df_path, index=False)
    print(f'len of all_pass1: {len(all_pass1)}')
    return {
        'mode': 'sampling',
        'num_questions': len(all_questions),
        'n_samples_per_question': n_samples,
        'avg_accuracy': avg_acc,
        'avg_auc': avg_auc,
        'avg_auc_ref': avg_auc_ref,
        'percent_minus_one': amb_pct,
        'pass1_only': sum(all_pass1) / len(all_pass1),
    }


def log_evaluation(result: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(out_dir, f"eval_{result.get('mode')}_{ts}.json")
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


def merge_eval_results(*results: Dict[str, Any]) -> Dict[str, Any]:
    out = {'timestamp': datetime.now().isoformat(), 'results': results}
    return out

async def run_full_evaluation(engine, rm_model, test_ds, q_field: str, a_field: str, tokenizer, generation_config: Dict[str, Any], evaluation_config: Dict[str, Any], rm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    greedy_res, sampling_res = await asyncio.gather(
        evaluate_greedy(engine, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config),
        evaluate_sampling(engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config, rm_config)
    )
    combined = merge_eval_results(greedy_res, sampling_res)
    log_evaluation(greedy_res, out_dir='evaluation_logs')
    log_evaluation(sampling_res, out_dir='evaluation_logs')
    return combined
