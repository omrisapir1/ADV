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

    all_questions: List[str] = []
    all_gold_answers: List[str] = []
    all_candidate_texts: List[List[str]] = []
    # rm_scores = torch.empty(total, n_samples, dtype=torch.float32).fill_(float('nan'))
    # rm_scores_ref = torch.empty(total, n_samples, dtype=torch.float32).fill_(float('nan'))  # new tensor for reference model

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_questions = [test_ds[i][q_field] for i in range(start, end)]
        batch_gold = [test_ds[i][a_field] for i in range(start, end)]
        prompts = build_prompts(batch_questions, tokenizer)
        raw_candidates = await engine.generate_candidates(prompts, n_samples=n_samples, **generation_config)
        batch_candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        # Append
        all_questions.extend(batch_questions)
        all_gold_answers.extend(batch_gold)
        all_candidate_texts.extend(batch_candidate_texts)
        # Reward scoring per batch
        # try:
        # batch_rm_scores_model, batch_rm_scores_ref = rm_model.score_reference(batch_questions, batch_candidate_texts, rm_config)
        # except Exception as e:
        #     print(f"[Eval Sampling] RM scoring exception on batch {start}:{end}: {e}; retry small batch.")
        #     torch.cuda.empty_cache()
        #     batch_rm_scores_model, batch_rm_scores_ref = rm_model.score_reference(batch_questions, batch_candidate_texts, rm_config, forced_small_batch_size=True)
        torch.cuda.empty_cache()
        # b_rows, b_cols = batch_rm_scores_model.shape
        # rm_scores[start:start + b_rows, :b_cols] = batch_rm_scores_model.detach().to(dtype=torch.float32, device='cpu')
        # rm_scores_ref[start:start + b_rows, :b_cols] = batch_rm_scores_ref.detach().to(dtype=torch.float32, device='cpu')  # fill reference scores
        # del batch_rm_scores_model, batch_rm_scores_ref, raw_candidates, batch_candidate_texts
        del raw_candidates, batch_candidate_texts

    correctness = compute_final_correctness(all_candidate_texts, all_gold_answers)
    avg_acc = _per_question_accuracy(correctness)
    # avg_auc = _average_auc(rm_scores, correctness)
    # avg_auc_ref = _average_auc(rm_scores_ref, correctness)  # second AUC
    amb_pct = _percent_ambiguous(correctness)

    # Build and persist a DataFrame with per-question details
    details_rows: List[Dict[str, Any]] = []
    for i, q in enumerate(all_questions):
        bool_correctness = [v == 1 for v in correctness[i]]
        details_rows.append({
            'question': q,
            'candidates': all_candidate_texts[i],
            # user requested column name with typo: correct_inccorect
            'correct_inccorect': bool_correctness,
            # keep previous correct spelling as alias for possible downstream use
            'correct_incorrect': bool_correctness,
            # 'rm_scores': [float(x) if x == x else None for x in rm_scores[i].tolist()],
            # user requested rm_rf_scores (interpreted as ref scores); provide both names
            # 'rm_rf_scores': [float(x) if x == x else None for x in rm_scores_ref[i].tolist()],
            # 'rm_ref_scores': [float(x) if x == x else None for x in rm_scores_ref[i].tolist()],
        })
    # DataFrame will include all columns; user-specified ordering first
    df = pd.DataFrame(details_rows, columns=[
        'question', 'candidates', 'correct_inccorect', 'rm_scores', 'rm_rf_scores',
        'correct_incorrect', 'rm_ref_scores'
    ])
    os.makedirs('evaluation_logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_path = os.path.join('evaluation_logs', f'sampling_details_{ts}.parquet')
    try:
        df.to_parquet(df_path, index=False)
    except Exception:
        # fallback to csv if parquet engine unavailable
        df_path = os.path.join('evaluation_logs', f'sampling_details_{ts}.csv')
        df.to_csv(df_path, index=False)

    return {
        'mode': 'sampling',
        'num_questions': len(all_questions),
        'n_samples_per_question': n_samples,
        'avg_accuracy': avg_acc,
        # 'avg_auc': avg_auc,
        # 'avg_auc_ref': avg_auc_ref,
        'percent_minus_one': amb_pct,
        'sampling_batch_size': batch_size,
        'details_dataframe_path': df_path
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
    await evaluate_sampling(engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config,
                      rm_config)
    # greedy_res, sampling_res = await asyncio.gather(
    #     evaluate_greedy(engine, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config),
    #     evaluate_sampling(engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config, rm_config)
    # )
    # combined = merge_eval_results(greedy_res, sampling_res)
    # log_evaluation(greedy_res, out_dir='evaluation_logs')
    # log_evaluation(sampling_res, out_dir='evaluation_logs')
    # return combined
