import asyncio
import yaml
import torch
import json
import os
import random
import shutil
from datetime import datetime
import numpy as np


from typing import Dict, Any, List, Tuple, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from .prompting import build_prompts
from .generation import build_sglang_engine, EngineCircuitBreaker  # added EngineCircuitBreaker import
from .reward_model import load_reward_model
from .answer_parse import compute_final_correctness
from .llm_trainer import load_llm_trainer
from .optimizer import soft_reset_adam
from .evaluation import run_full_evaluation  # added import
from .alpha_control import AlphaControl  # moved to its own module

import time
import re
import requests

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_handle(cfg: Dict[str, Any]):
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name")
    ds = load_dataset(name)
    return ds['train'], ds['test'], ds_cfg.get("field_question", "problem"), ds_cfg.get("field_answer", "final_answer")

def get_batch(dataset: List[str], batch_size: int, step: int) -> List[str]:
    start = (step * batch_size) % len(dataset)
    return dataset[start:start + batch_size]


def get_batch_records(dataset_obj, batch_size: int, step: int) -> List[Dict[str, Any]]:
    if isinstance(dataset_obj, list):
        start = (step * batch_size) % len(dataset_obj)
        return dataset_obj[start:start + batch_size]
    # huggingface Dataset object
    total = len(dataset_obj)
    start = (step * batch_size) % total
    end = min(start + batch_size, total)
    return [dataset_obj[i] for i in range(start, end)]


LOG_DIR = "/workspace/ADV/src/data"  # central log directory path

def log_questions(questions: List[str], gold_answers: List[str], candidates: List[List[str]], rm_scores: torch.Tensor,
                  explore_scores: List[List[float]], entropy_scores: List[List[float]],
                  correctness: List[List[int]], rm_avg_loss, llm_avg_loss, pass1):
    """Log training results to disk in JSON format.

    Ensures all tensor / non-serializable types are converted to native Python types.
    """
    # Create logs directory if it doesn't exist
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")

    # Prepare log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "batch_size": len(questions),
        "questions": []
    }

    # Process each question in the batch
    for i, (question, gold_answer) in enumerate(zip(questions, gold_answers)):
        # Safely extract rm_scores for this question
        question_rm_scores: List[float] = []
        if i < len(rm_scores):
            row = rm_scores[i]
            # row could be a tensor of shape [num_candidates] or something iterable
            if isinstance(row, torch.Tensor):
                # Flatten if needed then convert
                question_rm_scores = row.detach().cpu().flatten().tolist()
            else:
                # Fallback: iterate and convert any tensor elements
                tmp = []
                for v in row:
                    if isinstance(v, torch.Tensor):
                        tmp.append(float(v.detach().cpu().item()))
                    else:
                        tmp.append(float(v))
                tmp = [float(x) for x in tmp]
                question_rm_scores = tmp

        # Safely extract explore and entropy scores for this question
        question_explore_scores: List[float] = []
        if i < len(explore_scores):
            question_explore_scores = [float(x) for x in explore_scores[i]]
        question_entropy_scores: List[float] = []
        if i < len(entropy_scores):
            question_entropy_scores = [float(x) for x in entropy_scores[i]]

        # Convert correctness list items to plain ints (0/1) / bools
        raw_corr = correctness[i] if i < len(correctness) else []
        corr_list: List[int] = []
        for c in raw_corr:
            if isinstance(c, torch.Tensor):
                # Assume 0-d tensor
                corr_list.append(int(c.item()))
            else:
                # bool or int
                corr_list.append(int(c))
        correct_count = int(sum(corr_list))

        question_data = {
            "question_id": i,
            "question": question,
            "gold_answer": gold_answer,
            "candidates": candidates[i] if i < len(candidates) else [],
            "rm_scores": question_rm_scores,
            "explore_scores": question_explore_scores,
            "entropy_scores": question_entropy_scores,
            "correctness": corr_list,
            "num_candidates": len(candidates[i]) if i < len(candidates) else 0,
            "avg_rm_score": float(sum(question_rm_scores) / len(question_rm_scores)) if question_rm_scores else 0.0,
            "avg_explore_score": float(sum(question_explore_scores) / len(question_explore_scores)) if question_explore_scores else 0.0,
            "avg_entropy_score": float(sum(question_entropy_scores) / len(question_entropy_scores)) if question_entropy_scores else 0.0,
            "correct_count": correct_count,
            'rm_avg_loss': rm_avg_loss,
            'llm_avg_loss': llm_avg_loss,
            "pass1": pass1,
        }

        # Skip serialization error printing; silently ignore failures
        try:
            json.dumps(question_data)
        except Exception:
            continue
        log_data["questions"].append(question_data)

    # Write to file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Silenced log output
    # print(f"Logged results to {log_file}")

# ---- Helper normalization utilities (for LLM triplet selection) ----

def _safe_minmax(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.nanmin(x)
    x_max = torch.nanmax(x)
    if (not torch.isfinite(x_min)) or (not torch.isfinite(x_max)) or (x_max - x_min) < 1e-6:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _safe_zscore(x: torch.Tensor) -> torch.Tensor:
    # Fallback implementation without torch.nanstd (not available in some torch versions)
    # Treat non-finite values (nan, inf) as missing.
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    finite_mask = torch.isfinite(x)
    if finite_mask.sum() == 0:
        return torch.zeros_like(x)
    x_finite = x[finite_mask]
    mu = x_finite.mean()
    # population standard deviation (consistent with nanstd default)
    diff = x_finite - mu
    sigma = torch.sqrt(torch.clamp(diff.pow(2).mean(), min=0.0))
    if (not torch.isfinite(sigma)) or sigma < 1e-6:
        return torch.zeros_like(x)
    return (x - mu) / sigma


def _normalize_per_question(scores_row: torch.Tensor, mode: str = "z") -> torch.Tensor:
    if mode == "z":
        return _safe_zscore(scores_row)
    return _safe_minmax(scores_row)


def _select_triplet_for_llm(
    correct_ids,
    incorrect_ids,
    row_rm_score: torch.Tensor,
    row_explore_score: torch.Tensor,
    alpha: float,
    norm_mode: str = "z",
) -> tuple[int, int]:
    """Composite score S[j] = -alpha * norm(trained)[j] + (1 - alpha) * norm(ref)[j].
    Higher S => better positive selection; lower S => negative selection."""
    rm_score_norm = _normalize_per_question(row_rm_score, mode=norm_mode)
    explore_norm = _normalize_per_question(row_explore_score, mode=norm_mode)
    S = alpha * row_explore_score + (1.0 - alpha) * rm_score_norm
    llm_pos_j = max(correct_ids, key=lambda j: float(S[j]))
    llm_neg_j = min(incorrect_ids, key=lambda j: float(explore_norm[j]))
    return llm_pos_j, llm_neg_j


def choose_pos_neg_triplets(
    questions: List[str],
    candidates: List[List[str]],
    correctness: Any,  # can be List[List[int]] or torch.Tensor
    rm_scores: torch.Tensor,
    explore_scores: List[List[float]],
    alpha: float
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Return triplets (question, pos_solution, neg_solution) selecting hardest pos (lowest score among correct)
    and hardest neg (highest score among incorrect) for each question with mixed correctness.
    Accepts correctness as list-of-lists or padded tensor.
    """
    triplets_for_rm: List[Tuple[str, str, str]] = []
    triplets_for_llm: List[Tuple[str, str, str]] = []
    is_tensor = isinstance(correctness, torch.Tensor)
    for qi, (q, cand_list) in enumerate(zip(questions, candidates)):
        if is_tensor:
            row_flags = [int(correctness[qi, j].item()) for j in range(len(cand_list))]
        else:
            row_flags = correctness[qi]
        # rm_scores_row retained for RM triplet selection
        rm_scores_row = rm_scores[qi]

        correct_ids = [j for j, v in enumerate(row_flags) if v == 1]
        incorrect_ids = [j for j, v in enumerate(row_flags) if v == 0]
        if not correct_ids or not incorrect_ids:
            continue
        # RM selection unchanged
        rm_pos_j = min(correct_ids, key=lambda j: rm_scores_row[j])
        rm_neg_j = max(incorrect_ids, key=lambda j: rm_scores_row[j])

        # ---- New LLM selection logic with per-question normalization ----
        K = len(cand_list)
        # Slice to actual candidate count for this question
        row_rm_score = rm_scores[qi, :K]
        # Convert explore_scores row to tensor for consistent indexing
        row_explore_score = torch.tensor(explore_scores[qi][:K], dtype=torch.float32)

        llm_pos_j, llm_neg_j = _select_triplet_for_llm(
            correct_ids,
            incorrect_ids,
            row_rm_score=row_rm_score,
            row_explore_score=row_explore_score,
            alpha=alpha,
            norm_mode="z",
        )

        triplets_for_rm.append((q, cand_list[rm_pos_j], cand_list[rm_neg_j]))
        # add here one random pos and one random neg
        rand_pos_j = random.choice(correct_ids)
        rand_neg_j = random.choice(incorrect_ids)
        triplets_for_rm.append((q, cand_list[rand_pos_j], cand_list[rand_neg_j]))
        triplets_for_llm.append((q, cand_list[llm_pos_j], cand_list[llm_neg_j]))
    return triplets_for_rm, triplets_for_llm


def ensure_empty_log_dir(path: str):
    """Ensure log directory exists and is empty at start of training.
    If it contains files, remove them (recursively)."""
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            try:
                if os.path.isdir(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
            except Exception:
                pass
    else:
        os.makedirs(path, exist_ok=True)


def filter_and_select_mixed(
    questions: List[str],
    gold_answers: List[str],
    candidate_texts: List[List[str]],
    candidate_valid_flags: List[List[int]],
    correctness: List[List[int]],
) -> Tuple[List[str], List[str], List[List[str]], List[List[int]]]:
    """Filter candidates removing invalid-but-correct items and select only questions
    that have mixed correctness (both 0 and 1 present). Returns filtered versions.
    If no mixed items remain, returns empty lists.
    """
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

    # Identify mixed correctness questions
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


def clean_end_candidates(candidates: List[List[str]]):
    pattern = re.compile(r"\\boxed\s*\{(.*?)\}", flags=re.DOTALL)


    for row in candidates:
        for c in row:
            matches = list(pattern.finditer(c))
            if not matches:
                continue
            last_match = matches[-1]
            row[row.index(c)] = c[:last_match.span()[1] + 2]




async def _async_save_model(trainer, path: str):
    """Run model.save_pretrained in a thread to avoid blocking event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, trainer.save_model, path)

async def _async_hot_swap(engine, path: str):
    """Invoke engine.hot_swap off the event loop (blocking requests.post)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engine.hot_swap, path)

async def _async_flush_cache(url: str):  # new helper
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, requests.post, url)





async def training_loop(config: Dict[str, Any]):
    rm_config = config.get("reward_model", {})
    rm_save_every_steps = rm_config.get("save_every_steps")
    rm_train_in_explore_every = rm_config["train_in_explore_every"]
    rm_save_path = rm_config.get("save_path")
    llm_name = config["model"]["llm_name"]
    rm_name = config["model"]["rm_name"]
    generation_config = config.get("generation")

    llm_gpu = config["hardware"].get("llm_gpu_id")
    rm_gpu = config["hardware"].get("rm_gpu_id")
    llm_trainer__gpu = config["hardware"].get("llm_trainer_gpu_id")

    llm_trainer_config = config.get("llm_trainer")

    num_steps = config["train"]["num_steps"]
    batch_size = config["train"]["batch_size"]
    n_samples = config["train"]["n_samples_per_problem"]


    evaluation_config = config.get("evaluation")
    tmp_weights_path = config.get("tmp_weights_safetensors_path")  # path with potential typo kept as-is
    url = f"http://localhost:30000/flush_cache"

    # Initialize alpha control from config
    alpha_ctrl_cfg = config.get("alpha_control", {})
    alpha_control = AlphaControl(alpha_ctrl_cfg)

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    train_ds, test_ds, q_field, a_field = load_dataset_handle(config)
    engine = build_sglang_engine(llm_name, generation_config)

    rm_model = load_reward_model(rm_name, rm_gpu, rm_config, num_steps)
    llm_trainer = load_llm_trainer(llm_name, llm_trainer__gpu, num_steps, llm_trainer_config)

    ensure_empty_log_dir(LOG_DIR)

    last_save_task: Optional[asyncio.Task] = None  # async save task from previous iteration
    last_swap_task: Optional[asyncio.Task] = None


    for step in range(num_steps):
        if step % rm_save_every_steps == 0 :
            rm_model.save_model(rm_save_path)
        # LLM trainer reference refresh
        if evaluation_config and (step > 0 or evaluation_config['at_start']) and step % evaluation_config['every_steps'] == 0:
            eval_res = await run_full_evaluation(
                engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config, rm_config
            )
            print(f"[Eval@Step {step}] {json.dumps(eval_res, indent=2)}")
            await _async_flush_cache(url)


        records = get_batch_records(train_ds, batch_size, step)
        questions = [r[q_field] for r in records]
        gold_answers = [r[a_field] for r in records]
        prompts = build_prompts(questions, tokenizer)
        st = time.time()

        # --- Wrapped generation with timeout & single retry (5 min each) ---
        timeout_seconds = 300  # 5 minutes
        raw_candidates = None
        for attempt in range(2):  # attempt 0 + retry 1
            try:
                raw_candidates = await asyncio.wait_for(
                    engine.generate_candidates(prompts, n_samples=n_samples, **generation_config),
                    timeout=timeout_seconds,
                )
                break  # success
            except asyncio.TimeoutError:
                print(f"[Step {step}] Generation timeout after {timeout_seconds}s (attempt {attempt+1}/2). Retrying..." if attempt == 0 else f"[Step {step}] Generation timeout after second attempt; skipping step.")
                torch.cuda.empty_cache()
                await _async_flush_cache(url)
                engine = build_sglang_engine(llm_name, generation_config)
            except EngineCircuitBreaker as e:  # specific handling
                print(f"[Step {step}] Circuit breaker tripped: {e}. Recreating engine (attempt {attempt+1}/2).")
                torch.cuda.empty_cache()
                await _async_flush_cache(url)
                engine = build_sglang_engine(llm_name, generation_config)
            except Exception as e:
                print(f"[Step {step}] Generation failed (attempt {attempt+1}/2): {e}" if attempt == 0 else f"[Step {step}] Generation failed again: {e}; skipping step.")
                torch.cuda.empty_cache()
                await _async_flush_cache(url)
                engine = build_sglang_engine(llm_name, generation_config)
            if raw_candidates is None and attempt == 1:
                continue  # will hit loop 'continue' below
        if raw_candidates is None:
            continue
        print(f"[Step {step}] Generation time: {time.time() - st:.2f}s")
        # Engine metrics logging
        try:
            m = engine.get_metrics()
            print(f"[Step {step}] Engine metrics: req={m['total_requests']} timeouts={m['timeouts']} errors={m['errors']} in_flight={m['in_flight']} phase2_batches={m['phase2_batches']} cb_trips={m['circuit_breaker_trips']}")
        except Exception:
            pass

        await _async_flush_cache(url)
        if last_save_task is not None:
            await last_save_task  # wait for save completion
            last_save_task = None
            last_swap_task = asyncio.create_task(_async_hot_swap(engine, tmp_weights_path))

        candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        candidate_valid_flags = [[c[1] for c in row] for row in raw_candidates]
        correctness = compute_final_correctness(candidate_texts, gold_answers)
        pass1_avg = [any([c==1 for c in row]) for row in correctness]


        # Filter invalid candidates and choose mixed correctness subset
        questions_f, gold_answers_f, candidates_f, correctness_filtered_list = filter_and_select_mixed(
            questions, gold_answers, candidate_texts, candidate_valid_flags, correctness
        )
        if not questions_f:
            continue
        max_k = max((len(row) for row in candidates_f), default=0)
        correctness_tensor = torch.zeros(len(candidates_f), max_k, dtype=torch.int32)
        for qi, row in enumerate(correctness_filtered_list):
            correctness_tensor[qi, :len(row)] = torch.tensor(row, dtype=torch.int32)




        st = time.time()
        try:
            explore_scores, entropy_scores = llm_trainer.compute_explore_and_entropy_scores(questions_f, candidates_f, 5)
        except Exception as e:
            print(f"[Step {step}] Exception during explore/entropy scoring: {e} will retry batch with 0.25 batch size.")
            torch.cuda.empty_cache()
            explore_scores, entropy_scores = llm_trainer.compute_explore_and_entropy_scores(questions_f, candidates_f, 12)
        print(f"[Step {step}] Explore score time: {time.time() - st:.2f}s")



        st = time.time()
        try:
            rm_scores = rm_model.score_reference(questions_f, candidates_f, rm_config)
        except Exception as e:
            print(f"[Step {step}] Exception during RM scoring: {e} will retry batch with 0.25 batch size.")
            torch.cuda.empty_cache()
            try:
                rm_scores = rm_model.score_reference(questions_f, candidates_f, rm_config, forced_small_batch_size=True)
            except Exception as e:
                print(f"[Step {step}] Exception during RM scoring retry: {e} will skip step.")
                continue
        print(f"[Step {step}] RM Scoring time: {time.time() - st:.2f}s")



        torch.cuda.empty_cache()
        triplets_for_rm, triplets_for_llm = choose_pos_neg_triplets(questions_f, candidates_f, correctness_tensor, rm_scores, entropy_scores, alpha_control.alpha)
        if not triplets_for_rm:
            continue

        try:
            rm_avg_loss = rm_model.train_step(triplets_for_rm)
        except Exception as e:
            print(f"[Step {step}] Exception during RM training: {e} will skip")
            rm_avg_loss = 0.0

        try:
            llm_avg_loss = llm_trainer.train_step(triplets_for_llm)
        except Exception as e:
            print(f"[Step {step}] Exception during LLM training: {e} will skip")
            llm_avg_loss = 0.0
        print(f"[Step {step}] RM Loss: {rm_avg_loss:.4f}, LLM Loss: {llm_avg_loss:.4f} alpha: {alpha_control.alpha:.4f}")


        alpha_control.step(np.mean([np.mean(c) for c in correctness_filtered_list]), np.mean(pass1_avg), np.mean(entropy_scores), step)

        log_questions(questions_f, gold_answers_f, candidates_f, rm_scores, explore_scores, entropy_scores, correctness_filtered_list, rm_avg_loss, llm_avg_loss, pass1)
        if last_swap_task is not None:
            await last_swap_task
        last_save_task = asyncio.create_task(_async_save_model(llm_trainer, tmp_weights_path))

    if last_save_task is not None:
        await last_save_task


def run(config_path: str):
    config = load_config(config_path)
    asyncio.run(training_loop(config))
