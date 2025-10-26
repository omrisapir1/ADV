import asyncio
import yaml
import torch
import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from .prompting import build_prompts
from .generation import build_sglang_engine
from .reward_model import load_reward_model
from .answer_parse import compute_final_correctness
from .llm_trainer import load_llm_trainer
from .evaluation import run_full_evaluation  # added import

import time

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

def log_questions(questions: List[str], gold_answers: List[str], candidates: List[List[str]], rm_scores: torch.Tensor, correctness: List[List[int]], rm_avg_loss, llm_avg_loss):
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
                question_rm_scores = tmp

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
            "correctness": corr_list,
            "num_candidates": len(candidates[i]) if i < len(candidates) else 0,
            "avg_rm_score": float(sum(question_rm_scores) / len(question_rm_scores)) if question_rm_scores else 0.0,
            "correct_count": correct_count,
            'rm_avg_loss':rm_avg_loss,
            'llm_avg_loss': llm_avg_loss
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


def choose_pos_neg_triplets(
    questions: List[str],
    candidates: List[List[str]],
    correctness: Any,  # can be List[List[int]] or torch.Tensor
    rm_scores: torch.Tensor,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Return triplets (question, pos_solution, neg_solution) selecting hardest pos (lowest score among correct)
    and hardest neg (highest score among incorrect) for each question with mixed correctness.
    Accepts correctness as list-of-lists or padded tensor.
    """
    triplets: List[Tuple[str, str, str]] = []
    triplets_for_rm: List[Tuple[str, str, str]] = []
    is_tensor = isinstance(correctness, torch.Tensor)
    for qi, (q, cand_list) in enumerate(zip(questions, candidates)):
        if is_tensor:
            row_flags = [int(correctness[qi, j].item()) for j in range(len(cand_list))]
        else:
            row_flags = correctness[qi]
        scores_row = rm_scores[qi]
        correct_ids = [j for j, v in enumerate(row_flags) if v == 1]
        incorrect_ids = [j for j, v in enumerate(row_flags) if v == 0]
        if not correct_ids or not incorrect_ids:
            continue
        pos_j = min(correct_ids, key=lambda j: scores_row[j])
        neg_j = max(incorrect_ids, key=lambda j: scores_row[j])
        pos_j_r = min(correct_ids, key=lambda j: scores_row[j])
        triplets.append((q, cand_list[pos_j], cand_list[neg_j]))
        triplets_for_rm.append((q, cand_list[pos_j_r], cand_list[neg_j]))
    return triplets, triplets_for_rm


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


async def _async_save_model(trainer, path: str):
    """Run model.save_pretrained in a thread to avoid blocking event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, trainer.save_model, path)

async def _async_hot_swap(engine, path: str):
    """Invoke engine.hot_swap off the event loop (blocking requests.post)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engine.hot_swap, path)


async def training_loop(config: Dict[str, Any]):
    rm_config = config.get("reward_model", {})
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

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    train_ds, test_ds, q_field, a_field = load_dataset_handle(config)
    engine = build_sglang_engine(llm_name, generation_config)

    rm_model = load_reward_model(rm_name, rm_gpu, rm_config, num_steps)
    llm_trainer = load_llm_trainer(llm_name, llm_trainer__gpu, num_steps, llm_trainer_config)
    if evaluation_config:
        if evaluation_config.get('at_start'):
            eval_res = await run_full_evaluation(
                engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config, rm_config
            )
            print(f"[Eval@Start] {json.dumps(eval_res, indent=2)}")


    # ensure_empty_log_dir(LOG_DIR)

    last_save_task: Optional[asyncio.Task] = None  # async save task from previous iteration
    last_swap_task: Optional[asyncio.Task] = None
    last_half_batch: Optional[asyncio.Task] = None
    for step in range(num_steps):
        if evaluation_config and step > 0 and step % evaluation_config['every_steps'] == 0:
            eval_res = await run_full_evaluation(
                engine, rm_model, test_ds, q_field, a_field, tokenizer, generation_config, evaluation_config, rm_config
            )
            print(f"[Eval@Step {step}] {json.dumps(eval_res, indent=2)}")
        if llm_trainer_config['update_ref_model_every'] % step == 0:
            llm_trainer.update_ref_model()

        records = get_batch_records(train_ds, batch_size, step)
        questions = [r[q_field] for r in records]
        gold_answers = [r[a_field] for r in records]
        prompts = build_prompts(questions, tokenizer)
        st = time.time()
        if step == 0:
            raw_candidates = await engine.generate_candidates(prompts, n_samples=n_samples, **generation_config)
        else:
            raw_candidates = await engine.generate_candidates(prompts[:int(batch_size / 2)], n_samples=n_samples, **generation_config)

        print(f"[Step {step}] Generation time: {time.time() - st:.2f}s")

        if last_save_task is not None:
            await last_save_task  # wait for save completion
            last_save_task = None
            # hot-swap freshly saved weights before new generation
            last_swap_task = asyncio.create_task(_async_hot_swap(engine, tmp_weights_path))
        if step>0:
            await last_half_batch
            print(f'len(last_half_batch) = {len(last_half_batch)}')
            print(f'len(raw_candidates) = {len(raw_candidates)}')
            raw_candidates.extend(last_half_batch)
            print(f'after len(raw_candidates) = {len(raw_candidates)}')

        last_half_batch = asyncio.create_task(
            engine.generate_candidates(prompts[int(batch_size / 2):], n_samples=n_samples, **generation_config))
        candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        candidate_valid_flags = [[c[1] for c in row] for row in raw_candidates]
        correctness = compute_final_correctness(candidate_texts, gold_answers)

        questions, gold_answers, candidates, correctness_filtered_list = filter_and_select_mixed(
            questions, gold_answers, candidate_texts, candidate_valid_flags, correctness
        )
        if not questions:
            continue
        max_k = max((len(row) for row in candidates), default=0)
        correctness_tensor = torch.zeros(len(candidates), max_k, dtype=torch.int32)
        for qi, row in enumerate(correctness_filtered_list):
            correctness_tensor[qi, :len(row)] = torch.tensor(row, dtype=torch.int32)
        st = time.time()



        try:
            rm_scores = rm_model.score_reference(questions, candidates, rm_config)
        except Exception as e:
            print(f"[Step {step}] Exception during RM scoring: {e} will retry batch with 0.25 batch size.")
            torch.cuda.empty_cache()
            rm_scores = rm_model.score_reference(questions, candidates, rm_config, forced_small_batch_size=True)
        print(f"[Step {step}] RM Scoring time: {time.time() - st:.2f}s")
        torch.cuda.empty_cache()
        triplets, triplets_for_rm = choose_pos_neg_triplets(questions, candidates, correctness_tensor, rm_scores)
        if not triplets:
            continue
        try:
            rm_avg_loss = rm_model.train_step(triplets_for_rm)
        except Exception as e:
            print(f"[Step {step}] Exception during RM training: {e} will skip")
            rm_avg_loss = 0.0
        try:
            llm_avg_loss = llm_trainer.train_step(triplets)
        except Exception as e:
            print(f"[Step {step}] Exception during LLM training: {e} will skip")
            llm_avg_loss = 0.0

        print(f"[Step {step}] RM Loss: {rm_avg_loss:.4f}, LLM Loss: {llm_avg_loss:.4f}")

        log_questions(questions, gold_answers, candidates, rm_scores, correctness_filtered_list, rm_avg_loss, llm_avg_loss)

        # ---- ASYNC SAVE (end of iteration) ----
        # Before starting new save ensure earlier hot swap is done (we awaited it already above before generation).
        # Launch save task so disk write can overlap with next RM scoring & other CPU work.
        if last_swap_task is not None:
            await last_swap_task
        last_save_task = asyncio.create_task(_async_save_model(llm_trainer, tmp_weights_path))

    # Final wait to ensure last save completes.
    if last_save_task is not None:
        await last_save_task


def run(config_path: str):
    config = load_config(config_path)
    asyncio.run(training_loop(config))


