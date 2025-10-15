import time
from typing import Dict, Any, List, Tuple
import asyncio
import yaml
import torch
import json
import os
import shutil
from datetime import datetime
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
from .prompting import build_prompts
from .generation import build_sglang_engine
from .reward_model import load_reward_model
from .answer_parse import compute_final_correctness



def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_handle(cfg: Dict[str, Any]):
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name")
    split = ds_cfg.get("split", "train")
    ds = load_dataset(name, split=split)
    return ds, ds_cfg.get("field_question", "problem"), ds_cfg.get("field_answer", "final_answer")

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

def log_questions(questions: List[str], gold_answers: List[str], candidates: List[List[str]], rm_scores: torch.Tensor, correctness: List[List[bool]]):
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
            "correct_count": correct_count
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
) -> List[Tuple[str, str, str]]:
    """Return triplets (question, pos_solution, neg_solution) selecting hardest pos (lowest score among correct)
    and hardest neg (highest score among incorrect) for each question with mixed correctness.
    Accepts correctness as list-of-lists or padded tensor.
    """
    triplets: List[Tuple[str, str, str]] = []
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
        triplets.append((q, cand_list[pos_j], cand_list[neg_j]))
    return triplets


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


async def training_loop(config: Dict[str, Any]):
    rm_config = config.get("reward_model", {})
    train_config = rm_config.get("train", {})
    mixed_precision = train_config.get("mixed_precision", "bf16")
    grad_accum = train_config.get("grad_accum", 1)
    accel = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=grad_accum)

    llm_name = config["model"]["llm_name"]
    rm_name = config["model"]["rm_name"]
    sglang_config = config.get("sglang", {})  # new config section for SGLang
    llm_gpu = config["hardware"].get("llm_gpu_id", 0)
    rm_gpu = config["hardware"].get("rm_gpu_id", 1)
    gen_cfg = config["generation"]
    n_samples = gen_cfg["n_samples_per_problem"]
    num_steps = config["train"]["num_steps"]
    batch_size = config["train"]["batch_size"]
    save_every = train_config.get("save_every", 500)
    out_dir = train_config.get("out_dir", "/workspace/ADV/checkpoints/rm")

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    # Initialize reward model with integrated optimizer/scheduler via accelerator
    rm_model = load_reward_model(rm_name, rm_gpu, rm_config, num_steps, accel)

    # Build SGLang engine
    engine = build_sglang_engine(llm_name, sglang_config)
    dataset_obj, q_field, a_field = load_dataset_handle(config)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure log directory is clean at the start of training
    ensure_empty_log_dir(LOG_DIR)

    for step in range(num_steps):
        records = get_batch_records(dataset_obj, batch_size, step)
        questions = [r[q_field] for r in records]
        gold_answers = [r[a_field] for r in records]
        prompts = build_prompts(questions, tokenizer)
        st = time.time()
        raw_candidates = await engine.generate_candidates(prompts, n_samples=n_samples, **gen_cfg)
        # print(f'Candidate generation Total time: {time.time() - st}')
        # raw_candidates: List[List[(text, valid_flag)]] where valid_flag=1 if phase-2 executed, else 0
        # Extract candidate texts and validity flags
        candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        candidate_valid_flags = [[c[1] for c in row] for row in raw_candidates]
        # Silenced log output
        # print(f'candidates Total time: {time.time() - st}')
        # print(f"[Step {step}] Generated candidates per question: {[len(c) for c in candidate_texts]}")

        # Compute correctness on the raw candidate texts
        st = time.time()
        correctness = compute_final_correctness(candidate_texts, gold_answers)  # list of lists (0/1)
        # Silenced log output
        # print(f'correctness Total time: {time.time() - st}')
        # print(f'correctness: {correctness}')

        # Filter out candidates that are invalid (valid_flag==0) yet marked correct (correctness==1).
        filtered_candidate_texts: List[List[str]] = []
        filtered_correctness: List[List[int]] = []
        for qi, (texts_row, flags_row, corr_row) in enumerate(zip(candidate_texts, candidate_valid_flags, correctness)):
            new_texts: List[str] = []
            new_corr: List[int] = []
            for t, f, corr in zip(texts_row, flags_row, corr_row):
                # Drop only if candidate invalid (f==0) but correctness says it's correct (corr==1)
                if corr ==-1 or (f == 0 and corr == 1):
                    continue
                new_texts.append(t)
                new_corr.append(corr)
            filtered_candidate_texts.append(new_texts)
            filtered_correctness.append(new_corr)
        # Replace working variables with filtered versions
        candidates = filtered_candidate_texts
        correctness = filtered_correctness

        # Identify questions with mixed correctness (both 0 and 1 present after filtering)
        filtered_indices = []
        for i, question_correctness in enumerate(correctness):
            unique_values = set(question_correctness)
            # Silenced log output
            # print(f'question_correctness for question {i}: {question_correctness}, unique values: {unique_values}')
            # print(f'unique values: {unique_values}')
            if 1 in unique_values and 0 in unique_values:
                filtered_indices.append(i)
        if not filtered_indices:
            # Silenced log output
            # print(f"[Step {step}] No mixed correctness examples found, skipping this batch")
            continue
        # Silenced log output
        # print(f"[Step {step}] Filtered from {len(questions)} to {len(filtered_indices)} examples with mixed correctness")

        questions = [questions[i] for i in filtered_indices]
        gold_answers = [gold_answers[i] for i in filtered_indices]
        candidates = [candidates[i] for i in filtered_indices]
        correctness_filtered_list = [correctness[i] for i in filtered_indices]

        # Convert correctness to padded tensor after filtering
        max_k = max((len(row) for row in candidates), default=0)
        correctness_tensor = torch.zeros(len(candidates), max_k, dtype=torch.int32)
        for qi, row in enumerate(correctness_filtered_list):
            correctness_tensor[qi, :len(row)] = torch.tensor(row, dtype=torch.int32)
        # Silenced log output
        # print(f"[Step {step}] correctness tensor shape: {correctness_tensor.shape}")

        st = time.time()
        rm_scores = rm_model.score_reference(questions, candidates, rm_config)
        # Silenced log output
        # print(f'rm_scores Total time: {time.time() - st}')


        triplets = choose_pos_neg_triplets(questions, candidates, correctness_tensor, rm_scores)
        if not triplets:
            # Silenced log output
            # print(f"[Step {step}] No valid pos/neg triplets after selection, skipping.")
            continue
        st = time.time()
        # avg_loss, _ = rm_model.train_step(triplets, accel)
        # Silenced log output
        # print(f'rm_model.train_step Total time: {time.time() - st}')
        # print(f"[Step {step}] Loss: {avg_loss:.4f}")
        log_questions(questions, gold_answers, candidates, rm_scores, correctness_filtered_list)

        if step % save_every == 0 and step > 0:
            checkpoint_path = os.path.join(out_dir, f"checkpoint-{step}")
            accel.save_state(checkpoint_path)
            # Silenced log output
            # print(f"[Step {step}] Saved checkpoint to {checkpoint_path}")

        # Silenced log output
        # print(f"[Step {step}] Completed batch with {len(questions)} questions. Triplets selected: {len(triplets)}")

    final_path = os.path.join(out_dir, "final")
    accel.save_state(final_path)


def run(config_path: str):
    config = load_config(config_path)
    asyncio.run(training_loop(config))
