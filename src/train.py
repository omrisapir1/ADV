import time
from typing import Dict, Any, List, Tuple
import asyncio
import yaml
import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
from .prompting import build_prompts
from .generation import build_sglang_engine  # switched from build_vllm_engine to build_sglang_engine
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


def log_questions(questions: List[str], gold_answers: List[str], candidates: List[List[str]], rm_scores: torch.Tensor, correctness: List[List[bool]]):
    """Log training results to disk in JSON format."""

    # Create logs directory if it doesn't exist
    log_dir = "/workspace/ADV/src/data"
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
        question_rm_scores = []
        if i < len(rm_scores):
            if isinstance(rm_scores[i], torch.Tensor):
                question_rm_scores = rm_scores[i].cpu().detach().tolist()
            else:
                question_rm_scores = list(rm_scores[i])

        question_data = {
            "question_id": i,
            "question": question,
            "gold_answer": gold_answer,
            "candidates": candidates[i] if i < len(candidates) else [],
            "rm_scores": question_rm_scores,
            "correctness": correctness[i] if i < len(correctness) else [],
            "num_candidates": len(candidates[i]) if i < len(candidates) else 0,
            "avg_rm_score": float(sum(question_rm_scores) / len(question_rm_scores)) if question_rm_scores else 0.0,
            "correct_count": sum(correctness[i]) if i < len(correctness) else 0
        }

        try:
            json.dumps(question_data)
        except:
            print(f"Error serializing question data: {question_data}")
            continue
        log_data["questions"].append(question_data)

    # Write to file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"Logged results to {log_file}")


def choose_pos_neg_triplets(
    questions: List[str],
    candidates: List[List[str]],
    correctness: List[List[int]],
    rm_scores: torch.Tensor,
) -> List[Tuple[str, str, str]]:
    """Return triplets (question, pos_solution, neg_solution) selecting hardest pos (lowest score among correct)
    and hardest neg (highest score among incorrect) for each question with mixed correctness.
    """
    triplets: List[Tuple[str, str, str]] = []
    for qi, (q, cand_list, corr_list) in enumerate(zip(questions, candidates, correctness)):
        scores_row = rm_scores[qi]
        correct_ids = [j for j, v in enumerate(corr_list) if v == 1]
        incorrect_ids = [j for j, v in enumerate(corr_list) if v == 0]
        if not correct_ids or not incorrect_ids:
            continue
        pos_j = min(correct_ids, key=lambda j: scores_row[j])
        neg_j = max(incorrect_ids, key=lambda j: scores_row[j])
        triplets.append((q, cand_list[pos_j], cand_list[neg_j]))
    return triplets


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
    log_every = train_config.get("log_every", 10)
    save_every = train_config.get("save_every", 500)
    out_dir = train_config.get("out_dir", "/workspace/ADV/checkpoints/rm")

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    # Initialize reward model with integrated optimizer/scheduler via accelerator
    rm_model = load_reward_model(rm_name, rm_gpu, rm_config, num_steps, accel)

    # Build SGLang engine
    engine = build_sglang_engine(llm_name, sglang_config)
    dataset_obj, q_field, a_field = load_dataset_handle(config)
    os.makedirs(out_dir, exist_ok=True)

    for step in range(num_steps):
        records = get_batch_records(dataset_obj, batch_size, step)
        questions = [r[q_field] for r in records]
        gold_answers = [r[a_field] for r in records]
        prompts = build_prompts(questions, tokenizer)
        st = time.time()
        candidates = engine.generate_candidates(prompts, n_samples=n_samples, **gen_cfg)
        print(f'candidates Total time: {time.time() - st}')
        print(f"[Step {step}] Generated candidates per question: {[len(c) for c in candidates]}")
        st = time.time()
        correctness = compute_final_correctness(candidates, gold_answers)
        print(f'correctness Total time: {time.time() - st}')

        filtered_indices = []
        for i, question_correctness in enumerate(correctness):
            unique_values = set(question_correctness)
            if len(unique_values) > 1:
                filtered_indices.append(i)
        if not filtered_indices:
            print(f"[Step {step}] No mixed correctness examples found, skipping this batch")
            continue
        print(f"[Step {step}] Filtered from {len(questions)} to {len(filtered_indices)} examples with mixed correctness")

        questions = [questions[i] for i in filtered_indices]
        gold_answers = [gold_answers[i] for i in filtered_indices]
        candidates = [candidates[i] for i in filtered_indices]
        correctness = [correctness[i] for i in filtered_indices]

        st = time.time()
        rm_scores = rm_model.score_reference(questions, candidates, rm_config)
        print(f'rm_scores Total time: {time.time() - st}')

        triplets = choose_pos_neg_triplets(questions, candidates, correctness, rm_scores)
        if not triplets:
            print(f"[Step {step}] No valid pos/neg triplets after selection, skipping.")
            continue

        avg_loss, last_lr = rm_model.train_step(triplets, accel)
        print(f"[Step {step}] Loss: {avg_loss:.4f}, LR: {last_lr:.2e}, Triplets: {len(triplets)}")
        log_questions(questions, gold_answers, candidates, rm_scores, correctness)

        if step % save_every == 0 and step > 0:
            checkpoint_path = os.path.join(out_dir, f"checkpoint-{step}")
            accel.save_state(checkpoint_path)
            print(f"[Step {step}] Saved checkpoint to {checkpoint_path}")

        print(f"[Step {step}] Completed batch with {len(questions)} questions. Triplets selected: {len(triplets)}")

    final_path = os.path.join(out_dir, "final")
    accel.save_state(final_path)
    print(f"Training completed. Final checkpoint saved to {final_path}")


def run(config_path: str):
    config = load_config(config_path)
    asyncio.run(training_loop(config))
