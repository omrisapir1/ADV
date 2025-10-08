from __future__ import annotations
from typing import Dict, Any, List
import yaml
import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer

from .prompting import build_prompts
from .generation import build_vllm_engine
from .reward_model import load_reward_model, score_solutions
from .answer_parse import compute_final_correctness
from .losses import compute_joint_loss



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

        print(f'question_data {question_data}')
        log_data["questions"].append(question_data)

    # Write to file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"Logged results to {log_file}")


def training_loop(config: Dict[str, Any]):
    accel = Accelerator()
    llm_name = config["model"]["llm_name"]
    rm_name = config["model"]["rm_name"]
    vllm_config = config["vllm"]
    llm_gpu = config["hardware"].get("llm_gpu_id", 0)
    rm_gpu = config["hardware"].get("rm_gpu_id", 1)
    rm_config = config.get("reward_model", {})

    gen_cfg = config["generation"]
    n_samples = gen_cfg["n_samples_per_problem"]

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    engine = build_vllm_engine(llm_name, llm_gpu, vllm_config)
    rm_model = load_reward_model(rm_name, rm_gpu)

    batch_size = config["train"]["batch_size"]
    num_steps = config["train"]["num_steps"]

    dataset_obj, q_field, a_field = load_dataset_handle(config)

    for step in range(num_steps):
        records = get_batch_records(dataset_obj, batch_size, step)
        questions = [r[q_field] for r in records]
        gold_answers = [r[a_field] for r in records]
        prompts = build_prompts(questions, tokenizer)
        print(f'len prompts {len(prompts)}')
        candidates = engine.generate_candidates(prompts, n_samples=n_samples, **gen_cfg)
        print(f"[Step {step}] Generated candidates per question: {[len(c) for c in candidates]}")
        flat_solutions = [sol for cand_list in candidates for sol in cand_list]
        correctness = compute_final_correctness(candidates, gold_answers)
        #filter non mixed answers

        with torch.no_grad():
            rm_scores = score_solutions(questions, flat_solutions, rm_model, n_samples, rm_config)


        log_questions(questions, gold_answers, candidates, rm_scores, correctness.cpu().detach().tolist())

        loss = None# compute_joint_loss(rm_scores, correctness, candidates)

        if loss is not None and isinstance(loss, torch.Tensor) and loss.item() > 0:
            accel.backward(loss)
            print(f"[Step {step}] Computed loss: {loss.item():.4f}")
        else:
            print(f"[Step {step}] No valid loss computed (no correct/incorrect pairs)")
            if hasattr(rm_model, 'model') and hasattr(rm_model.model, 'parameters'):
                for p in rm_model.model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            elif hasattr(rm_model, 'parameters'):
                for p in rm_model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
        print(f"[Step {step}] Completed batch with {len(questions)} questions.")


def run(config_path: str):
    config = load_config(config_path)
    training_loop(config)
