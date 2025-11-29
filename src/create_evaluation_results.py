import time
from datasets import load_dataset
import asyncio
import yaml
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer

try:
    from .prompting import build_prompts
    from .answer_parse import compute_final_correctness
    from .generation import build_sglang_engine
    from .generation_tree_sglang import build_tree_sglang_engine
except ImportError:
    from prompting import build_prompts
    from answer_parse import compute_final_correctness
    from generation import build_sglang_engine
    from generation_tree_sglang import build_tree_sglang_engine

CONFIG_PATH = os.environ.get("ADV_CONFIG", "configs/config.yaml")

@dataclass
class EvalSettings:
    llm_name: str
    dataset_name: str
    q_field: str
    a_field: str
    split: str
    n_samples: int
    batch_size: int
    generation_cfg: Dict[str, Any]


def load_config(path: str) -> EvalSettings:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_name = cfg["model"]["llm_name"]
    dcfg = cfg["dataset"]
    dataset_name = dcfg["name"]
    q_field = dcfg.get("field_question", "problem")
    a_field = dcfg.get("field_answer", "final_answer")
    split = 'test'
    n_samples = 124
    batch_size = 16
    generation_cfg = cfg.get("generation")
    print(generation_cfg)
    return EvalSettings(
        llm_name=llm_name,
        dataset_name=dataset_name,
        q_field=q_field,
        a_field=a_field,
        split=split,
        n_samples=n_samples,
        batch_size=batch_size,
        generation_cfg=generation_cfg,
    )


def load_testset(dataset_name: str, split: str):
    ds = load_dataset(dataset_name)
    # fallbacks if desired split missing
    if split not in ds:  # choose test > validation > train
        for alt in ["test", "validation", "train"]:
            if alt in ds:
                split = alt
                break
    return ds[split]


def select_records(ds, q_field: str, a_field: str,):
    total = len(ds)
    use_n = total
    use_n = min(use_n, total)
    records = [ds[i] for i in range(use_n)]
    questions = [r[q_field] for r in records]
    gold_answers = [r[a_field] for r in records]
    return questions, gold_answers


async def generate_all(engine, tokenizer, questions: List[str], gold_answers: List[str], n_samples: int, generation_cfg: Dict[str, Any], batch_size: int):
    out_rows: List[Dict[str, Any]] = []
    for start in range(0, len(questions), batch_size):
        end = min(start + batch_size, len(questions))
        batch_q = questions[start:end]
        prompts = build_prompts(batch_q, tokenizer)
        raw_candidates = await engine.generate_candidates(prompts, n_samples=n_samples, **generation_cfg)
        candidate_texts = [[c[0] for c in row] for row in raw_candidates]
        candidate_entropy = [[c[2] for c in row] for row in raw_candidates]
        correctness = compute_final_correctness(candidate_texts, gold_answers[start:end])
        for i, q in enumerate(batch_q):
            row_candidates = candidate_texts[i]
            row_correct = correctness[i]
            out_rows.append({
                "question": q,
                "gold_answer": gold_answers[start + i],
                "candidates": row_candidates,
                "entropies": candidate_entropy[i],
                "correctness": row_correct,
            })
    return out_rows


async def run():
    settings = load_config(CONFIG_PATH)
    print(f"Loaded config for model={settings.llm_name} dataset={settings.dataset_name} split={settings.split}")
    test_ds = load_testset(settings.dataset_name, settings.split)
    questions, gold_answers = select_records(test_ds, settings.q_field, settings.a_field)
    print(f"Evaluating {len(questions)} questions with {settings.n_samples} samples each (batch_size={settings.batch_size})")
    tokenizer = AutoTokenizer.from_pretrained(settings.llm_name)
    engine = build_tree_sglang_engine(settings.llm_name, settings.generation_cfg)
    st = time.time()
    rows = await generate_all(engine, tokenizer, questions, gold_answers, settings.n_samples, settings.generation_cfg, settings.batch_size)
    print(f"Generation time: {time.time() - st:.2f}s")
    df = pd.DataFrame(rows, columns=["question", "gold_answer", "candidates", "correctness"])
    df.to_pickle("evaluation_results.pkl")


if __name__ == "__main__":
    asyncio.run(run())

