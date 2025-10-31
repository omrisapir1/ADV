#!/usr/bin/env python
from __future__ import annotations

import argparse
import yaml
import json
import os
from typing import Any, Dict
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluation_sync import run_full_evaluation_sync
from generation_sync import build_sglang_engine_sync
from reward_model import load_reward_model


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_generation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'think_temperature': cfg['think_temperature'],
        'think_top_p': cfg['think_top_p'],
        'think_top_k': cfg['think_top_k'],
        'think_repetition_penalty': cfg['think_repetition_penalty'],
        'think_max_new_tokens': cfg['think_max_new_tokens'],
        'answer_max_new_tokens': cfg['answer_max_new_tokens'],
        'answer_stop': cfg.get('answer_stop', []),
    }


def main():
    parser = argparse.ArgumentParser(description='Run synchronous evaluation (serial) using sync engine.')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to YAML config.')
    parser.add_argument('--max-questions', type=int, default=500, help='Limit number of questions for quick eval.')
    parser.add_argument('--output', default='sync_eval_result.json', help='Path to write combined result json.')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    llm_name = cfg['model']['llm_name']
    rm_name = cfg['model']['rm_name']
    q_field = cfg['dataset']['field_question']
    a_field = cfg['dataset']['field_answer']
    generation_cfg = prepare_generation_config(cfg['generation'])
    evaluation_cfg = cfg['evaluation']
    rm_config = cfg.get('reward_model', {})

    # Load dataset
    ds_name = cfg['dataset']['name']
    # split = cfg['dataset']['split']
    split = 'test'
    print(f'Loading dataset {ds_name} split {split} ...')
    dataset = load_dataset(ds_name, split=split)
    if args.max_questions > 0:
        dataset = dataset.select(range(min(args.max_questions, len(dataset))))
    print(f'Loaded {len(dataset)} records.')

    # Tokenizer for prompts
    print(f'Loading tokenizer for {llm_name} ...')
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token_id = getattr(tokenizer, 'eos_token_id', 0)
    tokenizer.padding_side = 'left'

    # Build synchronous engine
    print('Building synchronous SGLang engine wrapper ...')
    engine = build_sglang_engine_sync(llm_name, {})

    # Load reward model
    rm_gpu_id = int(cfg['hardware'].get('rm_gpu_id', 0))
    print(f'Loading reward model {rm_name} on GPU {rm_gpu_id} ...')
    rm_model = load_reward_model(rm_name, rm_gpu_id, rm_config)

    print('Running full synchronous evaluation ...')
    combined = run_full_evaluation_sync(
        engine,
        rm_model,
        dataset,
        q_field,
        a_field,
        tokenizer,
        generation_cfg,
        evaluation_cfg,
        rm_config,
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print('Evaluation complete.')
    print(json.dumps(combined, indent=2))
    print(f'Results written to {args.output}')


if __name__ == '__main__':
    main()

