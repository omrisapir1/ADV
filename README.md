# ADV

Minimal Working Prototype (MWP) for joint data flow between a small generation LLM and a larger reward model (RM). This prototype wires together:

1. Dummy math problem dataset → prompt construction.
2. SGLang (HTTP completion endpoint via OpenAI-compatible client) → multiple candidate solutions per problem.
3. Reward model stub OR AceMath-7B-RM (if available) → scores per candidate.
4. Placeholders for correctness evaluation and joint loss.

No real optimization, correctness, or RLHF/GRPO/PPO logic is implemented yet—only the plumbing and debug output.

## Repo Structure
```
configs/
  config.yaml          # Configuration (model names, generation + training params)
requirements.txt       # Dependencies
src/
  main.py              # CLI entry point
  train.py             # Training loop
  prompting.py         # Prompt construction utilities
  generation.py        # SGLang wrapper (OpenAI-compatible client)
  reward_model.py      # Reward model
  losses.py            # Joint loss placeholder
  answer_parse.py      # Correctness placeholder
```

## Configuration (configs/config.yaml)
Key fields:
- model.llm_name / model.rm_name: HF model identifiers.
- generation: sampling parameters + number of candidates per question.
- sglang: base_url and api_key for SGLang server.
- hardware.llm_gpu_id / hardware.rm_gpu_id: device IDs.
- train.batch_size / train.num_steps: loop control.

## Install (Minimal)
You only strictly need torch + pyyaml to run the flow:
```bash
pip install torch pyyaml
```
Optional (real components):
```bash
pip install accelerate transformers openai datasets
```

## Run
From repo root:
```bash
python -m src.main --config configs/config.yaml
```

## Expected Console Output (Example)
```
[Step 0] Generated candidates per question: [96, 96]
[Step 0] Loss: 0.4321, LR: 1.00e-04, Triplets: 12
[Step 0] Completed batch with 12 questions. Triplets selected: 12
```
Values and shapes depend on config (batch_size, n_samples_per_problem).

## Placeholders (Intentional)
- answer_parse.compute_final_correctness → simplistic correctness logic.
- losses.pairwise_rm_loss → pairwise preference loss only.
- Reward model may fallback depending on resources.

## Reward Model Notes
Loads AceMath-7B-RM via transformers when available; otherwise you must adjust for smaller models if constrained.

## SGLang Generation
The generation layer makes raw completion calls (no chat template) using an OpenAI-compatible endpoint. Configure `sglang.base_url` and `sglang.api_key` in `config.yaml` or environment vars (`SGLANG_BASE_URL`, `SGLANG_API_KEY`).

## Where to Implement Next
- Improve correctness parsing in `answer_parse.py`.
- Add more robust logging / evaluation metrics.
- Integrate curriculum or adaptive sampling.
- Add unit tests.

## Safety / Resource Notes
- High `n_samples_per_problem` increases latency and memory.
- Use smaller batch sizes if GPU memory is limited.

## License
Placeholder (add a license file as needed).
