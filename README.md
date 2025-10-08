# ADV

Minimal Working Prototype (MWP) for joint data flow between a small generation LLM and a larger reward model (RM). This prototype wires together:

1. Dummy math problem dataset → prompt construction.
2. vLLM (or a lightweight dummy fallback) → multiple candidate solutions per problem.
3. Reward model stub OR AceMath-7B-RM (if available) → scores per candidate.
4. Placeholders for correctness evaluation and joint loss.

No real optimization, correctness, or RLHF/GRPO/PPO logic is implemented yet—only the plumbing and debug output.

## Repo Structure
```
configs/
  default.yaml          # Configuration (model names, generation + training params)
requirements.txt        # Dependencies (heavy libs optional; dummy fallbacks are used if missing)
src/
  main.py               # CLI entry point
  train.py              # Training loop skeleton
  prompting.py          # Prompt construction utilities
  generation.py         # vLLM wrapper (with dummy fallback)
  reward_model.py       # Reward model stub
  losses.py             # Joint loss placeholder (NotImplementedError)
  answer_parse.py       # Correctness placeholder (NotImplementedError)
```

## Configuration (configs/default.yaml)
Key fields:
- model.llm_name: placeholder string (not downloaded unless vLLM present).
- model.rm_name: placeholder HF model name (only loaded if transformers available).
- generation: sampling parameters + number of candidates per question.
- hardware.llm_gpu_id / hardware.rm_gpu_id: target device IDs (falls back to CPU if CUDA unavailable).
- train.batch_size / train.num_steps: loop control.

## Install (Minimal)
You only strictly need torch + pyyaml to run the dummy path:
```bash
pip install torch pyyaml
```
Optional (enables real components if environment supports):
```bash
pip install accelerate transformers vllm
```

## Run
From repo root:
```bash
python -m src.main --config configs/default.yaml
```

## Expected Console Output (Example)
```
[Step 0] Generated candidates per question: [3, 3]
[Step 0] Reward model scores shape: (2, 3)
Correctness not implemented.
Loss not implemented.
[Step 0] Completed batch with 2 questions.
```
Values and shapes depend on config (batch_size, n_samples_per_problem).

## Placeholders (Intentional)
- answer_parse.compute_final_correctness → raises NotImplementedError (caught in loop).
- losses.compute_joint_loss_placeholder → raises NotImplementedError (caught in loop).
- reward_model uses random features unless a HF model loads successfully.
- vLLM engine replaced by DummyLLM if library or model unavailable.

## Reward Model Options
- Default config now sets `model.rm_name: nvidia/AceMath-7B-RM`.
- If `transformers` and system resources allow, it loads the AceMath sequence classification head and produces scalar logits as scores.
- If loading fails (no GPU / OOM / offline / missing lib), it transparently falls back to a tiny random MLP; console will show a fallback message.

Example manual snippet (mirrors integrated loader):
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model = AutoModelForSequenceClassification.from_pretrained(
    "nvidia/AceMath-7B-RM", device_map="auto", num_labels=1,
    torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained("nvidia/AceMath-7B-RM", trust_remote_code=True)
```

Scoring pipeline builds a chat-style conversation: system + user(question) + assistant(candidate solution) per candidate.

## Where to Implement Next
- Implement correctness parsing / answer matching in `answer_parse.py`.
- Implement a real joint loss (e.g., policy + reward shaping) in `losses.py`.
- Replace dummy reward model forward pass with proper scoring.
- Introduce real dataset + dataloader.
- Add logging, checkpoints, evaluation, metrics.

## Safety / Resource Notes
- Current prototype avoids large model downloads by failing over to lightweight dummy classes.
- Multi-GPU separation is declarative (IDs in config); if only one device exists, both fall back to CPU.

## License
Placeholder (add a license file as needed).
