from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

class AceMathRewardModel:
    """Wrapper for AceMath (sequence classification) reward model.
    Exposes model + tokenizer + device. No fallback placeholder model.
    """
    def __init__(self, model_name: str, gpu_id: int):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def build_chat_inputs(self, question: str, solution: str) -> dict:
        chat = [
            {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]
        return self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=False,
            padding=True,
            truncation=True
        )


def load_reward_model(model_name: str, gpu_id: int) -> AceMathRewardModel:
    return AceMathRewardModel(model_name=model_name, gpu_id=gpu_id)


def score_solutions(questions: List[str], solutions: List[str], model: AceMathRewardModel, n_candidates: int) -> torch.Tensor:
    if len(solutions) != len(questions) * n_candidates:
        raise ValueError("Mismatch between flattened solutions and expected shape")

    # Collect all tokenized inputs
    all_inputs = []
    for qi, q in enumerate(questions):
        base = qi * n_candidates
        for k in range(n_candidates):
            sol = solutions[base + k]
            inputs = model.build_chat_inputs(q, sol)
            all_inputs.append(inputs)

    # Batch all inputs together
    batch_inputs = {
        'input_ids': torch.cat([inp['input_ids'] for inp in all_inputs], dim=0),
        'attention_mask': torch.cat([inp['attention_mask'] for inp in all_inputs], dim=0)
    }

    # Move to device
    batch_inputs = {k: v.to(model.model.device if hasattr(model.model, 'device') else model.device) for k, v in batch_inputs.items()}

    with torch.no_grad():
        out = model.model(**batch_inputs)
    logits = out.logits.float().squeeze(-1)
    return logits.view(len(questions), n_candidates)
