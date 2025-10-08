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
        tokenized = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=False,
            padding=True,
            truncation=True
        )

        # Handle case where apply_chat_template returns just input_ids tensor
        if isinstance(tokenized, torch.Tensor):
            # Create attention mask manually
            attention_mask = torch.ones_like(tokenized)
            return {
                'input_ids': tokenized,
                'attention_mask': attention_mask
            }
        else:
            # If it returns a dict, return as is
            return tokenized


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

    # Handle tensor concatenation with padding
    if all_inputs:
        # Ensure all tensors are 2D and find max length
        input_ids_list = []
        attention_mask_list = []
        max_length = 0

        for inp in all_inputs:
            input_ids = inp['input_ids']
            attention_mask = inp['attention_mask']

            # Ensure tensors are 2D (batch_size=1, seq_len)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            max_length = max(max_length, input_ids.size(1))

        # Pad all tensors to max_length
        padded_input_ids = []
        padded_attention_masks = []

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            current_length = input_ids.size(1)
            if current_length < max_length:
                # Pad with tokenizer.pad_token_id (usually 0 for input_ids)
                pad_length = max_length - current_length
                input_ids_padded = torch.cat([
                    input_ids,
                    torch.zeros(1, pad_length, dtype=input_ids.dtype, device=input_ids.device)
                ], dim=1)
                attention_mask_padded = torch.cat([
                    attention_mask,
                    torch.zeros(1, pad_length, dtype=attention_mask.dtype, device=attention_mask.device)
                ], dim=1)
            else:
                input_ids_padded = input_ids
                attention_mask_padded = attention_mask

            padded_input_ids.append(input_ids_padded)
            padded_attention_masks.append(attention_mask_padded)

        batch_inputs = {
            'input_ids': torch.cat(padded_input_ids, dim=0),
            'attention_mask': torch.cat(padded_attention_masks, dim=0)
        }
    else:
        # Handle empty case
        batch_inputs = {
            'input_ids': torch.empty(0, 0, dtype=torch.long),
            'attention_mask': torch.empty(0, 0, dtype=torch.long)
        }

    # Move to device
    batch_inputs = {k: v.to(model.model.device if hasattr(model.model, 'device') else model.device) for k, v in batch_inputs.items()}

    with torch.no_grad():
        out = model.model(**batch_inputs)
    logits = out.logits.float().squeeze(-1)
    return logits.view(len(questions), n_candidates)
