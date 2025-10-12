from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful math reasoning assistant. Provide step-by-step reasoning and put the final answer in \\boxed{}."},
        {"role": "user", "content": question}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt_text


def build_prompts(questions: List[str], tokenizer: AutoTokenizer) -> List[str]:
    return [build_prompt(q, tokenizer) for q in questions]
