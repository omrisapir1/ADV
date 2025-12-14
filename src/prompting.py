from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Provide step-by-step reasoning and put the final answer in \\boxed{}."
THINK_TOKEN = '</think>'

def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt_text




def clear_solution(full_solution: str) -> str:
    if THINK_TOKEN in full_solution:
        return full_solution[:full_solution.rfind(THINK_TOKEN)] + THINK_TOKEN
    if len(full_solution) < 50:
        return THINK_TOKEN
    ind = full_solution[:-50].rfind("\n")
    if ind == -1:
        return THINK_TOKEN
    return full_solution[:ind] + THINK_TOKEN



def build_prompts(questions: List[str], tokenizer: AutoTokenizer) -> List[str]:
    return [build_prompt(q, tokenizer) for q in questions]
