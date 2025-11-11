import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# —- User parameters —-
MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
OUTPUT_MODEL = "omrisap/Qwen2.5-Math-1.5B-5K-SFT-think"
MAX_LENGTH = 4500
DATASET = "omrisap/6K-think-SFT-math"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# If your tokenizer doesn’t have a pad token, set it to eos:
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

new_tokens = ["</think>"]
num_added = tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))

# Load dataset
dataset = load_dataset(DATASET)

# Preprocess: convert each example into prompt/completion format
SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Provide step-by-step reasoning and put the final answer in \\boxed{}."


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


def preprocess(example):
    problem = build_prompt(str(example["problem"]), tokenizer)
    solution = str(example["sol_with_think"])
    # Structure for prompt-completion format (works with SFTTrainer)
    return {
        "prompt": problem,
        "completion": solution
    }


# Remove unused columns and map
dataset = dataset.map(preprocess, remove_columns=dataset.column_names['train'])

# Configure training arguments (correct parameter names)
training_args = SFTConfig(
    output_dir="sft_model",
    dataset_text_field=None,  # Since using prompt/completion, None = auto-detect
    max_length=MAX_LENGTH,  # ✅ correct param name
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=10,
    save_steps=5000,
    save_total_limit=2,
    completion_only_loss=True,
    push_to_hub=False,
    packing=False  # if you don’t want to pack shorter sequences
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    processing_class=tokenizer  # specify your tokenizer for preprocessing
)

# Start training
trainer.train()