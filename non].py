import asyncio
import math
from openai import AsyncOpenAI

# Adjust these to your setup
BASE_URL = "http://localhost:3000/v1"
API_KEY = "EMPTY"          # SGLang usually ignores this, but OpenAI client requires something
MODEL_NAME = "omrisap/Qwen2.5-Math-1.5B-5K-SFT-think"     # Or whatever you passed as --model-path / model name in sglang

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

def entropy_from_top_logprobs(token_info):
    """
    Compute entropy H = -Σ p_i log p_i from top_logprobs for a single position.
    Uses natural log; result is in 'nats'.
    """
    if not token_info.top_logprobs:
        return None

    # Convert logprobs to probabilities
    probs = [math.exp(t.logprob) for t in token_info.top_logprobs]
    Z = sum(probs)
    if Z == 0:
        return None

    probs = [p / Z for p in probs]
    return -sum(p * math.log(p) for p in probs)

async def main():
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain entropy in 2 short sentences."},
        ],
        max_tokens=64,
        temperature=0.7,
        # logprob controls per-token log probabilities; top_logprobs specifies how many to return
        # SGLang mirrors the OpenAI logprobs/top_logprobs behavior for chat completions. :contentReference[oaicite:0]{index=0}
        logprobs=True,
        top_logprobs=20,   # ask for top 20 alternatives per position (0–20 allowed in newer APIs). :contentReference[oaicite:1]{index=1}
    )

    choice = resp.choices[0]
    text = choice.message.content
    logprobs_obj = choice.logprobs

    avg_entropy = None
    if logprobs_obj and logprobs_obj.content:
        entropies = []
        for token_info in logprobs_obj.content:
            h = entropy_from_top_logprobs(token_info)
            if h is not None:
                entropies.append(h)

        if entropies:
            avg_entropy = sum(entropies) / len(entropies)

    print("=== MODEL OUTPUT ===")
    print(text)
    print("\n=== STATS ===")
    print(f"Num tokens with logprobs: {len(logprobs_obj.content) if logprobs_obj and logprobs_obj.content else 0}")
    print(f"Average entropy (nats, from top-20): {avg_entropy}")

if __name__ == "__main__":
    asyncio.run(main())
