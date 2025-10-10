from typing import List, Dict, Any, Optional
import os
import asyncio
import random
from openai import AsyncOpenAI
from openai._exceptions import APIStatusError, RateLimitError


class AsyncSGLangEngineWrapper:
    """
    Async OpenAI-compatible client wrapper for an SGLang server.

    Server example:
      sglang-serve --model Qwen/Qwen2.5-Math-1.5B-Instruct \
                   --host 0.0.0.0 --port 30000 --dtype bfloat16 \
                   --kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.75

    Env example:
      export SGLANG_BASE_URL="http://localhost:30000/v1"
      export SGLANG_API_KEY="EMPTY"
    """

    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        sglang_config = sglang_config or {}
        base_url = sglang_config.get("base_url") or os.environ.get("SGLANG_BASE_URL", "http://localhost:30000/v1")
        api_key  = sglang_config.get("api_key")  or os.environ.get("SGLANG_API_KEY", "EMPTY")

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        # Optional: limit client-side concurrency to avoid overload
        self._semaphore = asyncio.Semaphore(sglang_config.get("max_concurrency", 16))

        # Simple retry/backoff config
        self._max_retries = sglang_config.get("max_retries", 3)
        self._base_backoff = sglang_config.get("base_backoff", 0.5)  # seconds

    async def _call_one(
        self,
        prompt: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        top_k: int,
        repetition_penalty: float,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """One async call for a single prompt, with retries."""
        payload_extra = {
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        if extra_body:
            payload_extra.update(extra_body)

        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._semaphore:
                    # Use raw completions since you already crafted your prompt
                    resp = await self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        n=n_samples,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_new_tokens,
                        stop=stop,
                        extra_body=payload_extra,
                    )
                return [c.text or "" for c in resp.choices]

            except (RateLimitError, APIStatusError) as e:
                if attempt > self._max_retries:
                    raise
                # Exponential backoff with jitter
                sleep_s = self._base_backoff * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                await asyncio.sleep(sleep_s)
            except Exception:
                # Non-retryable or unexpected errors bubble up
                raise

    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[str]]:
        """
        Async batch API: fires all prompts concurrently and returns
        List[List[str]] aligned with `prompts`.
        """
        temperature = float(gen_cfg.get("temperature", 0.9))
        top_p = float(gen_cfg.get("top_p", 0.95))
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 1024))
        repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.05))
        top_k = int(gen_cfg.get("top_k", 0))
        stop = gen_cfg.get("stop")  # e.g. ["</think>", "\n\n###"]
        extra_body = gen_cfg.get("extra_body")  # pass any SGLang-specific extras

        tasks = [
            asyncio.create_task(
                self._call_one(
                    prompt=p,
                    n_samples=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    extra_body=extra_body,
                )
            )
            for p in prompts
        ]
        results = await asyncio.gather(*tasks)
        return results

def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
