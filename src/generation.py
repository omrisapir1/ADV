import asyncio
import random
import os
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from openai._exceptions import APIStatusError, RateLimitError

THINK_STOP = "</think>"

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

    async def _completions_call(
        self,
        prompt: str,
        n: int,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        top_k: int,
        repetition_penalty: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        payload_extra = {"top_k": top_k, "repetition_penalty": repetition_penalty}
        if extra_body:
            payload_extra.update(extra_body)

        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._semaphore:
                    resp = await self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        n=n,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        extra_body=payload_extra,
                    )
                return [c.text or "" for c in resp.choices]

            except (RateLimitError, APIStatusError):
                if attempt > self._max_retries:
                    raise
                # Exponential backoff with jitter
                sleep_s = self._base_backoff * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                await asyncio.sleep(sleep_s)

    @staticmethod
    def _ensure_think_stop_in_context(prefix: str, think_piece: str) -> Tuple[str, str]:
        """
        Make sure the continuation context ends with the THINK_STOP token exactly once.
        Returns (context_with_stop, clean_think_text_without_stop).
        """
        text = think_piece
        # Strip any accidental duplicates or partials and re-add a single terminator.
        if THINK_STOP in text:
            # Keep content up to the first occurrence; everything after is for phase 2
            text = text.split(THINK_STOP, 1)[0]
        context = prefix + text + THINK_STOP
        return context, text

    async def _two_phase_for_one_prompt(
        self,
        base_prompt: str,
        *,
        n_samples: int,
        # Phase 1 (thinking) params
        think_temperature: float,
        think_top_p: float,
        think_max_new_tokens: int,
        think_top_k: int,
        think_repetition_penalty: float,
        think_extra_body: Optional[Dict[str, Any]],
        # Phase 2 (answer) params
        answer_max_new_tokens: int,
        answer_stop: Optional[List[str]],
        answer_extra_body: Optional[Dict[str, Any]],
        # Return control
        return_split: bool,
    ) -> List[str] | List[Dict[str, str]]:
        """
        Two-phase generation for a single prompt:
          - Phase 1: sample until </think>
          - Phase 2: greedy continuation to the end (or until answer_stop)
        """
        # ------ Phase 1: sample until </think> ------
        phase1 = await self._completions_call(
            prompt=base_prompt,
            n=n_samples,
            temperature=think_temperature,
            top_p=think_top_p,
            max_tokens=think_max_new_tokens,
            stop=[THINK_STOP],  # critical: cut exactly at </think>
            top_k=think_top_k,
            repetition_penalty=think_repetition_penalty,
            extra_body=think_extra_body,
        )

        # ------ Phase 2: greedy continuation for each sample ------
        outputs = []
        for t in phase1:
            # Build exact continuation context: <prompt> + <think> + </think>
            context, think_clean = self._ensure_think_stop_in_context(base_prompt, t)

            # Greedy / deterministic params
            greedy_temperature = 0.0
            greedy_top_p = 1.0
            greedy_top_k = 0
            greedy_rep = 1.0

            ans_list = await self._completions_call(
                prompt=context,
                n=1,  # one greedy continuation per thought
                temperature=greedy_temperature,
                top_p=greedy_top_p,
                max_tokens=answer_max_new_tokens,
                stop=answer_stop,
                top_k=greedy_top_k,
                repetition_penalty=greedy_rep,
                extra_body=answer_extra_body,
            )
            answer = ans_list[0] if ans_list else ""

            if return_split:
                outputs.append({"think": think_clean, "answer": answer, "full": think_clean + THINK_STOP + answer})
            else:
                outputs.append(think_clean + THINK_STOP + answer)

        return outputs

    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[str]]:
        """
        Default behavior: TWO-PHASE generation.
        - Phase 1 (sampling): until </think>
        - Phase 2 (greedy): deterministic continuation

        Config keys (with defaults):
          think_temperature: 0.9
          think_top_p: 0.95
          think_top_k: 0
          think_repetition_penalty: 1.05
          think_max_new_tokens: 512
          answer_max_new_tokens: 512
          answer_stop: None or list[str]
          extra_body_think: dict | None
          extra_body_answer: dict | None
          return_split: False  # if True, returns dicts with {think, answer, full}
        """
        # Phase 1 (thinking) config
        think_temperature = float(gen_cfg.get("think_temperature", gen_cfg.get("temperature", 0.9)))
        think_top_p = float(gen_cfg.get("think_top_p", gen_cfg.get("top_p", 0.95)))
        think_top_k = int(gen_cfg.get("think_top_k", gen_cfg.get("top_k", 0)))
        think_repetition_penalty = float(gen_cfg.get("think_repetition_penalty", gen_cfg.get("repetition_penalty", 1.05)))
        think_max_new_tokens = int(gen_cfg.get("think_max_new_tokens", 512))
        extra_body_think = gen_cfg.get("extra_body_think", gen_cfg.get("extra_body"))

        # Phase 2 (answer) config
        answer_max_new_tokens = int(gen_cfg.get("answer_max_new_tokens", 512))
        answer_stop = gen_cfg.get("answer_stop")  # e.g. ["\n\n###"] or None
        extra_body_answer = gen_cfg.get("extra_body_answer")

        return_split = bool(gen_cfg.get("return_split", False))

        tasks = [
            asyncio.create_task(
                self._two_phase_for_one_prompt(
                    p,
                    n_samples=n_samples,
                    think_temperature=think_temperature,
                    think_top_p=think_top_p,
                    think_max_new_tokens=think_max_new_tokens,
                    think_top_k=think_top_k,
                    think_repetition_penalty=think_repetition_penalty,
                    think_extra_body=extra_body_think,
                    answer_max_new_tokens=answer_max_new_tokens,
                    answer_stop=answer_stop,
                    answer_extra_body=extra_body_answer,
                    return_split=return_split,
                )
            )
            for p in prompts
        ]
        per_prompt_lists = await asyncio.gather(*tasks)
        # Shape: List[ List[str|dict] ], aligned with `prompts`
        return per_prompt_lists


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
