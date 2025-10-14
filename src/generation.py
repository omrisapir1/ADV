from __future__ import annotations

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
        answer_max_new_tokens: int,
    ) -> List[tuple[str, int]]:
        """
        Two-phase generation for a single prompt:
          - Phase 1: sample until </think>
          - Phase 2: greedy continuation to the end (or until answer_stop)
        Returns:
          List of (text, flag), flag=1 if Phase-2 executed, else 0.
        """
        THINK_STOP = "</think>"

        # ---------- Phase 1: sample until </think> ----------
        payload_extra_1 = {"top_k": think_top_k, "repetition_penalty": think_repetition_penalty}

        resp1 = await self.client.completions.create(
            model=self.model_name,
            prompt=base_prompt,
            n=n_samples,
            temperature=think_temperature,
            top_p=think_top_p,
            max_tokens=think_max_new_tokens,
            stop=[THINK_STOP],
            extra_body=payload_extra_1,
        )

        results: List[tuple[str, int]] = []

        # ---------- Phase 2: conditionally greedy per sample ----------
        for choice in resp1.choices:
            think_piece = (choice.text or "")
            finish_reason = getattr(choice, "finish_reason", None)

            # Only proceed to Phase 2 if we actually stopped on THINK_STOP
            if finish_reason != "stop":
                results.append((think_piece, 0))
                continue

            # Build continuation context ending with exactly one THINK_STOP
            if THINK_STOP in think_piece:
                think_clean = think_piece.split(THINK_STOP, 1)[0]
            else:
                think_clean = think_piece
            context = base_prompt + think_clean + THINK_STOP

            # Greedy params
            payload_extra_2 = {"top_k": 0, "repetition_penalty": 1.0}

            resp2 = await self.client.completions.create(
                model=self.model_name,
                prompt=context,
                n=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=answer_max_new_tokens,
                extra_body=payload_extra_2,
            )

            answer_text = (resp2.choices[0].text or "") if resp2.choices else ""
            full_text = think_clean + THINK_STOP + answer_text

            results.append((full_text, 1))

        return results


    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[tuple[str, int]]]:
        """
        Default behavior: TWO-PHASE generation.
        - Phase 1 (sampling): until </think>
        - Phase 2 (greedy): deterministic continuation
        Returns:
          List[List[(text, flag)]] aligned with `prompts`.
        """
        # Phase 1 (thinking) config
        think_temperature = float(gen_cfg.get("think_temperature", gen_cfg.get("temperature", 0.9)))
        think_top_p = float(gen_cfg.get("think_top_p", gen_cfg.get("top_p", 0.95)))
        think_top_k = int(gen_cfg.get("think_top_k", gen_cfg.get("top_k", 0)))
        think_repetition_penalty = float(gen_cfg.get("think_repetition_penalty", gen_cfg.get("repetition_penalty", 1.05)))
        think_max_new_tokens = int(gen_cfg.get("think_max_new_tokens", 512))

        # Phase 2 (answer) config
        answer_max_new_tokens = int(gen_cfg.get("answer_max_new_tokens", 512))

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
                    answer_max_new_tokens=answer_max_new_tokens,
                )
            )
            for p in prompts
        ]
        per_prompt_lists: List[List[tuple[str, int]]] = await asyncio.gather(*tasks)
        return per_prompt_lists


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
