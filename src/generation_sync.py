from __future__ import annotations

import time
import re
from typing import List, Dict, Any, Optional, Tuple

# Optional imports with fallbacks to satisfy static analysis when libs not installed.
try:
    from openai import OpenAI  # type: ignore
    from openai._exceptions import APIStatusError, RateLimitError  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    class APIStatusError(Exception):  # type: ignore
        pass
    class RateLimitError(Exception):  # type: ignore
        pass

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

THINK_STOP = "</think>"

class SyncSGLangEngineWrapper:
    """Synchronous (serial) SGLang engine wrapper mirroring AsyncSGLangEngineWrapper API.

    generate_candidates returns the same structure: List[List[tuple[str, int]]]
    where each inner list is length n_samples and each tuple is (full_text, phase_flag)
    phase_flag: 0 => only think phase completed (or already contained boxed answer)
                1 => think + answer phases concatenated
    """

    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        sglang_config = sglang_config or {}
        base_url = "http://localhost:30000/v1"
        api_key = "EMPTY"
        if OpenAI is None:
            raise ImportError("openai package not available; cannot build synchronous engine")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_retries = int(sglang_config.get("max_retries", 3))
        self.retry_sleep = float(sglang_config.get("retry_sleep", 5.0))

    def hot_swap(self, tmp_weights_path: str):
        url = "http://localhost:30000/update_weights_from_disk"
        data = {"model_path": tmp_weights_path}
        if requests is None:
            print("[SyncEngine] requests unavailable; hot swap skipped")
            return
        try:
            requests.post(url, json=data, timeout=10)
        except Exception as e:
            print(f"[SyncEngine] Hot swap request failed: {e}")

    def _completion_with_retries(self, **kwargs):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return self.client.completions.create(**kwargs)
            except (APIStatusError, RateLimitError) as e:
                last_err = e
                print(f"[SyncEngine] API error attempt {attempt+1}/{self.max_retries}: {e}; sleeping {self.retry_sleep}s")
                time.sleep(self.retry_sleep)
            except Exception as e:
                last_err = e
                print(f"[SyncEngine] Generic error attempt {attempt+1}/{self.max_retries}: {e}; sleeping {self.retry_sleep}s")
                time.sleep(self.retry_sleep)
        print(f"[SyncEngine] Failed after {self.max_retries} attempts: {last_err}")
        # Return an object with empty choices shape to keep flow; mimic openai response minimal.
        class _Dummy:  # minimal stub
            choices: List
            def __init__(self):
                self.choices = []
        return _Dummy()

    def _two_phase_for_one_prompt(
        self,
        base_prompt: str,
        *,
        n_samples: int,
        think_temperature: float,
        think_top_p: float,
        think_max_new_tokens: int,
        think_top_k: int,
        think_repetition_penalty: float,
        answer_max_new_tokens: int,
        answer_stop: List[str],
    ) -> List[tuple[str, int]]:
        payload_extra_1 = {"top_k": think_top_k, "repetition_penalty": think_repetition_penalty}
        resp1 = self._completion_with_retries(
            model=self.model_name,
            prompt=base_prompt,
            n=n_samples,
            temperature=think_temperature,
            top_p=think_top_p,
            max_tokens=think_max_new_tokens,
            stop=[THINK_STOP],
            extra_body=payload_extra_1,
        )
        # Prepare results list; if resp1 has no choices produce empty list.
        choices_phase1 = getattr(resp1, "choices", [])
        results: List[tuple[str, int]] = [("", 0)] * len(choices_phase1)
        # Determine which need second phase.
        phase2_items: List[Tuple[int, str, str]] = []
        for idx, choice in enumerate(choices_phase1):
            think_piece = (getattr(choice, "text", "") or "")
            finish_reason = getattr(choice, "finish_reason", None)
            # If not stop OR already contains boxed answer -> finalize
            if finish_reason != "stop" or re.findall(r"\\boxed\\s*\{(.*?)\}", think_piece or "", flags=re.DOTALL):
                results[idx] = (think_piece, 0)
                continue
            think_clean = think_piece.split(THINK_STOP, 1)[0] if THINK_STOP in think_piece else think_piece
            context = base_prompt + think_clean + THINK_STOP
            phase2_items.append((idx, think_clean, context))
        if not phase2_items:
            return results
        payload_extra_2 = {"top_k": 0, "repetition_penalty": 1.0}
        for idx, think_clean, ctx in phase2_items:
            resp2 = self._completion_with_retries(
                model=self.model_name,
                prompt=ctx,
                n=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=answer_max_new_tokens,
                stop=answer_stop if answer_stop else None,
                extra_body=payload_extra_2,
            )
            answer_text = ""
            if getattr(resp2, "choices", None):
                answer_text = (getattr(resp2.choices[0], "text", "") or "")
            full_text = think_clean + THINK_STOP + answer_text
            results[idx] = (full_text, 1)
        return results

    def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[tuple[str, int]]]:
        think_temperature = gen_cfg.get("think_temperature")
        think_top_p = gen_cfg.get("think_top_p")
        think_top_k = gen_cfg.get("think_top_k")
        think_repetition_penalty = gen_cfg.get("think_repetition_penalty")
        think_max_new_tokens = gen_cfg.get("think_max_new_tokens")
        answer_max_new_tokens = gen_cfg.get("answer_max_new_tokens")
        answer_stop = gen_cfg.get("answer_stop")
        out: List[List[tuple[str, int]]] = []
        for p in prompts:
            try:
                res = self._two_phase_for_one_prompt(
                    p,
                    n_samples=n_samples,
                    think_temperature=think_temperature,
                    think_top_p=think_top_p,
                    think_max_new_tokens=think_max_new_tokens,
                    think_top_k=think_top_k,
                    think_repetition_penalty=think_repetition_penalty,
                    answer_max_new_tokens=answer_max_new_tokens,
                    answer_stop=answer_stop,
                )
            except Exception as e:
                print(f"[SyncEngine] Fatal error for prompt (truncated) {p[:60]!r}: {e}")
                res = []
            out.append(res)
        return out


def build_sglang_engine_sync(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> SyncSGLangEngineWrapper:
    return SyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
