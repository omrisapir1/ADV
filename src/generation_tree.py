from __future__ import annotations

import math
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Deque
from collections import deque

THINK_STOP = "</think>"

try:
    from openai import AsyncOpenAI  # type: ignore
    from openai._exceptions import APIStatusError, RateLimitError  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore
    class APIStatusError(Exception):  # type: ignore
        pass
    class RateLimitError(Exception):  # type: ignore
        pass


MAX_SPLIT = 4
MIN_SPLIT = 2

@dataclass
class _Node:
    ctx: str
    think: str  # accumulated think content so far (may NOT include THINK_STOP)
    depth: int

class AsyncTreeOfThoughtSGLangEngineWrapper:
    """Entropy-based Tree-of-Thought generator using streaming logprobs.

    Config keys expected in generate_candidates **gen_cfg:
      think_entropy_threshold (float): entropy above which we branch. Default 3.0.
      token_prob_threshold (float): min normalized probability for token to spawn child. Default 0.20.
      think_max_new_tokens (int): cap on think streaming tokens per node. Required.
      answer_max_new_tokens (int): cap on answer greedy decoding tokens. Required.
      answer_stop (List[str]|None): optional stop strings for answer phase.
      max_depth (int): maximum tree depth. Default 10.
      max_nodes (int): safety cap on total processed nodes. Default 1000.

    Return structure: List[List[(text, phase_flag)]] mirroring other engines.
      phase_flag 0 => only think portion (stop not reached or budget exhausted)
      phase_flag 1 => think + answer appended (stop reached then greedy answer)
    """

    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        if AsyncOpenAI is None:
            raise ImportError("openai async client unavailable; install openai")
        sglang_config = sglang_config or {}
        base_url = sglang_config.get("base_url", "http://localhost:30000/v1")
        api_key = sglang_config.get("api_key", "EMPTY")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self._semaphore = asyncio.Semaphore(int(sglang_config.get("max_concurrency", 4)))
        self.max_retries = int(sglang_config.get("max_retries", 3))
        self.retry_sleep = float(sglang_config.get("retry_sleep", 5.0))

    def _compute_entropy(self, top_logprob_items: List[Any]) -> Tuple[float, List[Tuple[str, float]]]:
        probs: List[Tuple[str, float]] = []
        for itm in top_logprob_items:
            token = itm.token
            lp = itm.logprob
            p = math.exp(lp)
            probs.append((token, p))
        if not probs:
            return 0.0, []
        total = sum(p for _, p in probs)
        norm = [(t, p / total) for t, p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for _, p in norm)
        return entropy, norm

    async def _stream_one_think(self, node: _Node, *, think_entropy_threshold: float, token_prob_threshold: float, think_max_new_tokens: int) -> Tuple[List[_Node], Optional[str]]:
        """Stream think phase for a single node until branch or THINK_STOP or budget.

        Returns (children_nodes, finished_think_text or None).
          - children_nodes non-empty => branched; finished_think_text None.
          - THINK_STOP encountered => finished_think_text (think without stop).
          - budget exhausted => finished_think_text current think (no stop) to treat as think-only.
        """
        # Use completions with prompt instead of chat messages.
        prompt = node.ctx + node.think
        try:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                n=1,
                stream=True,
                logprobs=5,
                temperature=0.7,
                top_p=0.7,
                max_tokens=think_max_new_tokens,
                stop=[THINK_STOP],
                extra_body={"top_k": 40},
            )
        except Exception as e:
            print(f"[Tree] Stream start error: {e}")
            return [], node.think
        token_count = 0
        accumulated = node.think
        async for event in stream:
            print(event)
            choice = event.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            text = getattr(choice, "text", "") or ""
            prefix_before_token = accumulated
            accumulated += text
            # Extract top logprobs; completions may expose choice.logprobs.top_logprobs OR content[0].top_logprobs
            top_logprobs = []
            logprobs_obj = getattr(choice, "logprobs", None)
            if logprobs_obj is not None:
                if hasattr(logprobs_obj, "top_logprobs"):
                    top_logprobs = getattr(logprobs_obj, "top_logprobs", []) or []
                elif hasattr(logprobs_obj, "content"):
                    content_list = getattr(logprobs_obj, "content", [])
                    if content_list:
                        top_logprobs = getattr(content_list[0], "top_logprobs", []) or []
                elif isinstance(logprobs_obj, dict):
                    top_logprobs = logprobs_obj.get("top_logprobs", []) or []
            entropy, norm_probs = self._compute_entropy(top_logprobs)
            token_count += 1
            if entropy > think_entropy_threshold and norm_probs and THINK_STOP not in accumulated:
                children: List[_Node] = []
                for i, (tok, prob) in enumerate(norm_probs):
                    if (prob >= token_prob_threshold and i < MAX_SPLIT) or i < MIN_SPLIT:
                        children.append(_Node(ctx=node.ctx, think=prefix_before_token + tok, depth=node.depth + 1))
                if children:
                    return children, None
            if THINK_STOP in accumulated:
                return [], accumulated
            if finish_reason == "stop" or token_count >= think_max_new_tokens:
                return [], accumulated
        return [], accumulated

    async def _greedy_answer(self, base_prompt_plus_think: str, answer_max_new_tokens: int, answer_stop: Optional[List[str]]) -> str:
        prompt = base_prompt_plus_think + THINK_STOP
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    n=1,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=answer_max_new_tokens,
                    stop=answer_stop if answer_stop else None,
                )
                choice = resp.choices[0] if resp.choices else None
                if choice is None:
                    return ""
                return getattr(choice, "text", "") or ""
            except (APIStatusError, RateLimitError) as e:
                print(f"[Tree] Answer API error {e}; retry {attempt+1}/{self.max_retries}")
                await asyncio.sleep(self.retry_sleep)
            except Exception as e:
                print(f"[Tree] Answer generic error {e}; retry {attempt+1}/{self.max_retries}")
                await asyncio.sleep(self.retry_sleep)
        return ""

    async def generate_candidates(self, prompts: List[str], n_samples: int, **gen_cfg: Any) -> List[List[tuple[str, int]]]:
        think_entropy_threshold = gen_cfg.get("think_entropy_threshold", 3.0)
        token_prob_threshold = gen_cfg.get("token_prob_threshold", 0.20)
        think_max_new_tokens = gen_cfg.get("think_max_new_tokens")
        answer_max_new_tokens = gen_cfg.get("answer_max_new_tokens")
        answer_stop = gen_cfg.get("answer_stop")
        max_depth = int(gen_cfg.get("max_depth", 2))
        max_nodes = int(gen_cfg.get("max_nodes", 10))
        node_parallel_limit = int(gen_cfg.get("node_parallel_limit", 4))
        if think_max_new_tokens is None or answer_max_new_tokens is None:
            raise ValueError("think_max_new_tokens and answer_max_new_tokens must be provided in gen_cfg")

        async def _process_prompt(p: str) -> List[tuple[str, int]]:
            async with self._semaphore:
                results: List[tuple[str, int]] = []
                queue: Deque[_Node] = deque([_Node(ctx=p, think="", depth=0)])
                nodes_processed = 0
                while queue and len(results) < n_samples and nodes_processed < max_nodes:
                    # Build a batch of nodes to expand in parallel
                    batch: List[_Node] = []
                    while queue and len(batch) < node_parallel_limit:
                        batch.append(queue.popleft())
                    # Filter out over-depth nodes early
                    valid_batch = [n for n in batch if n.depth <= max_depth]
                    nodes_processed += len(valid_batch)
                    # Launch expansions concurrently
                    expand_tasks = [
                        asyncio.create_task(
                            self._stream_one_think(
                                n,
                                think_entropy_threshold=think_entropy_threshold,
                                token_prob_threshold=token_prob_threshold,
                                think_max_new_tokens=think_max_new_tokens,
                            )
                        ) for n in valid_batch
                    ]
                    if not expand_tasks:
                        continue
                    expand_results = await asyncio.gather(*expand_tasks)
                    answer_tasks: List[asyncio.Task] = []
                    answer_task_meta: List[Tuple[int, str]] = []  # (result_index, think_clean)
                    # Process each expansion
                    for node_obj, (children, finished_think) in zip(valid_batch, expand_results):
                        if len(results) >= n_samples:
                            break
                        if children:  # enqueue children for future parallel expansion
                            for c in children:
                                if c.depth <= max_depth:
                                    queue.append(c)
                            continue
                        if finished_think is None:
                            continue
                        has_stop = THINK_STOP in finished_think
                        if has_stop:
                            think_clean = finished_think.split(THINK_STOP, 1)[0]
                            # Defer greedy answer (parallelize)
                            answer_tasks.append(asyncio.create_task(
                                self._greedy_answer(
                                    node_obj.ctx + think_clean,
                                    answer_max_new_tokens=answer_max_new_tokens,
                                    answer_stop=answer_stop,
                                )
                            ))
                            answer_task_meta.append((len(results), think_clean))
                            # Placeholder; will replace after answers return
                            results.append((think_clean + THINK_STOP + "", 1))
                        else:
                            results.append((finished_think, 0))
                        if len(results) >= n_samples:
                            break
                    # Wait for all answer decodes for this batch, then fill them in
                    if answer_tasks:
                        answers = await asyncio.gather(*answer_tasks)
                        for (res_idx, think_clean), ans in zip(answer_task_meta, answers):
                            full_text = think_clean + THINK_STOP + (ans or "")
                            results[res_idx] = (full_text, 1)
                if len(results) < n_samples:
                    results.extend([("", 0)] * (n_samples - len(results)))
                return results[:n_samples]
        tasks = [asyncio.create_task(_process_prompt(p)) for p in prompts]
        return await asyncio.gather(*tasks)


def build_tree_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncTreeOfThoughtSGLangEngineWrapper:
    return AsyncTreeOfThoughtSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
