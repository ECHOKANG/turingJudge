"""批量评估编排器（Batch Orchestrator）。

用 asyncio + asyncio.to_thread 把 EvalSubagent 包装为可并发调度的协程。
- 通过 Semaphore 限制并发上限（默认 16，env 可调）
- 每个 case 独立超时 + 重试
- 不依赖 langfuse
- 提供同步入口 `run_batch_sync`，方便 Flask 同步路由直接调用
- 支持可选 RunLogger 收集 per-case 模块耗时
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_container import AgentConfig
from simulator import Persona

from evaluator.eval_subagent import EvalSubagent
from evaluator.resource_pool import ResourcePool
from evaluator.run_logger import CaseTimer, RunLogger
from evaluator.progress_logger import ProgressLogger


DEFAULT_MAX_CONCURRENCY = int(os.getenv("EVAL_MAX_CONCURRENCY", "16"))
DEFAULT_TIMEOUT = float(os.getenv("EVAL_CASE_TIMEOUT", "180"))
DEFAULT_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "1"))


class BatchOrchestrator:
    def __init__(
        self,
        pool: ResourcePool,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.pool = pool
        self.max_concurrency = max(1, max_concurrency)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.subagent = EvalSubagent(pool)

    async def _run_one(
        self,
        sem: asyncio.Semaphore,
        index: int,
        agent_config: AgentConfig,
        persona: Persona,
        metadata: Dict[str, Any],
        max_turns: int,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_logger: Optional[RunLogger] = None,
        progress_logger: Optional[ProgressLogger] = None,
    ) -> Dict[str, Any]:
        async with sem:
            if progress_logger:
                progress_logger.case_start(index, metadata.get("case_type", "unknown"), persona.identity or "")
            attempt = 0
            last_err: Optional[BaseException] = None
            timeout_logged = False
            t_start = time.time()
            timer = CaseTimer()
            while attempt <= self.max_retries:
                attempt_timer = CaseTimer()
                try:
                    effective_timeout = max(self.timeout, max_turns * 12 + 60)
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.subagent.run,
                            agent_config,
                            persona,
                            persona.goal,
                            max_turns,
                            f"case{index}_a{attempt}",
                            attempt_timer,
                        ),
                        timeout=effective_timeout,
                    )
                    timer = attempt_timer
                    eval_res = result["evaluation_result"]
                    payload = {
                        "index": index,
                        "agent_name": agent_config.name,
                        "case_type": metadata.get("case_type", "unknown"),
                        "expected_behavior": metadata.get("expected_behavior", ""),
                        "scenario": {
                            "identity": persona.identity,
                            "scene_premise": persona.scene_premise,
                            "conversation_goal": persona.goal,
                            "personality_traits": persona.personality_traits,
                            "speaking_style": persona.speaking_style,
                            "background": persona.background,
                            "agent_business_context": persona.agent_business_context or "",
                            "user_role_hint": persona.user_role_hint or "",
                            "plot_hooks": persona.plot_hooks or [],
                        },
                        "effective_system_prompt": result.get("effective_system_prompt", ""),
                        "overall_score": eval_res.overall_score,
                        "dimension_scores": eval_res.dimension_scores,
                        "evidence_turns": eval_res.evidence_turns,
                        "confidence": eval_res.confidence,
                        "rule_flags": eval_res.rule_flags,
                        "conversation_history": eval_res.conversation_history,
                        "qualitative_analysis": eval_res.qualitative_analysis,
                        "bad_cases": eval_res.bad_cases,
                        "loop_terminated": result.get("loop_terminated", False),
                        "_meta": {
                            "duration": round(time.time() - t_start, 2),
                            "attempts": attempt + 1,
                            "timings": timer.snapshot(),
                        },
                    }
                    if run_logger:
                        snap = timer.snapshot()
                        if snap.get('judge_call_count', 0) > 1:
                            snap['_warn_multi_judge'] = True
                        run_logger.add_case({
                            "index": index,
                            "case_id": f"case{index}",
                            "duration": round(time.time() - t_start, 3),
                            "attempts": attempt + 1,
                            "timings": snap,
                            "rule_flags": eval_res.rule_flags,
                            "overall_score": eval_res.overall_score,
                            "loop_terminated": result.get("loop_terminated", False),
                            "error": None,
                        })
                    if progress_logger:
                        progress_logger.case_done(index, payload["case_type"], eval_res.overall_score, time.time() - t_start)
                    if progress_cb:
                        try:
                            progress_cb(payload)
                        except Exception:
                            pass
                    return payload
                except (asyncio.TimeoutError, Exception) as exc:
                    last_err = exc
                    timer = attempt_timer
                    attempt += 1
                    if isinstance(exc, asyncio.TimeoutError):
                        if progress_logger:
                            progress_logger.case_timeout(index, metadata.get("case_type", "unknown"), effective_timeout)
                            timeout_logged = True
                        break
                    if attempt > self.max_retries:
                        break
                    await asyncio.sleep(min(2 ** attempt, 8))

            err_msg = f"{type(last_err).__name__}: {last_err}"
            tb = traceback.format_exception(type(last_err), last_err, last_err.__traceback__) if last_err else []
            if run_logger:
                run_logger.add_case({
                    "index": index,
                    "case_id": f"case{index}",
                    "duration": round(time.time() - t_start, 3),
                    "attempts": attempt,
                    "timings": timer.snapshot(),
                    "rule_flags": [],
                    "overall_score": 0,
                    "loop_terminated": False,
                    "error": err_msg,
                })
            if progress_logger and not timeout_logged:
                progress_logger.case_fail(index, metadata.get("case_type", "unknown"), err_msg, time.time() - t_start)
            payload = {
                "index": index,
                "agent_name": agent_config.name,
                "case_type": metadata.get("case_type", "error"),
                "expected_behavior": metadata.get("expected_behavior", ""),
                "scenario": {
                    "identity": persona.identity,
                    "scene_premise": persona.scene_premise,
                    "conversation_goal": persona.goal,
                    "personality_traits": persona.personality_traits,
                    "speaking_style": persona.speaking_style,
                    "background": persona.background,
                },
                "overall_score": 0,
                "dimension_scores": {},
                "evidence_turns": {},
                "confidence": {},
                "rule_flags": [f"执行异常（{self.max_retries + 1} 次尝试均失败）: {err_msg}"],
                "conversation_history": [],
                "qualitative_analysis": "",
                "bad_cases": [],
                "loop_terminated": False,
                "error": err_msg,
                "traceback": "".join(tb)[-2000:],
                "_meta": {
                    "duration": round(time.time() - t_start, 2),
                    "attempts": attempt,
                },
            }
            if progress_cb:
                try:
                    progress_cb(payload)
                except Exception:
                    pass
            return payload

    async def run_batch(
        self,
        agent_config: AgentConfig,
        persona_list: List[Tuple[Persona, Dict[str, Any]]],
        max_turns: int = 5,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_logger: Optional[RunLogger] = None,
        progress_logger: Optional[ProgressLogger] = None,
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(min(self.max_concurrency, max(1, len(persona_list))))
        if progress_logger:
            progress_logger.set_total(len(persona_list))
        tasks = [
            self._run_one(sem, i, agent_config, persona, meta, max_turns, progress_cb, run_logger, progress_logger)
            for i, (persona, meta) in enumerate(persona_list)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x.get("index", 0))
        if progress_logger:
            progress_logger.batch_done({"total_cases": len(results)})
        return results

    def run_batch_sync(
        self,
        agent_config: AgentConfig,
        persona_list: List[Tuple[Persona, Dict[str, Any]]],
        max_turns: int = 5,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_logger: Optional[RunLogger] = None,
        progress_logger: Optional[ProgressLogger] = None,
    ) -> List[Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("nested event loop")
        except RuntimeError:
            pass
        return asyncio.run(
            self.run_batch(agent_config, persona_list, max_turns, progress_cb, run_logger, progress_logger)
        )
