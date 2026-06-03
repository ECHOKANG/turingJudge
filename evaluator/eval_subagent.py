"""单 case 评估子代理（Subagent）。

把 app.py 里 `TuringEvaluationWeb.run_evaluation` 的逻辑搬到独立单元，
特点：
- 无共享可变状态（每次 run 都自带 container/simulator/judge 实例）
- 不依赖 langfuse（彻底砍掉 trace 装饰器和 trace 调用）
- 保留死循环 / 占位符 / 约束 / 拒答检测
- 返回结构化 dict，方便 orchestrator 聚合
"""
from __future__ import annotations

import re
import uuid
from contextlib import nullcontext
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from agent_container import AgentConfig
from agent_container.agent_container import scan_prompt_placeholders
from simulator import CaseGenerator, Persona
from prompt_loader import load_prompt

from evaluator.resource_pool import ResourcePool


# ============== 全局配置（与 app.py 保持一致） ==============
_eval_cfg = load_prompt("evaluation")
LOOP_SIMILARITY_THRESHOLD = _eval_cfg["loop_detection"]["similarity_threshold"]
LOOP_CONSECUTIVE_LIMIT = _eval_cfg["loop_detection"]["consecutive_limit"]
REFUSAL_PHRASES = _eval_cfg["refusal_phrases"]
REFUSAL_THRESHOLD = _eval_cfg["refusal_threshold"]
OPENING_MESSAGE = _eval_cfg["opening_message"]
END_SIGNAL = _eval_cfg["end_signal"]


# ============== 检测工具函数 ==============
def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _similarity(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def detect_loop(agent_outputs: List[str]) -> Tuple[bool, int]:
    if len(agent_outputs) < 2:
        return False, 0
    consecutive = 1
    for i in range(len(agent_outputs) - 1, 0, -1):
        sim = _similarity(agent_outputs[i], agent_outputs[i - 1])
        if sim >= LOOP_SIMILARITY_THRESHOLD:
            consecutive += 1
        else:
            break
    return consecutive >= LOOP_CONSECUTIVE_LIMIT, consecutive


_CHAR_LIMIT_PATTERNS = [
    r"控制在.{0,8}?(?P<n>\d{2,3}).{0,4}?字",
    r"不.{0,4}?超过.{0,8}?(?P<n>\d{2,3}).{0,4}?字",
    r"(?P<n>\d{2,3}).{0,4}?字以?内",
    r"约.{0,4}?(?P<n>\d{2,3}).{0,4}?字",
]


def extract_char_limit(system_prompt: str) -> int:
    if not system_prompt:
        return 0
    for pat in _CHAR_LIMIT_PATTERNS:
        m = re.search(pat, system_prompt)
        if m:
            try:
                return int(m.group("n"))
            except (ValueError, KeyError):
                continue
    return 0


def detect_constraint_violations(
    agent_outputs: List[str],
    system_prompt: str,
) -> List[str]:
    flags: List[str] = []
    limit = extract_char_limit(system_prompt)
    if limit > 0 and agent_outputs:
        threshold = int(limit * 1.2)
        violations = [
            (i, len(o)) for i, o in enumerate(agent_outputs) if o and len(o) > threshold
        ]
        if violations:
            sample = "; ".join(f"第{i}轮 {ln}字" for i, ln in violations[:3])
            flags.append(
                f"违反『{limit}字以内』约束: {len(violations)}/{len(agent_outputs)} 轮超长（{sample}）"
            )

    if len(agent_outputs) >= 3:
        repeat_pairs = 0
        seen_pairs = set()
        for i in range(len(agent_outputs)):
            for j in range(i + 1, len(agent_outputs)):
                if (i, j) in seen_pairs:
                    continue
                if _similarity(agent_outputs[i], agent_outputs[j]) >= 0.85:
                    repeat_pairs += 1
                    seen_pairs.add((i, j))
        if repeat_pairs >= 2:
            flags.append(
                f"违反『避免重复回复』约束: 检测到 {repeat_pairs} 对高相似度回复（≥85%）"
            )

    return flags


# ============== Subagent 主体 ==============
class EvalSubagent:
    """单 case 评估闭环。无共享状态，可并发安全调用。"""

    def __init__(self, pool: ResourcePool):
        self.pool = pool

    def run(
        self,
        agent_config: AgentConfig,
        persona: Persona,
        conversation_goal: str,
        max_turns: int = 5,
        case_namespace: str = "",
        timer: "Any" = None,
    ) -> Dict[str, Any]:
        """执行单 case 评估。

        Args:
            case_namespace: 用于隔离 agent_name/persona_name，避免并发命名冲突。
                            为空时自动生成。

        Returns:
            {
                "test_case": ...,
                "conversation_history": [...],
                "evaluation_result": EvaluationResult,
                "rule_flags": [...],
                "loop_terminated": bool,
            }
        """
        ns = case_namespace or uuid.uuid4().hex[:8]

        # 计时器（可选）
        def _t(name):
            if timer is not None and hasattr(timer, 'measure'):
                return timer.measure(name)
            return nullcontext()

        # 每个 case 一组隔离的组件
        container = self.pool.make_agent_container()
        simulator = self.pool.make_user_simulator()
        judge = self.pool.make_judge()

        # 用 namespace 防止并发 register 冲突
        scoped_agent_cfg = AgentConfig(
            name=f"{agent_config.name}__{ns}",
            system_prompt=agent_config.system_prompt,
            model=agent_config.model,
            api_key=agent_config.api_key,
            base_url=agent_config.base_url,
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            context_window=agent_config.context_window,
            metadata=dict(agent_config.metadata),
            context_variables=dict(agent_config.context_variables),
        )
        scoped_persona = Persona(
            name=f"{persona.name}__{ns}",
            personality_traits=list(persona.personality_traits),
            speaking_style=persona.speaking_style,
            goal=persona.goal,
            constraints=dict(persona.constraints),
            background=persona.background,
            identity=persona.identity,
            scene_premise=persona.scene_premise,
            agent_business_context=persona.agent_business_context,
            user_role_hint=persona.user_role_hint,
            plot_hooks=list(persona.plot_hooks),
        )

        container.register_agent(scoped_agent_cfg)
        simulator.register_persona(scoped_persona)
        # 注意：故意不调用 create_prefix_cache —— 它在并发场景对吞吐没有帮助，
        # 反而会让每个 case 在 Ark 上多创建一次 response 资源，徒增网络往返。

        rule_flags: List[str] = []

        # ① 占位符扫描
        placeholder_findings = scan_prompt_placeholders(
            scoped_agent_cfg.effective_system_prompt
        )
        if placeholder_findings:
            samples = ", ".join(
                f"'{txt}'({desc})" for txt, desc in placeholder_findings[:3]
            )
            rule_flags.append(
                f"Agent system prompt 中存在未填充变量/占位符: {samples}"
                + ("，..." if len(placeholder_findings) > 3 else "")
            )

        case_gen = CaseGenerator()
        test_case = case_gen.generate_llm_simulation_case(
            conversation_goal=conversation_goal,
            persona_constraints=scoped_persona.constraints,
        )

        conversation_history: List[Dict[str, str]] = []
        current_message: str | None = None
        agent_outputs: List[str] = []
        loop_terminated = False

        for turn in range(max_turns):
            if turn == 0:
                with _t('agent_call'):
                    agent_response = container.generate_response(
                        agent_name=scoped_agent_cfg.name,
                        user_message=OPENING_MESSAGE,
                        conversation_history=conversation_history,
                        trace_collector=None,
                    )
                conversation_history.append(
                    {"role": "assistant", "content": agent_response}
                )
                agent_outputs.append(agent_response)
            else:
                if not current_message:
                    current_message = OPENING_MESSAGE
                with _t('simulator_call'):
                    user_response = simulator.generate_response(
                        persona_name=scoped_persona.name,
                        agent_message=current_message,
                        conversation_history=conversation_history,
                        trace_collector=None,
                    )
                if END_SIGNAL in user_response:
                    user_response = user_response.replace(END_SIGNAL, "").strip()
                    if user_response:
                        conversation_history.append(
                            {"role": "user", "content": user_response}
                        )
                    rule_flags.append(f"用户在第{turn}轮主动结束对话")
                    break
                conversation_history.append({"role": "user", "content": user_response})

                with _t('agent_call'):
                    agent_response = container.generate_response(
                        agent_name=scoped_agent_cfg.name,
                        user_message=user_response,
                        conversation_history=conversation_history[:-1],
                        trace_collector=None,
                    )
                conversation_history.append(
                    {"role": "assistant", "content": agent_response}
                )
                agent_outputs.append(agent_response)

            current_message = agent_response

            # ② 死循环检测
            is_loop, consecutive = detect_loop(agent_outputs)
            if is_loop:
                rule_flags.append(
                    f"检测到 Agent 死循环（连续{consecutive}轮输出相似度≥{LOOP_SIMILARITY_THRESHOLD}），"
                    f"已在第{turn}轮提前终止评估"
                )
                loop_terminated = True
                break

        # ③ 过度拒答启发式
        refusal_hits = sum(
            1
            for o in agent_outputs
            if any(p in (o or "") for p in REFUSAL_PHRASES)
        )
        if refusal_hits >= REFUSAL_THRESHOLD:
            rule_flags.append(
                f"检测到过度拒答/推诿话术 {refusal_hits} 次（出现『稍后答复/我去问同事』等表达），"
                f"未给出实质回答"
            )

        # ④ 约束违反检测
        constraint_flags = detect_constraint_violations(
            agent_outputs, scoped_agent_cfg.system_prompt
        )
        rule_flags.extend(constraint_flags)

        # ⑤ Judge 评估
        with _t('judge_call'):
            evaluation_result = judge.evaluate(
                case_id=test_case.case_id,
                agent_name=agent_config.name,  # 报告里展示原名
                conversation_history=conversation_history,
                agent_system_prompt=scoped_agent_cfg.effective_system_prompt,
                conversation_goal=conversation_goal,
                trace_collector=None,
                rule_flags=rule_flags,
            )

        return {
            "test_case": test_case,
            "conversation_history": conversation_history,
            "evaluation_result": evaluation_result,
            "rule_flags": rule_flags,
            "loop_terminated": loop_terminated,
            "effective_system_prompt": scoped_agent_cfg.effective_system_prompt,
        }
