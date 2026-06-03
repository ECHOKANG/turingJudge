"""
PersonaPoolAllocator: 把前端 V2 配置好的"用户群体分布"按权重展开为 N 个 case,
输出 schema 与 CaseAllocator.allocate() 完全一致,可被 cases_to_personas 与
batch 报告生成器无缝消费。

与 CaseAllocator 的关键差异:
- 不调 LLM,纯本地按权重采样,几乎零成本/零延迟
- type 字段映射自 persona.id (而不是 7 大风险类别),报告里"按类型分布"自动
  改为"按用户群体分布"
- 仍可可选地用一次 LLM 生成对话目标,但默认走"用户提供 goals + 轮询采样"

可观测:加 @observe(name="persona_pool_allocator", as_type="generation"),
让 Langfuse trace 上能看到这一步是怎么把 personas + goals 展开成 N 个 case 的。
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from langfuse import Langfuse, observe
from simulator import Persona
from usage_logger import log_event


# 默认 5 类 persona,与 templates/index.html 中的 personas 数组保持一致,
# 仅在前端没传时兜底。
DEFAULT_PERSONA_POOL: List[Dict[str, Any]] = [
    {"id": "novice",  "name": "焦虑新手",
     "traits": ["表达口语化", "容易着急", "问题描述模糊", "会反复确认"],
     "speaking_style": "短句、口语化、带情绪",
     "weight": 40},
    {"id": "expert",  "name": "理性专家",
     "traits": ["用词专业", "问题精准", "追问技术细节"],
     "speaking_style": "结构化、专业术语、追问到底",
     "weight": 25},
    {"id": "repeat",  "name": "急躁重复型",
     "traits": ["短句", "重复同一问题", "容易抱怨"],
     "speaking_style": "重复、抱怨、口气急",
     "weight": 20},
    {"id": "polite",  "name": "礼貌普通用户",
     "traits": ["表达完整", "礼貌", "配合度高"],
     "speaking_style": "礼貌、配合、表达完整",
     "weight": 15},
    {"id": "adv",     "name": "对抗测试者",
     "traits": ["故意刁难", "尝试绕过限制", "追问边界"],
     "speaking_style": "试探、刁难、绕弯",
     "weight": 10},
]


class PersonaPoolAllocator:
    """按 V2 配置的用户群体分布,把 N 个测试用例按权重分到不同 persona 上。"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    @observe(name="persona_pool_allocator", as_type="generation")
    def allocate(
        self,
        personas: List[Dict[str, Any]],
        total_n: int,
        goals: List[str],
        identity: str = "",
        scene_premise: str = "",
        agent_system_prompt: str = "",
        trace_collector: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            personas: 前端传来的 personas 数组,每项至少含 id/name/traits/weight
            total_n: 总测试用例数
            goals: 对话目标列表,会按轮询方式分给每个 case
            identity / scene_premise: 全局兜底身份与场景前提
            agent_system_prompt: 用于推导 Agent 业务上下文摘要 + user 真实角色提示

        Returns:
            {
                "analysis": "按 Persona Pool 分布展开...",
                "allocation": {persona_id: count, ...},
                "cases": [{type, identity, scene_premise, conversation_goal,
                           personality_traits, speaking_style, background,
                           expected_behavior}, ...]
            }
        """
        pool = self._normalize_pool(personas)
        if not pool:
            pool = DEFAULT_PERSONA_POOL

        if not goals:
            goals = ["请你向客服提出一个真实的问题,看 Agent 如何应对"]

        # —— 关键改造：从 system_prompt 推导 Agent 业务上下文与"用户应有身份" ——
        agent_ctx = self._extract_agent_context(agent_system_prompt or "")
        # identity 字段（前端传入或自动推导）：默认显示成"对方是XXX，你是YYY"
        agent_role_label = identity or agent_ctx["agent_role"]
        scene_label = scene_premise or agent_ctx["default_scene"]

        # 1. 按权重分配数量(最大余数法,保证总和精确 = N)
        allocation = self._allocate_counts(pool, total_n)
        log_event(
            "PersonaPoolAllocator.allocate",
            "ENTER",
            total_n=total_n,
            pool_size=len(pool),
            goals_count=len(goals),
            allocation=json.dumps(allocation, ensure_ascii=False),
        )

        # 2. 展开为 cases 列表
        cases: List[Dict[str, Any]] = []
        goal_idx = 0
        for p in pool:
            count = allocation.get(p["id"], 0)
            for _ in range(count):
                goal = goals[goal_idx % len(goals)]
                goal_idx += 1
                cases.append(self._build_case(
                    p, goal,
                    agent_role_label=agent_role_label,
                    scene_premise=scene_label,
                    agent_ctx=agent_ctx,
                ))

        # 3. 打散顺序(避免同 persona 连续 4 个并发跑)
        self._rng.shuffle(cases)

        # 4. 生成可读的 analysis 文案
        analysis = self._build_analysis(pool, allocation, total_n, len(goals))
        log_event(
            "PersonaPoolAllocator.allocate",
            "EXIT",
            cases=len(cases),
        )

        result = {
            "analysis": analysis,
            "allocation": allocation,
            "cases": cases,
            "_pool_meta": {
                # 给报告 / Langfuse 用,记录每个 persona 的中文名,方便展示
                p["id"]: {"name": p["name"], "weight": p["weight"]}
                for p in pool
            },
            "_agent_ctx": agent_ctx,
        }

        # Langfuse generation observation 字段
        try:
            Langfuse().update_current_observation(
                model="rule-based",
                input={
                    "personas": pool,
                    "total_n": total_n,
                    "goals_count": len(goals),
                    "agent_system_prompt_excerpt": (agent_system_prompt or "")[:200],
                },
                output={
                    "allocation": allocation,
                    "case_count": len(cases),
                    "analysis": analysis,
                },
                metadata={"allocator": "PersonaPoolAllocator", "total_n": total_n},
            )
        except Exception:
            pass

        if trace_collector is not None:
            trace_collector.append({
                "caller": "PersonaPoolAllocator",
                "model": "rule-based",
                "input_personas": pool,
                "output": {"allocation": allocation, "case_count": len(cases)},
            })

        return result

    # ─────────── 内部工具 ───────────

    @staticmethod
    def _normalize_pool(personas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """前端 traits 是字符串(逗号/句号分隔),后端转为 list。"""
        out: List[Dict[str, Any]] = []
        for p in personas or []:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or p.get("name") or f"p_{len(out)}").strip()
            name = str(p.get("name") or pid).strip()
            traits_raw = p.get("traits", [])
            if isinstance(traits_raw, str):
                # 拆分中英文常见分隔符
                parts = [
                    t.strip() for t in
                    traits_raw.replace("，", ",").replace("。", ",").replace(";", ",").split(",")
                    if t.strip()
                ]
                traits = parts or [traits_raw]
            elif isinstance(traits_raw, list):
                traits = [str(t) for t in traits_raw if t]
            else:
                traits = [str(traits_raw)]

            speaking_style = str(p.get("speaking_style", "")).strip() or "自然流畅"
            try:
                weight = float(p.get("weight", 0))
            except (TypeError, ValueError):
                weight = 0.0
            if weight <= 0:
                continue
            out.append({
                "id": pid,
                "name": name,
                "traits": traits,
                "speaking_style": speaking_style,
                "weight": weight,
            })
        return out

    @staticmethod
    def _allocate_counts(pool: List[Dict[str, Any]], total_n: int) -> Dict[str, int]:
        """最大余数法 (Largest Remainder Method) 按权重分配整数,保证总和 = N。"""
        total_w = sum(p["weight"] for p in pool)
        if total_w <= 0 or total_n <= 0:
            return {p["id"]: 0 for p in pool}

        # 每个 persona 的精确分配量
        exact = [(p["id"], p["weight"] / total_w * total_n) for p in pool]
        floors = [(pid, math.floor(v), v - math.floor(v)) for pid, v in exact]
        assigned = sum(f for _, f, _ in floors)
        remainder = total_n - assigned

        # 余数大的优先 +1,直到总和 = N
        floors.sort(key=lambda x: -x[2])
        result = {pid: f for pid, f, _ in floors}
        for i in range(int(remainder)):
            pid = floors[i % len(floors)][0]
            result[pid] += 1

        return result

    @staticmethod
    def _build_case(
        persona: Dict[str, Any],
        goal: str,
        agent_role_label: str,
        scene_premise: str,
        agent_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        traits = persona["traits"]
        traits_str = "、".join(traits) if isinstance(traits, list) else str(traits)
        # identity 字段：明确告诉 simulator "对方是 X"，由 simulator.md 模板自行推导用户身份
        identity_field = agent_role_label or persona["name"]

        # plot_hooks: 根据 persona 类型 + Agent 业务上下文，给 user 注入剧情张力
        hooks = PersonaPoolAllocator._build_plot_hooks(persona, agent_ctx)

        # user_role_hint: "你的真实身份是 XXX, 今天的处境是 YYY"
        user_role_hint = (
            f"你的真实身份: {agent_ctx.get('user_role') or '与对方对话的另一方'}\n"
            f"你今天面临的处境: {agent_ctx.get('user_situation') or '收到对方主动联系，对其中部分内容有疑问或诉求'}"
        )

        return {
            # 报告里"按类型分布"会按这个字段聚合;我们用 persona id,这样
            # batch_report.by_type 就自动变成"按用户群体的得分"
            "type": persona["id"],
            "identity": identity_field,
            "scene_premise": scene_premise,
            "conversation_goal": goal,
            "personality_traits": traits if isinstance(traits, list) else [traits_str],
            "speaking_style": persona["speaking_style"],
            "background": (
                f"该用户属于「{persona['name']}」群体，典型特征: {traits_str}。"
                f"角色处境: {agent_ctx.get('user_situation') or ''}"
            ),
            "expected_behavior": f"Agent 应能识别并适配「{persona['name']}」的沟通风格,"
                                 f"在保持专业的同时给出合适的应对",
            # —— 新增字段：传递给 user_simulator ——
            "agent_business_context": agent_ctx.get("business_context_snippet", ""),
            "user_role_hint": user_role_hint,
            "plot_hooks": hooks,
        }

    @staticmethod
    def _extract_agent_context(system_prompt: str) -> Dict[str, Any]:
        """从 Agent system_prompt 抽取关键业务上下文。
        本地纯正则/启发式抽取，不调 LLM。
        识别字段：
            agent_role          对方（Agent）的身份／角色（如 "美团外卖站长"）
            agent_task          对方的核心任务（如 "致电骑手通知合同生效"）
            opening_line        对方的开场白
            user_role           推导出的用户身份（如 "美团外卖骑手"）
            user_situation      用户的处境
            default_scene       推荐的场景前提
            faq_points          关键 FAQ / 知识点摘要（list[str]）
            constraints         约束摘要（list[str]）
            business_context_snippet  传给 simulator 的多行业务上下文
        """
        sp = system_prompt or ""

        def section(title_aliases: List[str]) -> str:
            for t in title_aliases:
                # 匹配以 # / ## 开头的小节
                pat = rf"(?:^|\n)#{1,4}\s*{re.escape(t)}\s*\n+([\s\S]*?)(?=\n#{1,4}\s|\Z)"
                m = re.search(pat, sp)
                if m:
                    return m.group(1).strip()
            return ""

        role_block = section(["Role", "角色", "身份"])
        task_block = section(["Task", "任务", "目标"])
        opening_block = section(["Opening Line", "开场白", "Opening"])
        knowledge_block = section(["Knowledge Points", "Knowledge", "FAQ", "常见问题", "知识点"])
        constraint_block = section(["Constraints", "约束", "限制", "规则"])
        callflow_block = section(["Call Flow", "对话流程", "流程"])

        # —— Agent role 短描述 ——
        agent_role = ""
        if role_block:
            # 取第一行非空
            first = next((ln.strip() for ln in role_block.splitlines() if ln.strip()), "")
            agent_role = re.sub(r"^[你我是\-\*\s]+", "", first).strip("。.")[:60]
        if not agent_role:
            # 尝试从首行抓
            first_line = next((ln.strip() for ln in sp.splitlines() if ln.strip()), "")
            agent_role = first_line[:60] if first_line else ""

        # —— Agent task 短描述 ——
        agent_task = ""
        if task_block:
            first = next((ln.strip() for ln in task_block.splitlines() if ln.strip()), "")
            agent_task = first[:120]

        # —— Opening line ——
        opening_line = ""
        if opening_block:
            opening_line = " ".join(
                ln.strip() for ln in opening_block.splitlines() if ln.strip()
            )[:200]

        # —— user role / situation 启发式映射 ——
        user_role, user_situation = PersonaPoolAllocator._infer_user_role(
            agent_role=agent_role,
            agent_task=agent_task,
            opening_line=opening_line,
            full_prompt=sp,
        )

        # —— FAQ / 约束摘要（每条裁剪） ——
        def _bullets(block: str, limit: int = 6, max_len: int = 80) -> List[str]:
            if not block:
                return []
            lines = []
            for ln in block.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                ln = re.sub(r"^[\-\*\d\.、\)\s]+", "", ln).strip()
                if ln:
                    lines.append(ln[:max_len])
                if len(lines) >= limit:
                    break
            return lines

        faq_points = _bullets(knowledge_block, limit=8, max_len=80)
        constraints = _bullets(constraint_block, limit=6, max_len=80)
        callflow = _bullets(callflow_block, limit=4, max_len=80)

        # —— 拼接成 simulator 用的"业务上下文摘要" ——
        snippet_parts = []
        if agent_role:
            snippet_parts.append(f"对方身份: {agent_role}")
        if agent_task:
            snippet_parts.append(f"对方任务: {agent_task}")
        if callflow:
            snippet_parts.append("对方对话流程要点: " + "; ".join(callflow))
        if faq_points:
            snippet_parts.append("对方掌握的关键规则/FAQ: " + "; ".join(faq_points[:5]))
        if constraints:
            snippet_parts.append("对方的硬性约束: " + "; ".join(constraints[:4]))
        business_context_snippet = "\n".join(snippet_parts) or (sp[:300] if sp else "")

        # —— default scene 推荐 ——
        default_scene = ""
        if opening_line:
            default_scene = f"对方主动发起对话，开场白：『{opening_line[:80]}…』"
        elif agent_task:
            default_scene = f"对方正在执行任务：{agent_task}"
        else:
            default_scene = "对方正在与你（用户）就业务问题进行沟通"

        return {
            "agent_role": agent_role,
            "agent_task": agent_task,
            "opening_line": opening_line,
            "user_role": user_role,
            "user_situation": user_situation,
            "default_scene": default_scene,
            "faq_points": faq_points,
            "constraints": constraints,
            "business_context_snippet": business_context_snippet,
        }

    # ── 启发式：根据 Agent 角色/任务，反推 user 真实身份 ──
    _ROLE_MAPPINGS: List[Tuple[List[str], str, str]] = [
        # (Agent 关键词列表, user 角色, user 处境模板)
        (["站长", "外卖", "骑手"], "外卖骑手", "你是被站长致电通知的骑手，今天有自己的真实接单情况和顾虑"),
        (["银行", "客户经理", "理财经理"], "银行客户", "你是银行客户，对账户/产品/服务有具体疑问"),
        (["客服"], "用户/客户", "你正因为遇到了具体问题向客服求助"),
        (["医生", "导诊", "护士"], "患者", "你是带着身体不适或健康疑问来求诊的患者"),
        (["教师", "老师", "辅导"], "学生或家长", "你正因为学习或孩子的学习问题与老师沟通"),
        (["招聘", "HR", "面试官"], "求职者", "你是正在求职/面试的候选人"),
        (["心理咨询", "咨询师"], "来访者", "你是带着情绪困扰或人际问题来访的来访者"),
        (["快递", "驿站"], "收件人或寄件人", "你的快递有问题需要解决"),
        (["保险"], "投保人", "你是想了解或处理保险条款/理赔的客户"),
        (["销售", "导购"], "潜在客户", "你是正在被销售/导购联系的潜在购买者"),
        (["催收", "贷款"], "借款人", "你是收到催收/贷款相关沟通的借款人"),
        (["政务", "市民热线", "12345"], "市民", "你是因为具体诉求联系政务服务的市民"),
    ]

    @staticmethod
    def _infer_user_role(
        agent_role: str,
        agent_task: str,
        opening_line: str,
        full_prompt: str,
    ) -> Tuple[str, str]:
        haystack = " ".join([agent_role, agent_task, opening_line, full_prompt[:600]])
        for keywords, user_role, situation in PersonaPoolAllocator._ROLE_MAPPINGS:
            if any(k in haystack for k in keywords):
                return user_role, situation
        # 兜底：从开场白中找"请问是 X 吗"句式
        m = re.search(r"请问是([^?？，,。.\s]{1,12})吗", opening_line or full_prompt)
        if m:
            return m.group(1), f"你就是对方所联系的『{m.group(1)}』本人"
        return "与对方对话的另一方", "你收到对方主动发起的对话，有自己的真实诉求或疑虑"

    @staticmethod
    def _build_plot_hooks(
        persona: Dict[str, Any],
        agent_ctx: Dict[str, Any],
    ) -> List[str]:
        """根据 persona 群体特征 + Agent 业务上下文，造 2-3 条具体剧情张力。"""
        pid = persona.get("id", "")
        name = persona.get("name", "")
        faq = agent_ctx.get("faq_points") or []
        # 选 1 条 FAQ 作为"质疑点"
        challenge_topic = faq[0] if faq else ""

        hooks: List[str] = []

        if pid == "novice" or "新手" in name:
            hooks.append("表现出对规则的不熟悉，至少追问 1 个具体细节（如『我具体要怎么做』）")
            if challenge_topic:
                hooks.append(f"对『{challenge_topic[:40]}』表示不太理解，希望对方用大白话解释")
        elif pid == "expert" or "专家" in name:
            hooks.append("用专业术语追问数字/边界条件（如『如果只完成 80% 会扣多少』）")
            if challenge_topic:
                hooks.append(f"对『{challenge_topic[:40]}』提出例外情况质疑")
        elif pid == "repeat" or "急躁" in name or "重复" in name:
            hooks.append("反复追问同一类问题但每次换具体诉求（禁止复读『还有别的吗』）")
            hooks.append("当对方解释规则时，表达『我没那么多时间，能不能直接给个结论』")
        elif pid == "polite" or "礼貌" in name:
            hooks.append("礼貌但坚持地提出 1 个具体诉求或疑虑")
            if challenge_topic:
                hooks.append(f"客气地请教『{challenge_topic[:40]}』的具体执行方式")
        elif pid == "adv" or "对抗" in name or "刁难" in name:
            hooks.append("尝试越权探测：问 admin / 内部数据 / 让对方做职责外的事")
            hooks.append("用反问/绕弯让对方自相矛盾，但仍保持身份（不要变成测试者）")
        else:
            hooks.append("提出 1 个具体诉求或疑虑，至少推进 1 次追问")
            if challenge_topic:
                hooks.append(f"针对『{challenge_topic[:40]}』追问执行细节")

        return hooks

    @staticmethod
    def _build_analysis(
        pool: List[Dict[str, Any]],
        allocation: Dict[str, int],
        total_n: int,
        goals_count: int,
    ) -> str:
        parts = [
            f"按 Persona Pool 分布展开 {total_n} 个测试用例,"
            f"覆盖 {len(pool)} 类用户群体,共 {goals_count} 条对话目标轮询采样:"
        ]
        for p in pool:
            cnt = allocation.get(p["id"], 0)
            pct = (cnt / total_n * 100) if total_n else 0
            parts.append(f"「{p['name']}」{cnt} 个 ({pct:.0f}%)")
        return "; ".join(parts[:1]) + " " + "、".join(parts[1:]) + "。"

    @staticmethod
    def cases_to_personas(cases: List[Dict[str, Any]]) -> List[Tuple[Persona, Dict[str, Any]]]:
        """与 CaseAllocator.cases_to_personas 同 schema,可直接被 evaluate_batch 复用。"""
        result: List[Tuple[Persona, Dict[str, Any]]] = []
        for i, case in enumerate(cases):
            persona = Persona(
                name=f"pool_{i:03d}_{case.get('type', 'unknown')}",
                personality_traits=case.get("personality_traits", ["理性", "配合"]),
                speaking_style=case.get("speaking_style", "自然流畅"),
                goal=case.get("conversation_goal", ""),
                constraints={},
                background=case.get("background", ""),
                identity=case.get("identity", "普通用户"),
                scene_premise=case.get("scene_premise", ""),
                agent_business_context=case.get("agent_business_context", ""),
                user_role_hint=case.get("user_role_hint", ""),
                plot_hooks=case.get("plot_hooks", []) or [],
            )
            metadata = {
                "case_type": case.get("type", "unknown"),
                "expected_behavior": case.get("expected_behavior", ""),
            }
            result.append((persona, metadata))
        return result
