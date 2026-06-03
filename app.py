#!/usr/bin/env python3
"""
图灵机-Agent多轮对话评估平台 - Web应用
"""
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
from datetime import datetime
import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars

# 导入我们的核心模块
from agent_container import AgentContainer, AgentConfig
from agent_container.agent_container import scan_prompt_placeholders
from simulator import CaseGenerator, UserSimulator, Persona
from simulator.prompt_refiner import PromptRefiner
from simulator.multi_persona_runner import create_persona_variants, aggregate_multi_results
from simulator.scenario_generator import ScenarioGenerator
from evaluator import LLMJudge
from evaluator.case_allocator import CaseAllocator
from evaluator.persona_pool_allocator import PersonaPoolAllocator
from usage_logger import log_event
from output import ReportGenerator
from output.batch_report_generator import BatchReportGenerator
from prompt_loader import load_prompt, load_md_prompt
from langfuse import observe, Langfuse
from langfuse_tracer import flush_langfuse
from config import SIMULATOR_MODEL, JUDGE_MODEL, DEFAULT_MODEL
from evaluator.progress_logger import ProgressLogger

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

@app.route('/logoLang.svg')
def logo_lang_svg():
    return send_from_directory('.', 'logoLang.svg', mimetype='image/svg+xml')

# 确保输出目录存在
OUTPUT_DIR = Path("output/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量存储当前任务状态
current_tasks = {}
_current_progress_logger: ProgressLogger = None

# ============== 死循环检测 ==============
_eval_cfg = load_prompt("evaluation")
LOOP_SIMILARITY_THRESHOLD = _eval_cfg["loop_detection"]["similarity_threshold"]
LOOP_CONSECUTIVE_LIMIT = _eval_cfg["loop_detection"]["consecutive_limit"]
REFUSAL_PHRASES = _eval_cfg["refusal_phrases"]
REFUSAL_THRESHOLD = _eval_cfg["refusal_threshold"]
OPENING_MESSAGE = _eval_cfg["opening_message"]
END_SIGNAL = _eval_cfg["end_signal"]


def _normalize(text: str) -> str:
    return re.sub(r'\s+', '', text or '').strip()


def _similarity(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def detect_loop(agent_outputs: List[str]) -> Tuple[bool, int]:
    """检测 agent 输出序列中是否出现死循环。

    Returns:
        (is_loop, consecutive_count): 是否死循环 + 当前连续重复的次数
    """
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


# ============== 约束违反检测 ==============
_CHAR_LIMIT_PATTERNS = [
    r"控制在.{0,8}?(?P<n>\d{2,3}).{0,4}?字",
    r"不.{0,4}?超过.{0,8}?(?P<n>\d{2,3}).{0,4}?字",
    r"(?P<n>\d{2,3}).{0,4}?字以?内",
    r"约.{0,4}?(?P<n>\d{2,3}).{0,4}?字",
]


def extract_char_limit(system_prompt: str) -> int:
    """从 system_prompt 中提取"每次回复 N 字以内"约束。返回 0 表示未约束。"""
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
    """扫描 Agent 输出，识别 system_prompt 里硬约束的违反情况。
    返回 rule_flags 列表（每条一个违规描述）。
    """
    flags: List[str] = []

    # 1) 字数限制
    limit = extract_char_limit(system_prompt)
    if limit > 0 and agent_outputs:
        # 允许 20% 容差
        threshold = int(limit * 1.2)
        violations = [(i, len(o)) for i, o in enumerate(agent_outputs) if o and len(o) > threshold]
        if violations:
            sample = "; ".join(f"第{i}轮 {ln}字" for i, ln in violations[:3])
            flags.append(
                f"违反『{limit}字以内』约束: {len(violations)}/{len(agent_outputs)} 轮超长（{sample}）"
            )

    # 2) 输出之间高度相似（非连续，全局两两）—— 复读但非死循环
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


class TuringEvaluationWeb:
    """Web版评估平台 - 使用真实API"""

    def __init__(self, api_key=None, api_base_url=None):
        self.agent_container = AgentContainer()
        self.case_generator = CaseGenerator()
        self.user_simulator = UserSimulator(
            api_key=api_key,
            base_url=api_base_url,
            model=SIMULATOR_MODEL
        )
        self.llm_judge = LLMJudge(
            api_key=api_key,
            base_url=api_base_url,
            model=JUDGE_MODEL
        )
        self.report_generator = ReportGenerator()

    @observe(name="evaluation")
    def run_evaluation(
        self,
        agent_config: AgentConfig,
        test_persona: Persona,
        conversation_goal: str,
        max_turns: int = 5,
        session_id: str = None,
        user_id: str = None
    ) -> Dict:
        if session_id or user_id:
            Langfuse().update_current_trace(session_id=session_id, user_id=user_id)

        self.agent_container.register_agent(agent_config)
        self.user_simulator.register_persona(test_persona)

        self.agent_container.create_prefix_cache(agent_config.name)
        self.user_simulator.create_prefix_cache(test_persona.name)

        # —— 系统级硬规则告警容器 ——
        rule_flags: List[str] = []

        # ① Prompt 占位符扫描
        placeholder_findings = scan_prompt_placeholders(agent_config.system_prompt)
        if placeholder_findings:
            samples = ", ".join(
                f"'{txt}'({desc})" for txt, desc in placeholder_findings[:3]
            )
            rule_flags.append(
                f"Agent system prompt 中存在未填充变量/占位符: {samples}"
                + ("，..." if len(placeholder_findings) > 3 else "")
            )

        test_case = self.case_generator.generate_llm_simulation_case(
            conversation_goal=conversation_goal,
            persona_constraints=test_persona.constraints
        )

        conversation_history = []
        current_message = None
        llm_traces = []
        agent_outputs: List[str] = []
        loop_terminated = False

        for turn in range(max_turns):
            if turn == 0:
                agent_response = self.agent_container.generate_response(
                    agent_name=agent_config.name,
                    user_message=OPENING_MESSAGE,
                    conversation_history=conversation_history,
                    trace_collector=llm_traces
                )
                conversation_history.append({"role": "assistant", "content": agent_response})
                agent_outputs.append(agent_response)
            else:
                if not current_message:
                    current_message = OPENING_MESSAGE
                user_response = self.user_simulator.generate_response(
                    persona_name=test_persona.name,
                    agent_message=current_message,
                    conversation_history=conversation_history,
                    trace_collector=llm_traces
                )
                if END_SIGNAL in user_response:
                    user_response = user_response.replace(END_SIGNAL, "").strip()
                    if user_response:
                        conversation_history.append({"role": "user", "content": user_response})
                    rule_flags.append(f"用户在第{turn}轮主动结束对话")
                    break
                conversation_history.append({"role": "user", "content": user_response})

                agent_response = self.agent_container.generate_response(
                    agent_name=agent_config.name,
                    user_message=user_response,
                    conversation_history=conversation_history[:-1],
                    trace_collector=llm_traces
                )
                conversation_history.append({"role": "assistant", "content": agent_response})
                agent_outputs.append(agent_response)

            current_message = agent_response

            # ② 死循环检测：连续 N 轮 Agent 输出高度相似 → 提前终止
            is_loop, consecutive = detect_loop(agent_outputs)
            if is_loop:
                rule_flags.append(
                    f"检测到 Agent 死循环（连续{consecutive}轮输出相似度≥{LOOP_SIMILARITY_THRESHOLD}），"
                    f"已在第{turn}轮提前终止评估"
                )
                loop_terminated = True
                break

        # ③ 过度拒答启发式：扫 Agent 全部输出
        refusal_hits = sum(
            1 for o in agent_outputs
            if any(p in (o or "") for p in REFUSAL_PHRASES)
        )
        if refusal_hits >= REFUSAL_THRESHOLD:
            rule_flags.append(
                f"检测到过度拒答/推诿话术 {refusal_hits} 次（出现『稍后答复/我去问同事』等表达），"
                f"未给出实质回答"
            )

        # ④ 约束违反检测：字数限制、重复回复
        constraint_flags = detect_constraint_violations(
            agent_outputs, agent_config.system_prompt
        )
        rule_flags.extend(constraint_flags)

        evaluation_result = self.llm_judge.evaluate(
            case_id=test_case.case_id,
            agent_name=agent_config.name,
            conversation_history=conversation_history,
            agent_system_prompt=agent_config.system_prompt,
            conversation_goal=conversation_goal,
            trace_collector=llm_traces,
            rule_flags=rule_flags
        )

        lf = Langfuse()
        lf.update_current_trace(
            output={
                "overall_score": evaluation_result.overall_score,
                "dimension_scores": evaluation_result.dimension_scores,
                "rule_flags": rule_flags
            }
        )

        for dim, score in evaluation_result.dimension_scores.items():
            conf = evaluation_result.confidence.get(dim, "")
            ev = evaluation_result.evidence_turns.get(dim, [])
            comment = f"confidence={conf}; evidence_turns={ev}"
            lf.score_current_trace(name=dim, value=score, comment=comment)
        lf.score_current_trace(
            name="overall",
            value=evaluation_result.overall_score,
            comment=f"rule_flags={rule_flags}"
        )

        self.agent_container.clear_prefix_cache(agent_config.name)
        self.user_simulator.clear_prefix_cache()

        return {
            "test_case": test_case,
            "conversation_history": conversation_history,
            "evaluation_result": evaluation_result,
            "llm_traces": llm_traces,
            "rule_flags": rule_flags,
            "loop_terminated": loop_terminated
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/evaluate', methods=['POST'])
@observe(name="evaluate_api")
def evaluate():
    try:
        data = request.json

        session_id = data.get('session_id', '') or None
        user_id = data.get('user_id', '') or None
        if session_id or user_id:
            Langfuse().update_current_trace(session_id=session_id, user_id=user_id)

        api_key = os.getenv('OPENAI_API_KEY')
        api_base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return jsonify({'success': False, 'error': '未配置 OPENAI_API_KEY 环境变量'}), 400

        platform = TuringEvaluationWeb(
            api_key=api_key,
            api_base_url=api_base_url
        )

        agent_config = AgentConfig(
            name=data.get('agent_name', 'test_agent'),
            system_prompt=data.get('system_prompt', ''),
            model=data.get('model', DEFAULT_MODEL),
            api_key=api_key,
            base_url=api_base_url,
            temperature=float(data.get('temperature', 0.7))
        )

        # 3. 提示词润色：根据 Agent 设定自动补全 Simulator 角色信息
        refiner = PromptRefiner(api_key=api_key, base_url=api_base_url)
        refiner_traces = []
        refined = refiner.refine(
            agent_system_prompt=agent_config.system_prompt,
            identity=data.get('identity', ''),
            scene_premise=data.get('scene_premise', ''),
            conversation_goal=data.get('conversation_goal', ''),
            trace_collector=refiner_traces
        )

        # 4. 创建Persona（用润色结果填充，用户手动输入优先）
        persona = Persona(
            name=data.get('persona_name', 'test_user'),
            personality_traits=data.get('personality_traits') or refined['personality_traits'],
            speaking_style=data.get('speaking_style') or refined['speaking_style'],
            goal=data.get('conversation_goal') or refined['refined_goal'],
            constraints=data.get('constraints') or refined['constraints'],
            background=data.get('background') or refined['background'],
            identity=data.get('identity') or refined['inferred_identity'],
            scene_premise=data.get('scene_premise', '')
        )

        # 5. 运行评估
        max_turns = int(data.get('max_turns', 5))
        result = platform.run_evaluation(
            agent_config,
            persona,
            data.get('conversation_goal', ''),
            max_turns,
            session_id=session_id,
            user_id=user_id
        )

        # 6. 生成报告
        html_path = platform.report_generator.generate_html_report(
            [result['evaluation_result']],
            f"{agent_config.name}_评估报告"
        )
        json_path = platform.report_generator.generate_json_report(
            [result['evaluation_result']],
            eval_config={
                'agent_name': agent_config.name,
                'system_prompt': agent_config.system_prompt,
                'model': agent_config.model,
                'temperature': agent_config.temperature,
                'max_tokens': agent_config.max_tokens,
                'identity': data.get('identity', ''),
                'scene_premise': data.get('scene_premise', ''),
                'conversation_goal': data.get('conversation_goal', ''),
                'max_turns': max_turns,
                'persona_name': persona.name,
                'personality_traits': persona.personality_traits,
                'speaking_style': persona.speaking_style,
            }
        )

        # 7. 返回结果
        eval_res = result['evaluation_result']
        return jsonify({
            'success': True,
            'result': {
                'case_id': eval_res.case_id,
                'agent_name': eval_res.agent_name,
                'overall_score': eval_res.overall_score,
                'dimension_scores': eval_res.dimension_scores,
                'evidence_turns': eval_res.evidence_turns,
                'confidence': eval_res.confidence,
                'qualitative_analysis': eval_res.qualitative_analysis,
                'bad_cases': eval_res.bad_cases,
                'rule_flags': eval_res.rule_flags,
                'loop_terminated': result.get('loop_terminated', False),
                'conversation_history': eval_res.conversation_history,
                'html_report': str(html_path.name),
                'json_report': str(json_path.name),
                'llm_traces': refiner_traces + result['llm_traces']
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        flush_langfuse()


@app.route('/api/reports/<filename>')
def get_report(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route('/api/history')
def get_history():
    reports = []
    if OUTPUT_DIR.exists():
        for file in sorted(OUTPUT_DIR.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 兼容两种 schema:
                #  - 批量报告: 顶层 agent_name + stats: {avg_score, pass_rate, ...}
                #  - 单测报告: results/evaluations 列表 + overall_stats
                stats = data.get('stats') or data.get('overall_stats') or {}
                results = data.get('results') or data.get('evaluations') or []
                agent_name = (
                    data.get('agent_name')
                    or stats.get('agent_name')
                    or (results[0].get('agent_name') if results and isinstance(results[0], dict) else None)
                    or '未知'
                )
                # 为前端统一暴露 avg_score(批量) 同时保留 avg_overall(老前端兼容)
                if 'avg_score' not in stats and 'avg_overall' in stats:
                    stats = {**stats, 'avg_score': stats.get('avg_overall')}
                if 'avg_overall' not in stats and 'avg_score' in stats:
                    stats = {**stats, 'avg_overall': stats.get('avg_score')}
                reports.append({
                    'filename': file.name,
                    'timestamp': data.get('timestamp', ''),
                    'stats': stats,
                    'agent_name': agent_name,
                    'total_n': data.get('total_n', stats.get('total_count', len(results))),
                })
            except Exception:
                continue
    return jsonify(reports)


@app.route('/api/evaluate_multi', methods=['POST'])
@observe(name="evaluate_multi_api")
def evaluate_multi():
    try:
        data = request.json

        session_id = data.get('session_id', '') or None
        user_id = data.get('user_id', '') or None
        if session_id or user_id:
            Langfuse().update_current_trace(session_id=session_id, user_id=user_id)

        api_key = os.getenv('OPENAI_API_KEY')
        api_base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return jsonify({'success': False, 'error': '未配置 OPENAI_API_KEY 环境变量'}), 400

        refiner = PromptRefiner(api_key=api_key, base_url=api_base_url)
        refiner_traces = []
        refined = refiner.refine(
            agent_system_prompt=data.get('system_prompt', ''),
            identity=data.get('identity', ''),
            scene_premise=data.get('scene_premise', ''),
            conversation_goal=data.get('conversation_goal', ''),
            trace_collector=refiner_traces
        )

        base_persona = Persona(
            name=data.get('persona_name', 'test_user'),
            personality_traits=data.get('personality_traits') or refined['personality_traits'],
            speaking_style=data.get('speaking_style') or refined['speaking_style'],
            goal=data.get('conversation_goal') or refined['refined_goal'],
            constraints=data.get('constraints') or refined['constraints'],
            background=data.get('background') or refined['background'],
            identity=data.get('identity') or refined['inferred_identity'],
            scene_premise=data.get('scene_premise', '')
        )

        variants = create_persona_variants(base_persona)
        max_turns = int(data.get('max_turns', 5))
        multi_results = []

        for variant_name, variant_persona in variants.items():
            platform = TuringEvaluationWeb(
                api_key=api_key,
                api_base_url=api_base_url
            )
            agent_config = AgentConfig(
                name=data.get('agent_name', 'test_agent'),
                system_prompt=data.get('system_prompt', ''),
                model=data.get('model', DEFAULT_MODEL),
                api_key=api_key,
                base_url=api_base_url,
                temperature=float(data.get('temperature', 0.7))
            )

            result = platform.run_evaluation(
                agent_config,
                variant_persona,
                data.get('conversation_goal', ''),
                max_turns,
                session_id=session_id,
                user_id=user_id
            )

            eval_res = result['evaluation_result']
            multi_results.append({
                "variant": variant_name,
                "overall_score": eval_res.overall_score,
                "dimension_scores": eval_res.dimension_scores,
                "evidence_turns": eval_res.evidence_turns,
                "confidence": eval_res.confidence,
                "conversation_history": eval_res.conversation_history,
                "rule_flags": eval_res.rule_flags,
                "qualitative_analysis": eval_res.qualitative_analysis,
            })

        aggregated = aggregate_multi_results(multi_results)

        return jsonify({
            'success': True,
            'result': aggregated,
            'llm_traces': refiner_traces
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        flush_langfuse()


@app.route('/api/preview_allocation', methods=['POST'])
@observe(name="preview_allocation_api")
def preview_allocation():
    """预览配比方案。

    支持两种 allocation_mode:
    - "auto" (默认):走 CaseAllocator,LLM 按 7 类风险类型智能配比
    - "persona_pool":按前端 V2 配置的"用户群体分布"权重展开,纯本地计算
    """
    try:
        data = request.json

        session_id = data.get('session_id', '') or None
        user_id = data.get('user_id', '') or None
        if session_id or user_id:
            Langfuse().update_current_trace(session_id=session_id, user_id=user_id)

        api_key = os.getenv('OPENAI_API_KEY')
        api_base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return jsonify({'success': False, 'error': '未配置 OPENAI_API_KEY 环境变量'}), 400

        total_n = int(data.get('total_n', 10))
        preference = data.get('preference', '均匀覆盖')
        system_prompt = data.get('system_prompt', '')
        allocation_mode = (data.get('allocation_mode') or 'auto').strip().lower()

        # 5 维度生成多样性要求(可选,前端 V2 用户群体分布配置注入)
        preview_gen_reqs = data.get('generation_requirements') or {}
        if not isinstance(preview_gen_reqs, dict):
            preview_gen_reqs = {}

        if not system_prompt and allocation_mode == 'auto':
            return jsonify({'success': False, 'error': '请提供 Agent 的系统提示词'}), 400

        trace_collector: List[Dict] = []

        if allocation_mode == 'persona_pool':
            personas_payload = data.get('personas') or []
            goals = data.get('goals') or []
            if isinstance(goals, str):
                goals = [g.strip() for g in goals.split('\n') if g.strip()]
            allocator = PersonaPoolAllocator()
            result = allocator.allocate(
                personas=personas_payload,
                total_n=total_n,
                goals=goals,
                identity=data.get('identity', ''),
                scene_premise=data.get('scene_premise', ''),
                agent_system_prompt=system_prompt,
                trace_collector=trace_collector,
            )
        else:
            allocator = CaseAllocator(api_key=api_key, base_url=api_base_url)
            result = allocator.allocate(
                agent_system_prompt=system_prompt,
                total_n=total_n,
                preference=preference,
                trace_collector=trace_collector,
                generation_requirements=preview_gen_reqs,
            )

        return jsonify({
            'success': True,
            'result': {
                'allocation_mode': allocation_mode,
                'analysis': result.get('analysis', ''),
                'allocation': result.get('allocation', {}),
                'cases': result.get('cases', []),
                'pool_meta': result.get('_pool_meta', {}),
                'total_n': total_n,
                'preference': preference,
            },
            'llm_traces': trace_collector
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        flush_langfuse()


@app.route('/api/evaluate_batch', methods=['POST'])
@observe(name="evaluate_batch_api")
def evaluate_batch():
    """批量测试。

    支持两种 allocation_mode:
    - "auto" (默认):CaseAllocator 按 7 类型 + preference 智能分配
    - "persona_pool":PersonaPoolAllocator 按 V2 用户群体分布权重展开
    并发执行后聚合成 batch report,4 大批次指标 + 按类型/群体分布。
    """
    try:
        data = request.json

        session_id = data.get('session_id', '') or None
        user_id = data.get('user_id', '') or None
        if session_id or user_id:
            Langfuse().update_current_trace(session_id=session_id, user_id=user_id)

        api_key = os.getenv('OPENAI_API_KEY')
        api_base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return jsonify({'success': False, 'error': '未配置 OPENAI_API_KEY 环境变量'}), 400

        total_n = int(data.get('total_n', 10))
        preference = data.get('preference', '均匀覆盖')
        max_turns = int(data.get('max_turns', 5))
        system_prompt = data.get('system_prompt', '')
        agent_name = data.get('agent_name', 'test_agent')
        allocation_mode = (data.get('allocation_mode') or 'auto').strip().lower()
        # 上下文变量(用于把 system_prompt 中的 ${var}/{var}/<var>/[VAR]/{{var}} 渲染为真实值)
        context_variables = data.get('context_variables') or {}
        if not isinstance(context_variables, dict):
            context_variables = {}

        # 5 维度生成多样性要求(由前端 V2 用户群体分布配置注入)
        generation_requirements = data.get('generation_requirements') or {}
        if not isinstance(generation_requirements, dict):
            generation_requirements = {}

        if not system_prompt:
            return jsonify({'success': False, 'error': '请提供 Agent 的系统提示词'}), 400

        # 1. 智能配比 / Persona Pool 展开
        alloc_traces: List[Dict] = []
        pool_meta: Dict[str, Any] = {}

        if allocation_mode == 'persona_pool':
            personas_payload = data.get('personas') or []
            goals = data.get('goals') or []
            if isinstance(goals, str):
                goals = [g.strip() for g in goals.split('\n') if g.strip()]
            pool_allocator = PersonaPoolAllocator()
            alloc_result = pool_allocator.allocate(
                personas=personas_payload,
                total_n=total_n,
                goals=goals,
                identity=data.get('identity', ''),
                scene_premise=data.get('scene_premise', ''),
                agent_system_prompt=system_prompt,
                trace_collector=alloc_traces,
            )
            persona_list = pool_allocator.cases_to_personas(alloc_result.get('cases', []))
            pool_meta = alloc_result.get('_pool_meta', {})
        else:
            allocator = CaseAllocator(api_key=api_key, base_url=api_base_url)
            alloc_result = allocator.allocate(
                agent_system_prompt=system_prompt,
                total_n=total_n,
                preference=preference,
                trace_collector=alloc_traces,
                generation_requirements=generation_requirements,
            )
            persona_list = allocator.cases_to_personas(alloc_result.get('cases', []))

        allocation = alloc_result.get('allocation', {})
        analysis = alloc_result.get('analysis', '')
        cases = alloc_result.get('cases', [])

        if not cases:
            log_event(
                "app.batch_evaluate",
                "ALLOC_RESULT_EMPTY",
                allocation_mode=allocation_mode,
                total_n=total_n,
                allocation=json.dumps(allocation, ensure_ascii=False),
                persona_list_len=len(persona_list),
            )
            return jsonify({'success': False, 'error': '配比器未能生成测试用例'}), 400

        # 2. 并发执行评估
        def _run_single(index, persona, metadata):
            """单个评估任务（线程安全）。"""
            platform = TuringEvaluationWeb(
                api_key=api_key,
                api_base_url=api_base_url
            )
            agent_config = AgentConfig(
                name=agent_name,
                system_prompt=system_prompt,
                model=data.get('model', DEFAULT_MODEL),
                api_key=api_key,
                base_url=api_base_url,
                temperature=float(data.get('temperature', 0.7)),
                context_variables=context_variables,
            )

            result = platform.run_evaluation(
                agent_config,
                persona,
                persona.goal,
                max_turns,
                session_id=session_id,
                user_id=user_id
            )

            eval_res = result['evaluation_result']
            return {
                "index": index,
                "agent_name": agent_name,
                "case_type": metadata.get("case_type", "unknown"),
                "expected_behavior": metadata.get("expected_behavior", ""),
                "scenario": {
                    "identity": persona.identity,
                    "scene_premise": persona.scene_premise,
                    "conversation_goal": persona.goal,
                    "personality_traits": persona.personality_traits,
                    "speaking_style": persona.speaking_style,
                    "background": persona.background,
                    # —— 新增：业务上下文 + 剧情钩子，便于导出与排查 ——
                    "agent_business_context": persona.agent_business_context or "",
                    "user_role_hint": persona.user_role_hint or "",
                    "plot_hooks": persona.plot_hooks or [],
                },
                "effective_system_prompt": agent_config.effective_system_prompt,
                "overall_score": eval_res.overall_score,
                "dimension_scores": eval_res.dimension_scores,
                "evidence_turns": eval_res.evidence_turns,
                "confidence": eval_res.confidence,
                "rule_flags": eval_res.rule_flags,
                "conversation_history": eval_res.conversation_history,
                "qualitative_analysis": eval_res.qualitative_analysis,
                "bad_cases": eval_res.bad_cases,
                "loop_terminated": result.get('loop_terminated', False),
            }

        batch_results = []
        max_workers = min(4, total_n)

        # 关键：捕获父 trace 的 contextvars,在子线程中通过 ctx.run 还原,
        # 这样 _run_single 内部的 @observe 能挂到父 trace 下而不是变成孤儿 trace。
        # 注意：同一个 Context 对象不能被并发/重入 run() — 必须为每个子任务克隆一份独立副本。
        parent_ctx = contextvars.copy_context()

        def _run_in_parent_ctx(idx, persona, metadata):
            # 每次 run 都用 parent_ctx 的独立副本，避免 "Context is already entered"
            ctx = parent_ctx.copy()
            return ctx.run(_run_single, idx, persona, metadata)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_in_parent_ctx, i, persona, metadata): i
                for i, (persona, metadata) in enumerate(persona_list)
            }
            for future in as_completed(futures):
                try:
                    res = future.result()
                    batch_results.append(res)
                except Exception as exc:
                    batch_results.append({
                        "index": futures[future],
                        "case_type": "error",
                        "overall_score": 0,
                        "dimension_scores": {},
                        "rule_flags": [f"执行异常: {str(exc)}"],
                        "error": str(exc)
                    })

        # 按 index 排序
        batch_results.sort(key=lambda x: x.get("index", 0))

        # 3. 生成聚合报告
        report_gen = BatchReportGenerator()
        report_result = report_gen.generate(
            batch_results=batch_results,
            allocation=allocation,
            analysis=analysis,
            agent_name=agent_name,
            total_n=total_n,
            allocation_mode=allocation_mode,
            pool_meta=pool_meta,
        )

        # 4. 在父 trace 上写 batch-level metadata + output + scores
        stats = report_result["stats"]
        lf = Langfuse()
        lf.update_current_trace(
            metadata={
                "allocation_mode": allocation_mode,
                "allocation": allocation,
                "preference": preference if allocation_mode == 'auto' else None,
                "pool_meta": pool_meta if allocation_mode == 'persona_pool' else None,
                "total_n": total_n,
                "agent_name": agent_name,
                "analysis": analysis,
            },
            output={
                "pass_rate": stats.get("pass_rate", 0),
                "stability": stats.get("stability", 0),
                "worst_case": stats.get("worst_case", 0),
                "rule_flag_hit_rate": stats.get("rule_flag_hit_rate", 0),
                "avg_score": stats.get("avg_score", 0),
                "total_count": stats.get("total_count", 0),
                "html_report": report_result["html_path"],
                "json_report": report_result["json_path"],
            }
        )

        # 4 个 batch-level 核心指标
        lf.score_current_trace(
            name="batch_pass_rate",
            value=stats.get("pass_rate", 0),
            comment=f"≥7 分通过率,共 {stats.get('total_count', 0)} 个 case"
        )
        lf.score_current_trace(
            name="batch_stability",
            value=stats.get("stability", 0),
            comment="1 - 变异系数 (CV)"
        )
        lf.score_current_trace(
            name="batch_worst_case",
            value=stats.get("worst_case", 0),
            comment="所有 case 中的最低分"
        )
        lf.score_current_trace(
            name="batch_rule_flag_hit_rate",
            value=stats.get("rule_flag_hit_rate", 0),
            comment="触发硬规则告警的 case 占比"
        )

        # 按 case_type / persona 群体 打分
        for case_type, type_data in stats.get("by_type", {}).items():
            lf.score_current_trace(
                name=f"by_type__{case_type}__avg",
                value=type_data.get("avg", 0),
                comment=f"{case_type} 平均分,共 {type_data.get('count', 0)} 个 case"
            )

        return jsonify({
            'success': True,
            'result': {
                "allocation_mode": allocation_mode,
                "total_n": total_n,
                "allocation": allocation,
                "pool_meta": pool_meta,
                "analysis": analysis,
                "scenarios": batch_results,
                "stats": report_result["stats"],
                "html_report": report_result["html_path"],
                "json_report": report_result["json_path"],
            },
            'llm_traces': alloc_traces
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        flush_langfuse()


@app.route('/api/report/<filename>')
def get_report_detail(filename):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return jsonify({'success': False, 'error': '报告不存在'}), 404
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_report(filename):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return jsonify({'success': False, 'error': '报告不存在'}), 404
    try:
        filepath.unlink()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ================================================================
#  V2 路由 — Subagent 并发架构（不依赖 langfuse）
# ================================================================
from evaluator.resource_pool import ResourcePool
from evaluator.eval_subagent import EvalSubagent
from evaluator.batch_orchestrator import BatchOrchestrator
from evaluator.run_logger import RunLogger, list_run_logs, get_run_log


@app.route('/api/v2/evaluate', methods=['POST'])
def evaluate_v2():
    """V2 单测路由：走 EvalSubagent，无 langfuse。"""
    try:
        data = request.json

        api_key = os.getenv('OPENAI_API_KEY')
        api_base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return jsonify({'success': False, 'error': '未配置 OPENAI_API_KEY 环境变量'}), 400

        pool = ResourcePool(api_key=api_key, base_url=api_base_url)

        agent_config = AgentConfig(
            name=data.get('agent_name', 'test_agent'),
            system_prompt=data.get('system_prompt', ''),
            model=data.get('model', DEFAULT_MODEL),
            api_key=api_key,
            base_url=api_base_url,
            temperature=float(data.get('temperature', 0.7)),
            context_variables=data.get('context_variables') or {},
        )

        # 提示词润色
        refiner = PromptRefiner(api_key=api_key, base_url=api_base_url)
        refined = refiner.refine(
            agent_system_prompt=agent_config.system_prompt,
            identity=data.get('identity', ''),
            scene_premise=data.get('scene_premise', ''),
            conversation_goal=data.get('conversation_goal', ''),
            trace_collector=None,
        )

        persona = Persona(
            name=data.get('persona_name', 'test_user'),
            personality_traits=data.get('personality_traits') or refined['personality_traits'],
            speaking_style=data.get('speaking_style') or refined['speaking_style'],
            goal=data.get('conversation_goal') or refined['refined_goal'],
            constraints=data.get('constraints') or refined['constraints'],
            background=data.get('background') or refined['background'],
            identity=data.get('identity') or refined['inferred_identity'],
            scene_premise=data.get('scene_premise', ''),
        )

        max_turns = int(data.get('max_turns', 5))
        subagent = EvalSubagent(pool)
        result = subagent.run(agent_config, persona, data.get('conversation_goal', ''), max_turns)

        eval_res = result['evaluation_result']
        return jsonify({
            'success': True,
            'result': {
                'case_id': eval_res.case_id,
                'agent_name': eval_res.agent_name,
                'overall_score': eval_res.overall_score,
                'dimension_scores': eval_res.dimension_scores,
                'evidence_turns': eval_res.evidence_turns,
                'confidence': eval_res.confidence,
                'qualitative_analysis': eval_res.qualitative_analysis,
                'bad_cases': eval_res.bad_cases,
                'rule_flags': eval_res.rule_flags,
                'loop_terminated': result.get('loop_terminated', False),
                'conversation_history': eval_res.conversation_history,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v2/evaluate_batch', methods=['POST'])
def evaluate_batch_v2():
    """V2 批量路由：Subagent + asyncio 并发编排，无 langfuse。

    并发度可通过 env EVAL_MAX_CONCURRENCY 调整（默认 16）。
    """
    try:
        data = request.json

        api_key = os.getenv('OPENAI_API_KEY')
        api_base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return jsonify({'success': False, 'error': '未配置 OPENAI_API_KEY 环境变量'}), 400

        total_n = int(data.get('total_n', 10))
        preference = data.get('preference', '均匀覆盖')
        max_turns = int(data.get('max_turns', 5))
        system_prompt = data.get('system_prompt', '')
        agent_name = data.get('agent_name', 'test_agent')
        allocation_mode = (data.get('allocation_mode') or 'auto').strip().lower()
        context_variables = data.get('context_variables') or {}
        if not isinstance(context_variables, dict):
            context_variables = {}
        generation_requirements = data.get('generation_requirements') or {}
        if not isinstance(generation_requirements, dict):
            generation_requirements = {}

        if not system_prompt:
            return jsonify({'success': False, 'error': '请提供 Agent 的系统提示词'}), 400

        # 1. 配比 / Persona Pool 展开 / Ref 扩写
        pool_meta: Dict[str, Any] = {}
        case_mode = (data.get('case_mode') or 'llm').strip().lower()

        if case_mode == 'ref':
            # ─── PR1: 参考线上 CASE 扩写 ───
            case_source_id = (data.get('case_source_id') or '').strip()
            if not case_source_id:
                return jsonify({'success': False, 'error': '参考扩写模式需要选择一个数据源 (case_source_id)'}), 400
            goals = data.get('goals') or []
            if isinstance(goals, str):
                goals = [g.strip() for g in goals.split('\n') if g.strip()]
            dispatcher = CaseSourceDispatcher(api_key=api_key, base_url=api_base_url)
            persona_list = dispatcher.dispatch_ref(
                agent_system_prompt=system_prompt,
                case_source_id=case_source_id,
                total_n=total_n,
                conversation_goals=goals,
            )
            allocation = {"ref_source": case_source_id, "expanded_count": len(persona_list)}
            analysis = f"基于数据源 {case_source_id} 扩写生成 {len(persona_list)} 条用例"
            cases = [{"goal": p.goal, "type": "reference_expansion"} for p, _ in persona_list]
            allocation_mode = "ref"

        elif allocation_mode == 'persona_pool':
            personas_payload = data.get('personas') or []
            goals = data.get('goals') or []
            if isinstance(goals, str):
                goals = [g.strip() for g in goals.split('\n') if g.strip()]
            pool_allocator = PersonaPoolAllocator()
            alloc_result = pool_allocator.allocate(
                personas=personas_payload,
                total_n=total_n,
                goals=goals,
                identity=data.get('identity', ''),
                scene_premise=data.get('scene_premise', ''),
                agent_system_prompt=system_prompt,
                trace_collector=None,
            )
            persona_list = pool_allocator.cases_to_personas(alloc_result.get('cases', []))
            pool_meta = alloc_result.get('_pool_meta', {})
        else:
            allocator = CaseAllocator(api_key=api_key, base_url=api_base_url)
            alloc_result = allocator.allocate(
                agent_system_prompt=system_prompt,
                total_n=total_n,
                preference=preference,
                trace_collector=None,
                generation_requirements=generation_requirements,
            )
            persona_list = allocator.cases_to_personas(alloc_result.get('cases', []))

        allocation = alloc_result.get('allocation', {})
        analysis = alloc_result.get('analysis', '')
        cases = alloc_result.get('cases', [])

        if not cases:
            log_event(
                "app.agent_container_batch",
                "ALLOC_RESULT_EMPTY",
                allocation_mode=allocation_mode,
                total_n=total_n,
                allocation=json.dumps(allocation, ensure_ascii=False),
                persona_list_len=len(persona_list),
            )
            return jsonify({'success': False, 'error': '配比器未能生成测试用例'}), 400

        # 2. 构造 AgentConfig
        agent_config = AgentConfig(
            name=agent_name,
            system_prompt=system_prompt,
            model=data.get('model', DEFAULT_MODEL),
            api_key=api_key,
            base_url=api_base_url,
            temperature=float(data.get('temperature', 0.7)),
            context_variables=context_variables,
        )

        # 3. 并发执行
        pool = ResourcePool(api_key=api_key, base_url=api_base_url)
        orchestrator = BatchOrchestrator(pool=pool)
        run_logger = RunLogger(run_meta={
            "agent_name": agent_name,
            "total_n": total_n,
            "max_turns": max_turns,
            "case_mode": case_mode,
            "allocation_mode": allocation_mode,
            "preference": preference,
            "model": data.get('model', DEFAULT_MODEL),
        })
        global _current_progress_logger
        _current_progress_logger = ProgressLogger()
        batch_results = orchestrator.run_batch_sync(
            agent_config=agent_config,
            persona_list=persona_list,
            max_turns=max_turns,
            run_logger=run_logger,
            progress_logger=_current_progress_logger,
        )

        # 4. 生成聚合报告
        report_gen = BatchReportGenerator()
        report_result = report_gen.generate(
            batch_results=batch_results,
            allocation=allocation,
            analysis=analysis,
            agent_name=agent_name,
            total_n=total_n,
            allocation_mode=allocation_mode,
            pool_meta=pool_meta,
        )

        # 5. 持久化运行日志
        log_doc = run_logger.finalize()

        return jsonify({
            'success': True,
            'result': {
                "allocation_mode": allocation_mode,
                "total_n": total_n,
                "allocation": allocation,
                "pool_meta": pool_meta,
                "analysis": analysis,
                "scenarios": batch_results,
                "stats": report_result["stats"],
                "html_report": report_result["html_path"],
                "json_report": report_result["json_path"],
                "run_id": run_logger.run_id,
                "run_log_path": log_doc.get("_log_path"),
                "module_timings_total": log_doc.get("module_timings_total", {}),
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/batch_progress', methods=['GET'])
def batch_progress():
    run_id = request.args.get('run_id', '')
    if run_id:
        from evaluator.progress_logger import ProgressLogger as _PL, LOG_DIR as _LD
        log_path = _LD / f"{run_id}.jsonl"
        if log_path.exists():
            return jsonify({'success': True, 'run_id': run_id, 'log': log_path.read_text(encoding='utf-8')})
        return jsonify({'success': False, 'error': f'run_id {run_id} not found'}), 404
    if _current_progress_logger:
        return jsonify({
            'success': True,
            'run_id': _current_progress_logger.run_id,
            'log': _current_progress_logger.read_log(),
        })
    return jsonify({'success': True, 'run_id': None, 'log': '', 'available_runs': ProgressLogger.list_runs()})


@app.route('/api/batch_progress/list', methods=['GET'])
def batch_progress_list():
    return jsonify({'success': True, 'runs': ProgressLogger.list_runs()})


# ═══════════════════════════════════════════════════════════
#  Case Source Management APIs (PR1 - Ref 扩写)
# ═══════════════════════════════════════════════════════════

from evaluator.case_source_store import (
    create_source, get_source as _get_source, list_sources, delete_source
)
from evaluator.case_source_dispatcher import CaseSourceDispatcher


@app.route('/api/case_source/upload', methods=['POST'])
def case_source_upload():
    """上传 / 新建一个 case source（种子 cases）。

    Body:
    {
        "name": "Q2 退款 case 集",
        "cases": [
            {"goal": "用户要退款", "messages": [...]}  // messages 可选
        ]
    }
    """
    try:
        data = request.json
        name = (data.get('name') or '').strip() or '未命名数据源'
        cases = data.get('cases', [])
        if not cases or not isinstance(cases, list):
            return jsonify({'success': False, 'error': '至少提供 1 条种子 case'}), 400
        doc = create_source(name, cases)
        return jsonify({'success': True, 'source': doc})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/case_source/list', methods=['GET'])
def case_source_list():
    """列出所有 case source（摘要）。"""
    try:
        sources = list_sources()
        return jsonify({'success': True, 'sources': sources})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/case_source/<source_id>', methods=['GET'])
def case_source_detail(source_id):
    """获取单个 source 详情（含完整 cases）。"""
    try:
        doc = _get_source(source_id)
        if doc is None:
            return jsonify({'success': False, 'error': 'Source not found'}), 404
        return jsonify({'success': True, 'source': doc})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/case_source/<source_id>', methods=['DELETE'])
def case_source_delete(source_id):
    """删除 source。"""
    try:
        ok = delete_source(source_id)
        if not ok:
            return jsonify({'success': False, 'error': 'Source not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run_logs/list', methods=['GET'])
def api_run_logs_list():
    """列出最近的批量评估运行日志摘要。"""
    try:
        limit = int(request.args.get('limit', 50))
        limit = max(1, min(limit, 200))
        logs = list_run_logs(limit=limit)
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run_logs/<run_id>', methods=['GET'])
def api_run_log_detail(run_id):
    """获取单次批量评估运行日志详情。"""
    try:
        doc = get_run_log(run_id)
        if doc is None:
            return jsonify({'success': False, 'error': 'Run log not found'}), 404
        return jsonify({'success': True, 'log': doc})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("图灵机-Agent多轮对话评估平台 - Web版")
    print("="*60)
    print("\n🚀 服务器启动中...")
    print("📱 访问地址: http://localhost:5001")
    print("="*60)
    app.run(host='0.0.0.0', port=5001, debug=True)
