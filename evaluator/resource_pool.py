"""共享资源池：复用 AgentContainer / UserSimulator / LLMJudge 工厂。

设计目标：
- 每次评估任务都新建一组隔离的 container/simulator/judge，避免并发命名冲突。
- 但底层 OpenAI/Ark client 通过参数共享（SDK 客户端本身是线程安全的）。
- 不依赖 langfuse。
"""
from __future__ import annotations

from typing import Optional
import os

from agent_container import AgentContainer
from simulator import UserSimulator
from evaluator.llm_judge import LLMJudge
from config import SIMULATOR_MODEL, JUDGE_MODEL


class ResourcePool:
    """轻量工厂：根据配置批量产出隔离的 evaluator 组件。

    单 case 用 `make_for_case(case_idx)` 拿一组全新的 container/simulator/judge,
    用完即丢，避免 register_agent 的命名冲突 + prefix cache 串台。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        simulator_model: str = None,
        judge_model: str = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.simulator_model = simulator_model or SIMULATOR_MODEL
        self.judge_model = judge_model or JUDGE_MODEL

    def make_agent_container(self) -> AgentContainer:
        return AgentContainer()

    def make_user_simulator(self) -> UserSimulator:
        return UserSimulator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.simulator_model,
        )

    def make_judge(self) -> LLMJudge:
        return LLMJudge(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.judge_model,
        )
