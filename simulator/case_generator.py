from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
from pathlib import Path
from config import DATA_DIR


class CaseMode(Enum):
    LLM_SIMULATION = "llm_simulation"
    REFERENCE_SIMULATION = "reference_simulation"
    ONLINE_REGRESSION = "online_regression"


@dataclass
class TestCase:
    case_id: str
    mode: CaseMode
    conversation_goal: str
    initial_message: Optional[str] = None
    reference_case: Optional[Dict] = None
    persona_constraints: Dict = field(default_factory=dict)


class CaseGenerator:
    def __init__(self):
        self.cases: List[TestCase] = []
        self._case_counter = 0

    def generate_llm_simulation_case(
        self,
        conversation_goal: str,
        persona_constraints: Optional[Dict] = None,
        initial_message: Optional[str] = None
    ) -> TestCase:
        self._case_counter += 1
        case = TestCase(
            case_id=f"case_{self._case_counter}",
            mode=CaseMode.LLM_SIMULATION,
            conversation_goal=conversation_goal,
            initial_message=initial_message,
            persona_constraints=persona_constraints or {}
        )
        self.cases.append(case)
        return case

    def generate_reference_simulation_case(
        self,
        conversation_goal: str,
        reference_case: Dict,
        persona_constraints: Optional[Dict] = None
    ) -> TestCase:
        self._case_counter += 1
        case = TestCase(
            case_id=f"case_{self._case_counter}",
            mode=CaseMode.REFERENCE_SIMULATION,
            conversation_goal=conversation_goal,
            reference_case=reference_case,
            persona_constraints=persona_constraints or {}
        )
        self.cases.append(case)
        return case

    def load_online_cases(self, case_file: str) -> List[TestCase]:
        case_path = DATA_DIR / case_file
        if not case_path.exists():
            return []
        
        import json
        with open(case_path, 'r', encoding='utf-8') as f:
            cases_data = json.load(f)
        
        for data in cases_data:
            self._case_counter += 1
            case = TestCase(
                case_id=f"online_case_{self._case_counter}",
                mode=CaseMode.ONLINE_REGRESSION,
                conversation_goal=data.get("goal", ""),
                reference_case=data
            )
            self.cases.append(case)
        
        return self.cases[-len(cases_data):]

    def get_all_cases(self) -> List[TestCase]:
        return self.cases.copy()

    def get_case(self, case_id: str) -> Optional[TestCase]:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def sample_cases(self, n: int) -> List[TestCase]:
        if n >= len(self.cases):
            return self.cases.copy()
        return random.sample(self.cases, n)
