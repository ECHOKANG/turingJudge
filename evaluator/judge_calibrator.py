from dataclasses import dataclass
from typing import List, Dict, Optional
from .llm_judge import EvaluationResult


@dataclass
class CalibrationCheck:
    case_id: str
    human_evaluation: Optional[Dict] = None
    judge_evaluation: Optional[EvaluationResult] = None
    consistency_score: float = 0.0


class JudgeCalibrator:
    def __init__(self):
        self.calibration_checks: List[CalibrationCheck] = []
        self.baseline_scores: Dict[str, float] = {}

    def add_human_evaluation(
        self,
        case_id: str,
        human_scores: Dict[str, float],
        human_analysis: str,
        judge_result: EvaluationResult
    ) -> CalibrationCheck:
        consistency = self._calculate_consistency(human_scores, judge_result.dimension_scores)
        
        check = CalibrationCheck(
            case_id=case_id,
            human_evaluation={
                "scores": human_scores,
                "analysis": human_analysis
            },
            judge_evaluation=judge_result,
            consistency_score=consistency
        )
        
        self.calibration_checks.append(check)
        return check

    def _calculate_consistency(self, human_scores: Dict[str, float], judge_scores: Dict[str, float]) -> float:
        common_dims = set(human_scores.keys()) & set(judge_scores.keys())
        if not common_dims:
            return 0.0
        
        total_diff = 0.0
        for dim in common_dims:
            total_diff += abs(human_scores[dim] - judge_scores[dim])
        
        avg_diff = total_diff / len(common_dims)
        return max(0.0, 10.0 - avg_diff) / 10.0

    def get_overall_consistency(self) -> float:
        if not self.calibration_checks:
            return 1.0
        
        total_consistency = sum(check.consistency_score for check in self.calibration_checks)
        return total_consistency / len(self.calibration_checks)

    def generate_calibration_report(self) -> Dict:
        if not self.calibration_checks:
            return {
                "status": "insufficient_data",
                "message": "没有足够的校准数据"
            }
        
        overall_consistency = self.get_overall_consistency()
        
        dimension_consistency = {}
        for check in self.calibration_checks:
            if check.human_evaluation and check.judge_evaluation:
                for dim, h_score in check.human_evaluation["scores"].items():
                    if dim in check.judge_evaluation.dimension_scores:
                        if dim not in dimension_consistency:
                            dimension_consistency[dim] = []
                        j_score = check.judge_evaluation.dimension_scores[dim]
                        dim_consistency = max(0.0, 10.0 - abs(h_score - j_score)) / 10.0
                        dimension_consistency[dim].append(dim_consistency)
        
        avg_dim_consistency = {
            dim: sum(scores) / len(scores)
            for dim, scores in dimension_consistency.items()
        }
        
        return {
            "status": "calibrated",
            "overall_consistency": overall_consistency,
            "dimension_consistency": avg_dim_consistency,
            "calibration_count": len(self.calibration_checks),
            "recommendation": "需要继续校准" if overall_consistency < 0.7 else "校准良好"
        }

    def adjust_judge_scores(
        self,
        results: List[EvaluationResult],
        adjustment_factor: Optional[float] = None
    ) -> List[EvaluationResult]:
        if adjustment_factor is None:
            adjustment_factor = 1.0 - (1.0 - self.get_overall_consistency()) * 0.5
        
        adjusted_results = []
        for result in results:
            adjusted_dim_scores = {
                dim: min(10.0, max(0.0, score * adjustment_factor))
                for dim, score in result.dimension_scores.items()
            }
            
            adjusted_overall = sum(adjusted_dim_scores.values()) / len(adjusted_dim_scores)
            
            adjusted_result = EvaluationResult(
                case_id=result.case_id,
                agent_name=result.agent_name,
                overall_score=adjusted_overall,
                dimension_scores=adjusted_dim_scores,
                qualitative_analysis=result.qualitative_analysis,
                bad_cases=result.bad_cases,
                conversation_history=result.conversation_history,
                metadata={
                    **result.metadata,
                    "adjusted": True,
                    "adjustment_factor": adjustment_factor
                }
            )
            
            adjusted_results.append(adjusted_result)
        
        return adjusted_results
