from dataclasses import replace
from typing import Dict, List, Any
from simulator import Persona


PERSONA_VARIANTS = {
    "normal": {
        "personality_traits": ["理性", "配合", "务实"],
        "speaking_style": "自然流畅，语气平和",
    },
    "hesitant": {
        "personality_traits": ["犹豫", "顾虑多", "谨慎", "反复确认"],
        "speaking_style": "语气犹豫，经常说'我再想想''真的没问题吗'，需要多次确认",
    },
    "aggressive": {
        "personality_traits": ["急躁", "质疑", "不耐烦", "强势"],
        "speaking_style": "语气急促，经常打断，直接质疑'你确定？''别绕弯子'",
    },
}


def create_persona_variants(base_persona: Persona) -> Dict[str, Persona]:
    variants = {}
    for variant_name, overrides in PERSONA_VARIANTS.items():
        variant = replace(
            base_persona,
            personality_traits=overrides["personality_traits"],
            speaking_style=overrides["speaking_style"],
            name=f"{base_persona.name}_{variant_name}",
        )
        variants[variant_name] = variant
    return variants


def compute_pass_at_k(scores: List[float], threshold: float = 7.0) -> float:
    if not scores:
        return 0.0
    passing = sum(1 for s in scores if s >= threshold)
    return passing / len(scores)


def compute_stability(scores: List[float], max_range: float = 1.5) -> bool:
    if not scores:
        return False
    return (max(scores) - min(scores)) <= max_range


def aggregate_multi_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["overall_score"] for r in results]
    variant_details = []
    for r in results:
        variant_details.append({
            "variant": r.get("variant", "unknown"),
            "overall_score": r["overall_score"],
            "dimension_scores": r["dimension_scores"],
            "evidence_turns": r.get("evidence_turns", {}),
            "confidence": r.get("confidence", {}),
            "conversation_history": r.get("conversation_history", []),
            "rule_flags": r.get("rule_flags", []),
        })

    return {
        "variants": variant_details,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "pass_at_k": compute_pass_at_k(scores),
        "stable": compute_stability(scores),
    }
