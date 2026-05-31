"""
MASS Agent 数据模型
"""
from .agent_response import (
    TAOpinion,
    FAOpinion,
    CAOpinion,
    SAOpinion,
    MAOpinion,
    RAOpinion,
    ChairmanDecision,
    DecisionPackage,
)

__all__ = [
    "TAOpinion", "FAOpinion", "CAOpinion", "SAOpinion",
    "MAOpinion", "RAOpinion", "ChairmanDecision", "DecisionPackage"
]
