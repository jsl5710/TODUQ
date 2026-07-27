"""TODUQ Simulator — replay a TODUQ sample turn-by-turn and test whether a UQ
metric flags uncertainty at the injected turn."""
from toduq.simulator.bot import Chatbot
from toduq.simulator.metrics import (
    LexicalUncertaintyMetric,
    SemanticEntropyMetric,
    UQMetric,
    VerbalizedConfidenceMetric,
)
from toduq.simulator.simulator import (
    SimResult,
    TurnScore,
    perturbed_user_turns,
    simulate_record,
)

__all__ = [
    "Chatbot", "UQMetric", "LexicalUncertaintyMetric", "SemanticEntropyMetric",
    "VerbalizedConfidenceMetric", "SimResult", "TurnScore",
    "simulate_record", "perturbed_user_turns",
]
