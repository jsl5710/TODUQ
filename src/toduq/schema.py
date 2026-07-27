"""Dataclass mirror of data/schema/annotation.schema.json.

Kept dependency-free (stdlib dataclasses) so the schema is importable without
pydantic. `to_dict()` produces JSON matching the JSON Schema; validation against
the schema file lives in tests and in `toduq.validate`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Action = Literal["answer", "clarify", "rag_structured", "rag_unstructured", "handoff_llm", "hitl"]
UncertaintyType = Literal["input", "reasoning", "parameter", "prediction"]
UncertaintySource = Literal["aleatoric", "epistemic", "mixed"]
Family = Literal["paraphrase", "perturbation", "injection"]
Severity = Literal["none", "minor", "major"]
Status = Literal["accepted", "needs_review", "rejected"]


@dataclass
class Frame:
    active_intent: Optional[str] = None
    requested_slots: list[str] = field(default_factory=list)
    slot_values: dict[str, Any] = field(default_factory=dict)


# belief_state maps a service name -> Frame
BeliefState = dict[str, Frame]


@dataclass
class Turn:
    utterance: str
    belief_state: BeliefState = field(default_factory=dict)
    # Per-turn slot provenance (populated by the SGD ingest):
    #   verbalized_slots: service -> [slots whose value is SPOKEN in this utterance]
    #                     (from SGD frame `slots` spans — the ground truth for
    #                     "what this turn actually says", used by operators).
    #   introduced_slots: service -> [slots NEW/changed vs. the previous user turn]
    verbalized_slots: dict[str, list[str]] = field(default_factory=dict)
    introduced_slots: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Position:
    """Where in the dialogue the injected turn sits, among all user turns."""
    user_turn_ordinal: int              # 0-based index among USER turns
    num_user_turns: int
    relative_position: float            # ordinal / (num_user_turns - 1), in [0, 1]
    band: Literal["early", "middle", "late"] = "early"


@dataclass
class GoldQuery:
    service: str
    intent: str
    constraints: dict[str, Any] = field(default_factory=dict)
    retrieval_kind: Literal["structured", "unstructured"] = "structured"


@dataclass
class AnalysePass:
    modifiable: bool
    target_service: Optional[str] = None
    target_slot: Optional[str] = None
    target_intent: Optional[str] = None
    candidate_operators: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DocumentPass:
    operator: str
    change_from: str
    change_to: str
    slot_delta: dict[str, dict[str, Any]]
    intended_uncertainty: UncertaintyType
    expected_severity: Severity
    gold_action: Action
    gold_clarification_question: Optional[str] = None
    gold_query: Optional[GoldQuery] = None


@dataclass
class ApplyPass:
    modified_utterance: str
    method: str
    paraphrase_variants: list[str] = field(default_factory=list)
    new_belief_state: BeliefState = field(default_factory=dict)


@dataclass
class ConfirmPass:
    change_applied: bool
    status: Status
    structural_checks: dict[str, Any] = field(default_factory=dict)
    judge_verdict: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class EditPass:
    """Pass 5 — finalize. Either copies the confirmed output forward as the
    canonical final version, or repairs a defect Pass 4 flagged."""
    mode: Literal["copy", "repair"]
    final_utterance: str
    final_belief_state: BeliefState = field(default_factory=dict)
    final_status: Literal["finalized", "unresolved"] = "finalized"
    changes: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Gold:
    action: Action
    severity: Severity
    should_abstain: bool
    clarification_question: Optional[str] = None
    query: Optional[GoldQuery] = None


@dataclass
class Provenance:
    sgd_version: str
    seed: int
    generator_model: Optional[str] = None
    judge_model: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Record:
    record_id: str
    dialogue_id: str
    turn_idx: int
    services: list[str]
    family: Family
    operator: str
    uncertainty_type: UncertaintyType
    uncertainty_source: UncertaintySource
    position: Position
    source: Turn
    passes_analyse: AnalysePass
    passes_document: DocumentPass
    passes_apply: ApplyPass
    passes_confirm: ConfirmPass
    passes_edit: EditPass
    gold: Gold
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk JSON layout (passes nested under `passes`)."""
        def frames(bs: BeliefState) -> dict[str, Any]:
            return {svc: asdict(fr) for svc, fr in bs.items()}

        d = {
            "record_id": self.record_id,
            "dialogue_id": self.dialogue_id,
            "turn_idx": self.turn_idx,
            "services": self.services,
            "family": self.family,
            "operator": self.operator,
            "uncertainty_type": self.uncertainty_type,
            "uncertainty_source": self.uncertainty_source,
            "position": asdict(self.position),
            "source": {
                "utterance": self.source.utterance,
                "belief_state": frames(self.source.belief_state),
            },
            "passes": {
                "analyse": asdict(self.passes_analyse),
                "document": _document_to_dict(self.passes_document),
                "apply": {
                    "modified_utterance": self.passes_apply.modified_utterance,
                    "method": self.passes_apply.method,
                    "paraphrase_variants": self.passes_apply.paraphrase_variants,
                    "new_belief_state": frames(self.passes_apply.new_belief_state),
                },
                "confirm": asdict(self.passes_confirm),
                "edit": {
                    "mode": self.passes_edit.mode,
                    "final_utterance": self.passes_edit.final_utterance,
                    "final_belief_state": frames(self.passes_edit.final_belief_state),
                    "final_status": self.passes_edit.final_status,
                    "changes": self.passes_edit.changes,
                    "notes": self.passes_edit.notes,
                },
            },
            "gold": {
                "action": self.gold.action,
                "severity": self.gold.severity,
                "should_abstain": self.gold.should_abstain,
                "clarification_question": self.gold.clarification_question,
                "query": asdict(self.gold.query) if self.gold.query else None,
            },
            "provenance": asdict(self.provenance),
        }
        return d


def _document_to_dict(dp: DocumentPass) -> dict[str, Any]:
    d = asdict(dp)
    d["gold_query"] = asdict(dp.gold_query) if dp.gold_query else None
    return d
