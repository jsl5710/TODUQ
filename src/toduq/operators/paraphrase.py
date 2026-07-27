"""paraphrase — meaning- and belief-state-preserving control.

The gold action MUST stay `answer`. A system that abstains here is
over-abstaining on surface noise. Belief state is unchanged.
"""
from __future__ import annotations

from typing import Optional

from toduq.operators.base import Operator
from toduq.prompts import render_paraphrase_prompt
from toduq.routing.gold_action import derive_action, derive_severity
from toduq.runners.base import LLMClient
from toduq.schema import AnalysePass, ApplyPass, DocumentPass, Turn


class Paraphrase(Operator):
    id = "paraphrase"
    family = "paraphrase"
    uncertainty_type = "input"  # nominal; no uncertainty is actually induced

    def is_applicable(self, turn: Turn) -> bool:
        return bool(turn.utterance.strip())

    def analyse(self, turn: Turn) -> AnalysePass:
        return AnalysePass(
            modifiable=True,
            candidate_operators=["paraphrase"],
            rationale="Control: reword while preserving meaning and belief state.",
        )

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        severity = derive_severity(self.id)  # -> "none"
        return DocumentPass(
            operator=self.id,
            change_from=turn.utterance,
            change_to=turn.utterance,  # filled by apply(); meaning is identical
            slot_delta={},             # invariant: paraphrase never edits slots
            intended_uncertainty=self.uncertainty_type,
            expected_severity=severity,
            gold_action=derive_action(self.id, self.uncertainty_type, severity),  # answer
            gold_clarification_question=None,
            gold_query=None,
        )

    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        variants: list[str] = []
        method = "identity"
        modified = turn.utterance
        if llm is not None:
            prompt = render_paraphrase_prompt(turn.utterance, n=3)
            out = llm.generate(prompt)
            variants = [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()][:3]
            if variants:
                modified, variants = variants[0], variants[1:]
                method = "llm_paraphrase"
        return ApplyPass(
            modified_utterance=modified,
            method=method,
            paraphrase_variants=variants,
            new_belief_state=self._clone_state(turn.belief_state),  # unchanged
        )
