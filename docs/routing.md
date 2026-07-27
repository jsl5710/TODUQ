# Routing modules & gold actions

When the system-under-test decides it should **not** answer, it abstains and
routes to one of five downstream actions. TODUQ labels the *gold* action for
every injected turn so routing can be scored.

## Action vocabulary

| `gold_action`      | Trigger                                             | Downstream module                         |
| ------------------ | -------------------------------------------------- | ----------------------------------------- |
| `answer`           | No real uncertainty (controls, paraphrase)         | — (respond normally)                      |
| `clarify`          | Missing/ambiguous info that is *askable*           | Ask the user a clarification question     |
| `rag_structured`   | Missing fact answerable by a **DB / API query**    | Structured retrieval (dataset query req.) |
| `rag_unstructured` | Needs **free-text context** to answer              | Unstructured retrieval (text from a KB)   |
| `handoff_llm`      | Minor issue; needs a **stronger reasoner**         | Escalate to a larger LLM                  |
| `hitl`             | Major issue; risky/unknowable/safety-relevant      | **Human-in-the-loop**                     |

Mapping to your original spec:
1. activate **human-in-the-loop** (major issue) → `hitl`
2. hand over to a **larger LLM** (minor issue) → `handoff_llm`
3. get **clarification** → `clarify`
4. **RAG**, structured (a dataset query request) or unstructured (text context)
   → `rag_structured` / `rag_unstructured`

## Structured vs. unstructured RAG — the decision rule

- `rag_structured` — the answer is a **slot value the service schema can hold**
  and would come from a DB/API lookup (e.g. a restaurant's phone number, an
  event date). The gold payload includes the intended **query** (service +
  intent + constraints).
- `rag_unstructured` — the answer needs **prose context** not expressible as
  slots (e.g. "is this venue wheelchair accessible?", policy/description text).
  The gold payload names the **retrieval intent**, not a slot query.

## Gold-action derivation

`gold_action` is *derived*, not free-chosen, from the operator + severity so it
stays consistent across the dataset. The mapping lives in
`src/toduq/routing/gold_action.py`; Pass 2 records the derived value and Pass 4
confirms it against the realized edit.

## Scoring (eval harness)

For each turn the system-under-test emits `(abstain?, action, [clarification])`.
We score:
- **Abstention accuracy** — abstain iff `gold_action != answer`.
- **Routing accuracy** — predicted action == `gold_action` (given abstention).
- **Clarification quality** — LLM-judge vs `gold_clarification_question`.
- **Calibration** — model confidence vs correctness (ECE, AUROC on should-abstain).
