# Injection operators (v1)

An **operator** is a deterministic transformation of one SGD user turn. It owns
the gold label; the LLM only rewords its output. Operators are grouped by
manipulation **family** and tagged with the **uncertainty dimension** they induce.

Legend — family: `P` paraphrase · `X` perturbation · `I` injection.

## Input uncertainty → `clarify`

| id                   | fam | What it does                                              | Slot delta                    |
| -------------------- | --- | -------------------------------------------------------- | ----------------------------- |
| `slot_drop`          | X   | Remove a required filled slot from the utterance         | `slot: value → null`          |
| `referential_ambig`  | X   | Replace a filled slot with an unresolved referent        | `city → "there"`              |
| `multi_value`        | X   | Offer two mutually-exclusive values for one slot          | `cuisine: {A} → {A or B}`     |
| `underspecify`       | X   | Generalize a specific request to an ambiguous one         | drop constraint, keep intent  |

## Parameter uncertainty → `rag_structured` / `rag_unstructured` / `hitl`

| id                   | fam | What it does                                              | Gold action                   |
| -------------------- | --- | -------------------------------------------------------- | ----------------------------- |
| `out_of_kb_entity`   | I   | Name an entity absent from the service DB                 | `rag_structured`              |
| `out_of_schema_req`  | I   | Ask for info the schema has no slot for                   | `rag_unstructured`            |
| `long_tail_entity`   | X   | Swap a common entity for a rare/long-tail one             | `rag_structured`              |
| `unknowable_fact`    | I   | Ask something no source can answer (future/private)       | `hitl`                        |

## Reasoning uncertainty → `handoff_llm` / `clarify`

| id                    | fam | What it does                                            | Needs               |
| --------------------- | --- | ------------------------------------------------------- | ------------------- |
| `cross_turn_contra`   | X   | New turn contradicts an accumulated `slot_value`         | filled slot history |
| `implicit_constraint` | I   | Add a constraint requiring world-knowledge inference     | —                   |
| `cross_service_dep`   | I   | At a service switch, make new intent depend on old frame | multi-domain dialog |

## Controls (all dimensions) → `answer`

| id                   | fam | What it does                                              | Gold action                   |
| -------------------- | --- | -------------------------------------------------------- | ----------------------------- |
| `paraphrase`         | P   | Reword, preserving meaning **and** belief state           | `answer` (must not change)    |

Controls are essential: a system that abstains on `paraphrase` is
**over-abstaining** on surface noise. We target a healthy ratio of controls to
perturbations in the seed set (configurable in `configs/injection/`).

## Operator contract

Each operator is a small class implementing:

```python
class Operator(Protocol):
    id: str
    family: Literal["paraphrase", "perturbation", "injection"]
    uncertainty_type: Literal["input", "reasoning", "parameter", "prediction"]

    def is_applicable(self, turn: Turn, state: BeliefState) -> bool: ...
    def analyse(self, turn: Turn, state: BeliefState) -> AnalysePass: ...
    def document(self, analysis: AnalysePass) -> DocumentPass: ...   # sets gold label
    def apply(self, spec: DocumentPass, llm: LLMClient) -> ApplyPass: ...
    def confirm(self, record, llm: LLMClient, judge: Judge) -> ConfirmPass: ...
```

`is_applicable` + `analyse` are pure/deterministic. `apply` and `confirm` may
call the LLM (paraphrasing) and judge (validation) respectively. The gold label
in `document` is never produced by an LLM.
