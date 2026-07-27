# TODUQ — Task-Oriented Dialogue Uncertainty Quantification

A dataset-development framework for injecting **controlled uncertainty** into
task-oriented dialogue (TOD) and measuring whether a conversational / agentic AI
system correctly **abstains** and **routes** to a fallback module instead of
answering blindly.

TODUQ builds on the [Schema-Guided Dialogue (SGD)](https://huggingface.co/datasets/GEM/schema_guided_dialog)
dataset. SGD annotates every turn with per-service **frames** carrying
`state = {active_intent, requested_slots, slot_values}`. That structured belief
state is what lets us inject uncertainty at a *specific* slot/intent and produce
**grounded gold labels** rather than best-guess ones.

## The idea in one loop

```
SGD user turn ──► inject controlled uncertainty ──► system-under-test
                                                          │
                        answer confidently  ◄────────────┤
                                                          │ abstain + route to:
                                     ┌────────────────────┼────────────────────┐
                                     ▼            ▼        ▼          ▼          ▼
                              human-in-loop  larger-LLM  clarify  RAG(struct) RAG(unstruct)
                               (major)       (minor)              DB query    text context
```

Each modified turn ships with the gold answer to: *should the model abstain
here, and if so, which module should it hand off to?*

## Uncertainty taxonomy (v1 scope)

Injection operators are organized by the four dimensions of LLM uncertainty
(Input, Reasoning, Parameter, Prediction). v1 implements the first three;
Prediction (sampling-variance) is measured, not injected, and lands in v2.

| Dimension (source)     | Operator on the SGD turn                                   | Gold action        |
| ---------------------- | ---------------------------------------------------------- | ------------------ |
| **Input** (aleatoric)  | slot-drop, referential ambiguity, multi-value, underspec   | `clarify`          |
| **Parameter** (epist.) | out-of-KB / out-of-schema entity, unknowable fact          | `rag_*` / `hitl`   |
| **Reasoning** (mixed)  | cross-turn contradiction, implicit constraint, cross-svc   | `handoff_llm`      |
| *Prediction* (v2)      | *measured via N-sample semantic entropy, not injected*     | escalate by sev.   |

See [`docs/taxonomy.md`](docs/taxonomy.md) and
[`docs/injection_operators.md`](docs/injection_operators.md).

## Positioning — spread across the dialogue

Uncertainty is **not** fixed to one turn. A dialogue has many user turns; each
sample injects at a **different** turn, so the dataset spreads perturbations over
early / middle / late positions. One dialogue therefore yields several samples,
each carrying a `position` (`user_turn_ordinal`, `band`, `relative_position`).
This supports depth-sensitivity and position-invariance experiments, and — for
multi-domain dialogues — injecting near vs. far from the service switch. Site
selection is applicability-aware (an operator only fires where it's valid) and
positionally stratified. See [`docs/positioning.md`](docs/positioning.md).

## Manipulation families

- **Paraphrase** — meaning- and belief-state-preserving. A *control*: gold
  action must stay `answer`. Catches over-abstention on surface noise.
- **Perturbation** — alters meaning/slots → changes belief state and gold action.
- **Injection** — adds content (distractor, out-of-KB entity, unanswerable aside).

## Chain-of-passes generation

Every modified turn is produced by a deterministic **5-pass LLM chain**, and the
result is a single JSON record where **each pass is retrievable by key**:

| Pass         | Key                | Question it answers                                            |
| ------------ | ------------------ | ------------------------------------------------------------- |
| 1 · Analyse  | `passes.analyse`   | *Where* can uncertainty type X be injected in this turn?      |
| 2 · Document | `passes.document`  | *What* changes — `from` → `to`, slot delta, gold action.      |
| 3 · Apply    | `passes.apply`     | *Make* the edit (template + LLM paraphrase variants).         |
| 4 · Confirm  | `passes.confirm`   | *Did* the change land? Judge-gate → accept/review/reject.     |
| 5 · Edit     | `passes.edit`      | *Fix* anything missed, else **copy the confirmed final version**. |

Pass 5 guarantees every record ends with one authoritative `final_utterance` +
`final_belief_state` to consume — repaired if Pass 4 found a defect, promoted
as-is otherwise.

The template operator owns the **label**; the LLM owns the **wording**; the judge
guards the join. See [`docs/pass_chain.md`](docs/pass_chain.md) and the worked
example in [`examples/record_slot_drop.json`](examples/record_slot_drop.json).

## Models & evaluation

- Provider-agnostic `LLMClient` protocol (`generate`, `sample(n)`) with adapters
  for **closed** (Claude, OpenAI) and **open** (vLLM / HF / Ollama) models.
- **LLM-as-judge** for (a) validating injections and (b) scoring system responses.
- Metrics: abstention accuracy, routing accuracy, calibration (ECE / AUROC on
  "should-abstain"), over/under-abstention, and cross-service **uncertainty bleed**
  (unique to multi-domain dialogues).

## Deliverable

v1 = a **curated seed set** (a few hundred hand-verified annotated turns spanning
single- and multi-domain dialogues) **plus** the reproducible pipeline that scales
to full SGD later.

## Repo layout

```
docs/         taxonomy · injection_operators · annotation_schema · pass_chain · routing
configs/      injection/ · models/ (open+closed) · judge/
src/toduq/    ingest/ operators/ uncertainty/ routing/ runners/ judge/ passes/ eval/
data/schema/  JSON Schema for the annotation record (+ dataset card)
prompts/      per-pass prompt templates
examples/     worked JSON records
tests/        schema + operator tests
```

## Status

**Milestones 1–5 done.** SGD ingest reconstructs per-turn belief state + slot
provenance; all eleven v1 operators are implemented and positionally spread via
the 5-pass chain; the seed-set generator + judge gate produce a schema-valid seed
set; and the eval harness scores abstention, routing, calibration (ECE / AUROC),
and cross-service uncertainty bleed. Provider-agnostic runners (Claude / OpenAI /
vLLM / Ollama) make the generator, judge, and system-under-test swappable.

Try it offline (no API key needed):

```bash
PYTHONPATH=src python -m toduq.cli dialogue    # one dialogue -> many samples
PYTHONPATH=src python -m toduq.cli generate    # build data/seed_v1/ (heuristic gate)
PYTHONPATH=src python -m toduq.cli generate --live   # LLM paraphrase + LLM judge
PYTHONPATH=src python -m toduq.cli demo        # one full JSON record
PYTHONPATH=src python -m toduq.cli simulate    # replay a sample; test UQ-metric localization
```

See [`data/seed_v1/`](data/seed_v1/) for the generated seed set,
[`docs/evaluation.md`](docs/evaluation.md) for the metrics, and
[`docs/simulator.md`](docs/simulator.md) for the **TODUQ Simulator** (replays a
sample turn-by-turn and tests whether a UQ metric flags uncertainty at the
injected turn). Pick the UQ method with `--metric`
(`lexical | semantic_entropy | self_consistency | verbalized_confidence`) — these
come from the **shared UQ layer** ([`docs/uq.md`](docs/uq.md)), one implementation
loadable by name and used by both TODUQ and TODUQ-MoA.

## License & attribution

Derived from SGD (Google Research, **CC BY-SA 4.0**). This repository inherits
CC BY-SA 4.0 for data artifacts; see [`LICENSE`](LICENSE). SGD utterances and
annotations are used under that license with attribution.
