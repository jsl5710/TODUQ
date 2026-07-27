# Project Overview — the TODUQ / TODUQ-MoA / TODDC program

This document explains the **three-repository research program** these projects
form, how they connect, and where **TODUQ** sits in it.

## The problem

Conversational and agentic AI systems in **task-oriented dialogue** (booking,
search, support) fail in two coupled ways:

1. They **answer when they shouldn't** — confidently hallucinating instead of
   recognizing they are uncertain and deferring (asking, retrieving, escalating).
2. Even when they act correctly, the intervention (a clarification, a hand-off, a
   retrieval detour) can **break the discourse** — the dialogue stops cohering.

Both failures need **labeled data** to measure. This program builds it from a
common base — the **Schema-Guided Dialogue (SGD)** dataset, whose per-turn belief
state (`active_intent`, `requested_slots`, `slot_values`) grounds every label.

## The three repositories

| Repo | Builds | Question it answers |
| --- | --- | --- |
| **TODUQ** (this repo) | uncertainty-injection dataset | *Where is the model uncertain, and what should it do — clarify, RAG, hand off, or escalate to a human?* |
| **[TODUQ-MoA](https://github.com/jsl5710/TODUQ-MoA)** | mixture-of-agents inference system | *Given an uncertainty flag, route to the right experts and let a reasoning model aggregate.* |
| **[TODDC](https://github.com/jsl5710/TODDC)** | coherence-violation dataset | *Did the dialogue stay coherent — locally, referentially, globally, in relevance, and belief-state consistency?* |

```
   SGD ──► TODUQ (inject uncertainty, label route) ──► TODUQ-MoA (act: experts → aggregator)
    └────► TODDC (inject coherence violations, label) ──► evaluates the coherence of MoA output
```

## The shared method: chain-of-passes generation

Every sample is produced by a deterministic **5-pass chain**, serialized to one
JSON record with each pass retrievable by key:

`analyse` (where) → `document` (what change + **gold label**) → `apply` (make the
edit + LLM paraphrase) → `confirm` (judge-gate) → `edit` (finalize/repair).

Invariant across all repos: **the template operator owns the label, the LLM owns
the wording, the judge guards the join** — so gold labels are reproducible and
never buried in a model generation.

## TODUQ's role

TODUQ injects **controlled uncertainty** into SGD user turns and labels the
correct **abstain/route** action:

| Uncertainty (source) | Operators | Gold action |
| --- | --- | --- |
| Input (aleatoric) | slot_drop, referential_ambig, multi_value, underspecify | `clarify` |
| Parameter (epistemic) | out_of_kb_entity, out_of_schema_req, long_tail_entity, unknowable_fact | `rag_structured` / `rag_unstructured` / `hitl` |
| Reasoning (mixed) | cross_turn_contra, implicit_constraint, cross_service_dep | `handoff_llm` |
| — (control) | paraphrase | `answer` |

Injections are **spread across dialogue positions** (early/middle/late) so the
data supports depth-sensitivity and position-invariance experiments, and — in
multi-domain dialogues — cross-service **uncertainty bleed**.

**Pipeline stages** (milestones 1–5, all shipped): SGD ingest with per-turn slot
provenance → 11 operators → positional site selection → 5-pass chain → seed-set
generator + judge → eval (abstention/routing accuracy, ECE, AUROC, semantic
entropy, uncertainty bleed). Provider-agnostic runners (Claude / OpenAI / vLLM /
Ollama) make generator, judge, and system-under-test swappable.

## How the loop closes

1. **TODUQ** perturbs a user turn and labels the correct route.
2. **TODUQ-MoA** flags the uncertainty and routes it to experts, then aggregates.
3. **TODDC** checks whether the resulting dialogue stayed coherent.

Each repo runs standalone and offline; together they form an end-to-end pipeline
for building, acting on, and evaluating uncertainty-aware task-oriented dialogue.

## Consolidation note

All three repos currently each carry the shared infrastructure (SGD ingest,
`LLMClient` runners, judge, 5-pass pipeline). A planned `tod-core` package would
factor these out into one implementation the three import.
