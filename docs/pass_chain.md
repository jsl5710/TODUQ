# The chain-of-passes pipeline

Every modified turn is produced by a **5-pass chain**. Each pass is a discrete,
inspectable step, and the whole thing serializes to **one JSON record** in which
each pass is retrievable by key (`passes.analyse`, `passes.document`,
`passes.apply`, `passes.confirm`, `passes.edit`). This separation is deliberate:
it keeps the *label* (owned by deterministic templates) independent of the
*wording* (owned by the LLM), with a *judge* guarding the join and a *finalize*
step guaranteeing one canonical output.

```
        ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐
SGD ──► │1 ANALYSE│─►│2 DOCUMENT│─►│3 APPLY  │─►│4 CONFIRM │─►│5 EDIT  │──► record
turn +  │ where   │  │ from→to, │  │ make the│  │ did it   │  │ repair │
belief  │ can X   │  │ slot     │  │ edit +  │  │ land?    │  │ or copy│
state   │ go?     │  │ delta,   │  │ para-   │  │ judge-   │  │ final  │
        │ target  │  │ gold     │  │ phrase  │  │ gate     │  │ version│
        └─────────┘  └──────────┘  └─────────┘  └──────────┘  └────────┘
```

## Pass 1 — Analyse  (`passes.analyse`)
**Input:** the SGD turn, its belief state, and a target `uncertainty_type`.
**Job:** decide whether that uncertainty type can be injected here and *where*.
Identifies the target slot/intent/frame and gives a rationale. If the turn is not
a viable site (`modifiable: false`), the chain stops and the record is dropped
with a reason.

Key fields: `modifiable`, `target_slot`, `target_intent`, `target_service`,
`candidate_operators`, `rationale`.

## Pass 2 — Document  (`passes.document`)
**Input:** Pass-1 analysis + the chosen operator.
**Job:** fully specify the change *before* making it. Records the exact
`change_from` → `change_to` intent, the `slot_delta` (before/after values), the
`intended_uncertainty`, `expected_severity`, and the **derived** `gold_action`
(+ `gold_clarification_question` or `gold_query` payload where relevant).

This pass owns the **gold label**. It is template/rule-driven so labels are
reproducible; the LLM is only asked to fill natural-language fields (like the
clarification question), never the label itself.

## Pass 3 — Apply  (`passes.apply`)
**Input:** the Pass-2 change spec.
**Job:** realize the edit. The template operator produces the canonical modified
utterance; an LLM paraphrase pass produces `paraphrase_variants` (fluent surface
forms that preserve the Pass-2 meaning). Records `modified_utterance`, `method`,
`variants`, and the updated frame state.

## Pass 4 — Confirm  (`passes.confirm`)
**Input:** everything above.
**Job:** verify the change actually happened and is valid. Two layers:
1. **Structural checks** (deterministic): was the target slot really
   dropped/added? does the new belief state match the Pass-2 spec?
2. **LLM-judge gate**: `fidelity` (does the edit match the documented intent?),
   `uncertainty_present` (is the intended uncertainty actually there?),
   `naturalness`. Optionally a control check that paraphrase variants did *not*
   change meaning.

Sets `status ∈ {accepted, needs_review, rejected}`. Only `accepted` records enter
the curated seed set; `needs_review` goes to the human queue.

## Pass 5 — Edit  (`passes.edit`)
**Input:** the full record after Confirm.
**Job:** produce the single canonical **final version** of the turn, and fix
anything Pass 4 flagged as missed. Two modes:
1. `copy` — Confirm found the change landed cleanly, so the applied output is
   promoted verbatim as `final_utterance` + `final_belief_state`. (The
   "copy the final version" case.)
2. `repair` — a structural check failed (e.g. a slot the spec said to drop is
   still present). Pass 5 deterministically re-enforces the documented
   `slot_delta`, re-verifies, and records each fix in `changes`.

Sets `final_status ∈ {finalized, unresolved}`. `unresolved` means deterministic
repair could not satisfy the documented spec — the record is routed to a human
regardless of Confirm's verdict. Pass 5 **never** overrides the human-review gate
in `confirm.status`; it only guarantees a usable final artifact and repairs
structural defects. Downstream consumers read `passes.edit.final_utterance` as
the authoritative modified turn.

## Why five passes instead of one prompt

- **Auditability** — each decision is a separate, inspectable key.
- **Label integrity** — the gold label is set by a rule (Pass 2), not buried in a
  generation. The LLM never invents labels.
- **Cheap controls** — paraphrase variants and their "meaning unchanged" check
  fall out of Passes 3–4 for free.
- **Reproducibility** — re-running with a fixed seed + fixed templates reproduces
  labels exactly; only wording varies.
- **One canonical output** — Pass 5 guarantees every record ends with exactly one
  authoritative final version, self-repairing structural defects instead of
  silently shipping them.

See [`annotation_schema.md`](annotation_schema.md) for the full record and
[`examples/record_slot_drop.json`](../examples/record_slot_drop.json) for a
worked instance.
