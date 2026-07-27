# Annotation record schema

One injected turn = one JSON record. The canonical machine schema is
[`data/schema/annotation.schema.json`](../data/schema/annotation.schema.json)
(JSON Schema draft 2020-12); this page is the human-readable companion.

## Top level

| Field              | Type      | Notes                                                    |
| ------------------ | --------- | -------------------------------------------------------- |
| `record_id`        | string    | Stable id: `{dialogue_id}:{turn_idx}:{operator}:{seed}`  |
| `dialogue_id`      | string    | SGD dialogue id, e.g. `1_00000`                          |
| `turn_idx`         | int       | 0-based index of the user turn                           |
| `services`         | string[]  | Active SGD services/frames at this turn                  |
| `family`           | enum      | `paraphrase` \| `perturbation` \| `injection`            |
| `operator`         | string    | Operator id (see `injection_operators.md`)               |
| `uncertainty_type` | enum      | `input` \| `reasoning` \| `parameter` \| `prediction`    |
| `uncertainty_source`| enum     | `aleatoric` \| `epistemic` \| `mixed`                    |
| `source`           | object    | Original SGD turn + belief state (verbatim provenance)   |
| `passes`           | object    | The four passes, each retrievable by key (below)         |
| `gold`             | object    | Flattened final labels for convenient eval access        |
| `provenance`       | object    | dataset version, seed, model/judge ids, timestamps       |

## `source`

```jsonc
{
  "utterance": "I would like for it to be in San Jose.",
  "belief_state": {
    "Restaurants_1": {
      "active_intent": "FindRestaurants",
      "requested_slots": [],
      "slot_values": { "city": "San Jose" }
    }
  }
}
```

## `passes` — the chain output

Each key mirrors one pass (see [`pass_chain.md`](pass_chain.md)).

```jsonc
"passes": {
  "analyse":  { "modifiable": true, "target_service": "Restaurants_1",
                "target_slot": "city", "target_intent": "FindRestaurants",
                "candidate_operators": ["slot_drop", "referential_ambig"],
                "rationale": "..." },
  "document": { "operator": "slot_drop",
                "change_from": "I would like for it to be in San Jose.",
                "change_to":   "I would like to find somewhere to eat.",
                "slot_delta":  { "city": { "before": "San Jose", "after": null } },
                "intended_uncertainty": "input",
                "expected_severity": "minor",
                "gold_action": "clarify",
                "gold_clarification_question": "Which city should I search in?",
                "gold_query": null },
  "apply":    { "modified_utterance": "I would like to find somewhere to eat.",
                "method": "template+llm_paraphrase",
                "paraphrase_variants": ["I'm hungry — help me find a spot to eat?"],
                "new_belief_state": { "Restaurants_1": { "slot_values": {} } } },
  "confirm":  { "change_applied": true,
                "structural_checks": { "target_slot_removed": true,
                                       "belief_state_matches_spec": true },
                "judge_verdict": { "fidelity": "pass",
                                   "uncertainty_present": true,
                                   "naturalness": 0.92 },
                "status": "accepted",
                "notes": "" },
  "edit":     { "mode": "copy",
                "final_utterance": "I would like to find somewhere to eat.",
                "final_belief_state": { "Restaurants_1": { "slot_values": {} } },
                "final_status": "finalized",
                "changes": [],
                "notes": "Confirmed clean; promoted the applied version." }
}
```

`mode`: `copy` (Confirm was clean → promote the applied output) or `repair`
(a structural check failed → re-enforce the documented `slot_delta`).
`final_status`: `finalized` or `unresolved` (deterministic repair could not
satisfy the spec → human). `passes.edit.final_utterance` is the authoritative
modified turn for downstream consumers.

## `gold` — flattened for eval

A denormalized copy of the final labels so the eval harness needn't walk
`passes`:

```jsonc
"gold": {
  "action": "clarify",
  "severity": "minor",
  "clarification_question": "Which city should I search in?",
  "query": null,
  "should_abstain": true
}
```

## Status values

- `accepted` — passed structural + judge gates → enters seed set.
- `needs_review` — judge uncertain or a structural check soft-failed → human queue.
- `rejected` — edit did not land or introduced the wrong uncertainty → dropped.

## Design invariants (enforced by tests)

1. `gold.should_abstain == (gold.action != "answer")`.
2. `family == "paraphrase"` ⇒ `gold.action == "answer"` and `slot_delta` empty.
3. `uncertainty_type == "parameter"` ⇒ `gold.action ∈ {rag_structured, rag_unstructured, hitl}`.
4. `gold.action == "rag_structured"` ⇒ `passes.document.gold_query != null`.
5. `passes.confirm.status == "accepted"` ⇒ `change_applied == true`.
6. `passes.edit.mode == "copy"` ⇒ `passes.confirm.change_applied == true` (else `repair`).
