# Positioning — where in the dialogue to inject

Uncertainty must not always land on the same turn. A task-oriented dialogue has
many user turns; each generated **sample** places the injection at a **different
turn**, so across the dataset the perturbation is spread over early / middle /
late positions. This enables experiments a single fixed site cannot:

- **Depth sensitivity** — does a model's abstention degrade as ambiguity appears
  later, after more context has accumulated?
- **Position invariance** — is detection robust to *where* the uncertainty is?
- **Service-switch proximity** (multi-domain) — does injecting near the
  unscripted domain switch behave differently from far from it?

## Pipeline

```
user_turns + operators
      │  enumerate_sites()      every (turn, operator) where the operator applies
      ▼
   candidate Sites  ──►  select_sites(policy)  ──►  run_chain() per chosen Site
      │                                                    │
      └── each Site carries a Position (ordinal, band)     └── one Record per site
```

## Position

Every record carries `position`:

| Field                | Meaning                                                    |
| -------------------- | ---------------------------------------------------------- |
| `user_turn_ordinal`  | 0-based index among the dialogue's **user** turns          |
| `num_user_turns`     | total user turns in the dialogue                           |
| `relative_position`  | `ordinal / (num_user_turns - 1)`, in `[0, 1]`              |
| `band`               | `early` (<⅓) · `middle` (<⅔) · `late` (≥⅔)                 |

## Applicability decides sites, not just presence

A site exists only where the operator is genuinely valid. For `slot_drop` that
means the turn must **verbalize** a slot value — turn 1 ("in San Jose") and
turn 2 ("American food") are sites; a later turn that only *requests* info
("give me the address") is **not** a slot-drop site (it's a site for
parameter/RAG operators instead). This keeps injections faithful to what each
turn actually says.

## Selection policies

Set via `configs/injection/*.yaml` → `positioning.policy`.

| Policy                 | Behavior                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `all`                  | every applicable site (exhaustive)                             |
| `one_per_turn`         | at most one operator per user turn                             |
| `stratified_position`  | draw evenly across early/middle/late bands; `k` caps the count |
| `n_per_dialogue`       | exactly `k` sites, maximally spread across bands               |

Selection is seeded (`positioning.seed`) so the same config reproduces the same
sites. `stratified_position` round-robins across bands so no position dominates
the dataset.

## Multi-site (v2)

v1 injects **one** site per sample. Injecting *two* correlated sites in a single
dialogue (e.g. an ambiguity in `Music_1` that only bites after the switch to
`Events_1`) is the natural v2 extension for measuring cross-service
uncertainty **bleed**; the `Site` list already supports enumerating such pairs.
