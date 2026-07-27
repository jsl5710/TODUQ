# Shared UQ layer (`uq/`)

One implementation of uncertainty-quantification methods, **vendored identically**
into TODUQ (`toduq.uq`) and TODUQ-MoA (`toduq_moa.uq`), loadable by name so any
method of choice drives either system. (Destined for a shared `tod-core`
package; the files are byte-identical across repos.)

## Load and use any method

```python
from <pkg>.uq import load_uq, available

available()                       # ['lexical', 'self_consistency', 'semantic_entropy', 'verbalized_confidence']
uq = load_uq("lexical")           # offline, no model
uq = load_uq("semantic_entropy", n=8)   # response-based, needs a client
r = uq.score(text, context=history, client=llm)   # -> UQScore(score, method, uncertainty_type, detail)
```

`UQScore.score` is a `[0, 1]` uncertainty; `uncertainty_type` is one of
`input | reasoning | parameter | prediction` (or `None`). `client` is any object
with `generate(prompt)` / `sample(prompt, n)` — the TODUQ and MoA runners both
satisfy it; input-based methods ignore it.

## Methods

| name | kind | needs model | signal |
| --- | --- | --- | --- |
| `lexical` | input-based | no | hedge / underspecification markers (`input`) + unknowable-fact markers (`parameter`) |
| `semantic_entropy` | response-based | yes | normalized entropy over N sampled responses (`prediction`) |
| `self_consistency` | response-based | yes | 1 − agreement (top-cluster fraction) over N samples |
| `verbalized_confidence` | response-based | yes | asks the model its confidence; score = 1 − confidence |

Semantic clustering is an exact-string stand-in in v1; swap in an
entailment/embedding clusterer without touching callers.

## How each repo uses it

- **TODUQ** — the simulator metric is a thin adapter over a loaded method:
  `toduq.cli simulate --metric semantic_entropy`. See `docs/simulator.md`.
- **TODUQ-MoA** — the gate composes a loaded method with the safety screen:
  `Gate("semantic_entropy", client=llm)`. The default `HeuristicGate` is
  `Gate("lexical")`. The safety screen always precedes the UQ score, so a
  harmful input escalates to HITL regardless of the method chosen.

## Adding a method

1. Add a class to `uq/methods.py` (implement `score(text, *, context, client)`).
2. Register it in `uq/registry.py`.
3. It is immediately loadable by name in both repos — keep the two `uq/` copies
   byte-identical.
