# TODUQ Simulator

The simulator turns a single TODUQ sample into a **turn-by-turn evaluation of a
UQ metric**: it replays the perturbed dialogue through a chatbot module, scores
every user turn with an uncertainty metric, and checks whether the metric flags
uncertainty at the **turn where uncertainty was injected / paraphrased /
perturbed**.

```
TODUQ sample (Record) ─► reconstruct perturbed dialogue ─► replay turn-by-turn
                                                              │  per turn:
                                                              ▼
                              chatbot responds  +  UQ metric scores the turn
                                                              │
                                                              ▼
                          did the metric PEAK at the injected turn?  → identified
```

## Components (`src/toduq/simulator/`)

- **`Chatbot`** (`bot.py`) — the module under test. Wraps any `LLMClient` as a
  turn-taking TOD assistant (`respond`, `sample(n)`); offline `EchoClient` by
  default.
- **UQ metrics** (`metrics.py`) — each scores one turn in `[0, 1]`, **never seeing
  the injected-turn label**:
  - `LexicalUncertaintyMetric` — input-based, offline. Counts hedges /
    underspecification in the user turn.
  - `SemanticEntropyMetric(n)` — response-based. Semantic entropy over N bot
    samples (needs a live model).
  - `VerbalizedConfidenceMetric` — asks the model its confidence; score = 1−conf.
- **`simulate_record`** (`simulator.py`) — rebuilds the perturbed dialogue
  (`perturbed_user_turns`), replays it, and returns a `SimResult`
  (`turn_scores`, `predicted_ordinal`, `rank_of_injected`, `identified`).

## What "identified" means

- For an **injected** sample (`gold.should_abstain == True`): the metric's
  highest-scoring turn is the injected turn.
- For a **control** (`paraphrase`, `should_abstain == False`): **no** turn spikes
  — a good metric must not raise a false alarm on a meaning-preserving paraphrase.

## Two modes & metric choice

- `--mode history` (default) — each turn is scored with the **prior conversation
  history** (if any).
- `--mode immediate` — each turn is scored **in isolation** (current turn only).
- `--metric` — any method from the shared UQ registry (`lexical`,
  `semantic_entropy`, `self_consistency`, `verbalized_confidence`).

Input-based metrics (`lexical`) are mode-invariant; response-based ones
(`semantic_entropy`, `verbalized_confidence`) prompt the bot with vs without the
history, so mode changes their score (needs a live model to observe).

## Run it

```bash
PYTHONPATH=src python -m toduq.cli simulate --metric lexical --mode history
PYTHONPATH=src python -m toduq.cli simulate --metric semantic_entropy --mode immediate
```

Offline output (lexical metric):

```
slot_drop          injected@t1  peak@t1 rank=1 identified=True   scores>0: t1=0.5
underspecify       injected@t1  peak@t1 rank=1 identified=True   scores>0: t1=0.5
multi_value        injected@t1  peak@t1 rank=1 identified=True   scores>0: t1=0.5
referential_ambig  injected@t1  peak@t0 rank=2 identified=False  scores>0: (no spike)
out_of_kb_entity   injected@t0  peak@t0 rank=1 identified=False  scores>0: (no spike)
cross_turn_contra  injected@t2  peak@t2 rank=1 identified=True   scores>0: t2=0.5
paraphrase         injected@t0  peak@t0 rank=1 identified=True   scores>0: (no spike)
```

## Reading the result — metric coverage by uncertainty type

The lexical baseline localizes **lexicalized input uncertainty** (`slot_drop`,
`underspecify`, `multi_value`) and the **control**, but misses `referential_ambig`
(a pronoun swap — needs coreference) and `out_of_kb_entity` (a knowledge gap that
looks fluent — needs a **response-based** metric). This is the simulator's whole
point: it measures *which UQ metric catches which uncertainty type, at the right
turn*. Swap in `SemanticEntropyMetric` or `VerbalizedConfidenceMetric` with a live
model (via `configs/models/models.yaml`) to cover the response-based types.

## Aggregate evaluation

Across many samples, `SimResult.identified` and `rank_of_injected` give a metric's
**localization accuracy** and **mean rank** per uncertainty type — the headline
numbers for comparing UQ methods on TODUQ data.
