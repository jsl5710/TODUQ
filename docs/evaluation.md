# Evaluation (Milestone 5)

The eval harness scores a system-under-test's per-turn decision — `(abstain?,
action, [clarification])` — against the gold labels in the seed set.

## Metrics (`toduq.eval`)

| Metric | What it answers |
| --- | --- |
| `abstention_accuracy` | Did the model abstain iff `gold.should_abstain`? |
| `routing_accuracy` | Given abstention, did it pick the right module (`gold.action`)? |
| `over_abstention_rate` | How often did it abstain when it should have answered? (paraphrase controls target this) |
| `expected_calibration_error` | Binned ECE — is the model's confidence calibrated to correctness? |
| `auroc` | How well does a should-abstain score rank should-abstain turns above the rest? |
| `semantic_entropy` | Dispersion across N samples (v2 prediction-uncertainty; exact-string clustering now, entailment/embedding later) |
| `uncertainty_bleed` | In a multi-domain dialogue, did perturbing service A move the model's state in service B? |

All metrics are pure-stdlib (no numpy) so they run offline; `semantic_entropy`
and `uncertainty_bleed` are the two measures the SGD structure uniquely enables.

## Running the system-under-test

The evaluated model is any `LLMClient` (`role: system_under_test` in
`configs/models/models.yaml`) — open or closed. For prediction uncertainty, call
`client.sample(prompt, n)` and feed the results to `semantic_entropy`. The
`sample(n)` API exists in v1 precisely so this drops in without a refactor.

## Cross-service uncertainty bleed

For a multi-domain dialogue (e.g. `Music_1 + Events_1`), generate a sample that
perturbs one service, run the model on both the clean and perturbed dialogue, and
compare the resulting belief states with `uncertainty_bleed(base, perturbed,
perturbed_service=...)`. A nonzero result means the injected uncertainty leaked
into a frame it shouldn't have touched — a failure mode single-domain data can't
surface.
