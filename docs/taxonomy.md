# Uncertainty taxonomy

TODUQ organizes injection around the four **dimensions** of LLM uncertainty,
each grounded in the aleatoric / epistemic distinction. This page is the
reference the operators and labelers cite by name.

## Aleatoric vs. epistemic

- **Aleatoric** — irreducible uncertainty from noise/ambiguity in the *data* or
  *input*. A "perfect model" still cannot resolve it. Injecting it means making
  the turn genuinely ambiguous or underspecified.
- **Epistemic** — reducible uncertainty from the *model's lack of knowledge*.
  Injecting it means asking about something outside the model's/DB's knowledge.
  Reducible via retrieval, tools, or a stronger model — which is exactly why it
  routes to RAG / handoff / HITL.

## The four dimensions

### 1. Input uncertainty — *aleatoric*
The prompt is ambiguous or underspecified; no single definitive response exists.
> "Find me a place to eat." (which city? what cuisine?)

- **Injection site in SGD:** any turn that fills or references a required slot.
- **Gold action:** `clarify` — the missing information is *askable*.

### 2. Reasoning uncertainty — *mixed*
The answer needs multi-step reasoning or retrieval; per-step uncertainty
compounds. Aleatoric when the problem is ambiguous, epistemic when the model
cannot reason robustly.
> "Actually make it near the beach" after `city = San Jose` (contradiction to reconcile).

- **Injection site in SGD:** turns with accumulated slots to contradict; the
  multi-domain **service-switch** turn (cross-service dependency).
- **Gold action:** `handoff_llm` (minor) — hand to a stronger reasoner; `clarify`
  when a single question resolves it.

### 3. Parameter uncertainty — *epistemic*
Training-data / knowledge gap: the model never saw, or misrepresents, the fact.
> "Is the Ethiopian place on 5th still open?" (entity not in the service DB).

- **Injection site in SGD:** swap a DB-resolvable entity for an out-of-KB one, or
  request a slot the service schema cannot return.
- **Gold action:** `rag_structured` (DB-answerable fact), `rag_unstructured`
  (needs text context), or `hitl` (unknowable / high-stakes).

### 4. Prediction uncertainty — *mixed* — **v2, measured not injected**
Variance of outputs across sampling runs. Detected by sampling the
system-under-test N times and computing semantic dispersion (e.g. semantic
entropy / self-consistency), not by editing the turn.

- **Gold action:** escalate by severity — high dispersion on a high-stakes turn
  → `handoff_llm` or `hitl`.
- The `LLMClient.sample(n)` API exists in v1 so this drops in without refactor.

## Severity → routing

Severity is a property of the *injected uncertainty*, set during Pass 2
(Document) and confirmed in Pass 4.

| Severity | Meaning                                   | Default route      |
| -------- | ----------------------------------------- | ------------------ |
| `none`   | control / paraphrase; no real uncertainty | `answer`           |
| `minor`  | resolvable automatically                  | `handoff_llm` / `rag_*` / `clarify` |
| `major`  | risky, unknowable, or safety-relevant     | `hitl`             |

See [`routing.md`](routing.md) for the module contracts.
