# Generation on a server — multi-model, even split

For a large generation run you can split the Pass-3 paraphrasing **evenly across
several models** (Qwen, Llama 3.1, Mistral, …), each served on its own
OpenAI-compatible endpoint (vLLM / Ollama / TGI). Work is divided by a
`ModelPool`, each sample records the model that produced it, and sites can run in
parallel across endpoints.

## Option B — JHU AI Gateway (no self-hosting)

If you have a JHU gateway key, skip steps 1–2: the gateway fronts many providers
through one OpenAI-compatible endpoint, so you point the pool straight at it.

```bash
export GATEWAY_KEY=...            # your JHU gateway key (never commit it)
cp configs/models/gateway.example.yaml configs/models/models.yaml
PYTHONPATH=src python -m toduq.cli generate --dry-run          # expect all OK
PYTHONPATH=src python -m toduq.cli generate --live --workers 4
```

The shipped `gateway.example.yaml` splits both generation and judging evenly
across **`openai/gpt-4o-mini`** and **`anthropic/claude-haiku-4.5`**. The adapter
(`toduq.runners.gateway:GatewayClient`) posts to
`{base}/compat/chat/completions` with a bearer `GATEWAY_KEY` and uses
`max_completion_tokens` (the gateway's field). Override the host with
`GATEWAY_BASE`; add `send_temperature: false` to a spec for reasoning models that
reject `temperature`. The rest of this doc (self-hosted vLLM/Ollama) is the
alternative when you serve your own weights.

## 1. Serve the models

Each model on its own endpoint, e.g. with vLLM:

```bash
# gpu0
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --port 8000
# gpu1
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --port 8000
# gpu2
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-Small-Instruct-2409 --port 8000
```

## 2. Configure the pool

Copy [`configs/models/servers.example.yaml`](../configs/models/servers.example.yaml)
to `configs/models/models.yaml` and list the `generators` + split strategy:

```yaml
generators:
  - {adapter: toduq.runners.open_source:VLLMClient, model_id: Qwen/Qwen2.5-32B-Instruct, endpoint: http://gpu0:8000/v1}
  - {adapter: toduq.runners.open_source:VLLMClient, model_id: meta-llama/Llama-3.1-70B-Instruct, endpoint: http://gpu1:8000/v1}
  - {adapter: toduq.runners.open_source:VLLMClient, model_id: mistralai/Mistral-Small-Instruct-2409, endpoint: http://gpu2:8000/v1}
generation:
  split: round_robin     # round_robin (exactly even) | random | weighted
  # weights: [2, 1, 1]   # weighted: faster GPUs get a bigger share
  seed: 0
```

## 3. Validate the config first (`--dry-run`)

Before a big run, check every endpoint/key and preview the split — **no
generation, no SDK or GPU needed** (uses only the stdlib):

```bash
PYTHONPATH=src python -m toduq.cli generate --dry-run
```

It `GET`s `/v1/models` on each `generators:`/`judges:` endpoint (marking each
`OK`/`XX` with the reason), checks the env var for any closed API, and prints the
planned per-model split over the real unit count. Exit code is `0` only when every
check passes, so it drops straight into a CI/pre-flight gate. Example:

```
generators (3):
  [OK ] Qwen/Qwen2.5-32B-Instruct       open   reachable, serving Qwen/Qwen2.5-32B-Instruct
  [OK ] meta-llama/Llama-3.1-70B-Instr… open   reachable, serving meta-llama/Llama-3.1-70B-Instruct
  [XX ] mistralai/Mistral-Small-…       open   UNREACHABLE at http://gpu2:8000/v1/models (...)
Planned run: 104 units (38 violation + 66 control, control_multiplier=11) over 1 dialogue(s).
  generator split: {"Qwen/...": 35, "meta-llama/...": 35, "mistralai/...": 34}
```

## 4. Run

```bash
PYTHONPATH=src python -m toduq.cli generate --live --workers 8
```

- `--workers N` runs sites concurrently across the model endpoints (0/1 =
  sequential). Model **assignment** is done sequentially first, so the split
  stays deterministic and even regardless of `--workers`.
- The judge (Pass 4) stays a single model (`roles.judge`).

## Split strategies

| strategy | behavior |
| --- | --- |
| `round_robin` | exactly even — model *i* gets sites `i, i+len, i+2·len, …` |
| `weighted` | shares proportional to `weights` (e.g. bigger share to faster GPUs) |
| `random` | uniform random assignment |

One `ModelPool` is shared across the whole run, so the split is even over **all**
dialogues, not just within one.

## Provenance & the manifest

Every record's `provenance.generator_model` names the model that generated it.
`data/seed_v1/manifest.json` reports the split two ways:

```json
"generators":    {"Qwen/...": 11, "meta-llama/...": 11, "mistralai/...": 11},  // pool counters
"by_generator":  {"Qwen/...": 11, "meta-llama/...": 11, "mistralai/...": 11}   // from record provenance
```

so you can confirm the work was divided as intended and trace any sample back to
its model.

## Class balance (positive vs negative)

Each turn gets many *uncertainty* samples but only one *control* (answer) sample,
so the raw output is ~90 % positive (should-abstain). Generation balances it in
two steps so `records.jsonl` ships **1:1** by default:

1. **Grow the negative class** — `--control-multiplier` (default `auto`) emits N
   distinct paraphrase variants per turn. `auto` sizes N from the operator mix
   (11 uncertainty ops + 1 control → 11), so a live sampling model produces
   several diverse `answer` samples per turn to match its uncertainty samples. No
   positives are duplicated or discarded to grow negatives.
2. **Exact trim** — `balance()` then undersamples whichever class is still the
   majority down to `--balance-ratio × minority` (default `1.0` = exact 1:1),
   deterministically (seeded). The dropped rows are cheap paraphrases, never
   positives when positives are the minority.

```bash
PYTHONPATH=src python -m toduq.cli generate --live --workers 8            # 1:1 (default)
PYTHONPATH=src python -m toduq.cli generate --balance-ratio 2            # 2:1 positive:negative
PYTHONPATH=src python -m toduq.cli generate --no-balance                 # raw, skewed
```

Two files are written: **`records.jsonl`** (balanced — the shipped set) and
**`records_all.jsonl`** (the full accepted set, nothing lost). The manifest's
`class_balance` block reports `positive` / `negative` / `majority_class` /
`dropped` / `balanced_total`, and `control_multiplier` records the N used.

## Library use

```python
from toduq.runners import ModelPool
from toduq.runners.factory import build_client
from toduq.generate import generate_seed

clients = [build_client(s) for s in generator_specs]
pool = ModelPool(clients, strategy="round_robin")
generate_seed(pool=pool, workers=8, judge=my_judge, raw_dialogues=dialogues,
              balance=True, balance_ratio=1.0, control_multiplier="auto")
```
