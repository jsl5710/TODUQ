# Generation on a server — multi-model, even split

For a large generation run you can split the Pass-3 paraphrasing **evenly across
several models** (Qwen, Llama 3.1, Mistral, …), each served on its own
OpenAI-compatible endpoint (vLLM / Ollama / TGI). Work is divided by a
`ModelPool`, each sample records the model that produced it, and sites can run in
parallel across endpoints.

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

## 3. Run

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

## Library use

```python
from toduq.runners import ModelPool
from toduq.runners.factory import build_client
from toduq.generate import generate_seed

clients = [build_client(s) for s in generator_specs]
pool = ModelPool(clients, strategy="round_robin")
generate_seed(pool=pool, workers=8, judge=my_judge, raw_dialogues=dialogues)
```
