# Infinite RL Execution Guide

This is the runnable walkthrough: start at the launch command, follow every call the code makes, and see exactly where your environment hooks fire. Read it like an execution trace—because that’s how it’s organized.

## Table of Contents
- Start Here: Launch Command
- Repository Map & Key Modules
- Full Execution Flow (Call Graph)
- Step‑By‑Step Execution
- Status So Far
- Configuration Quick Reference
- Verifiers Adapter & Dataset Streaming
- Environment Contract (What You Must Implement)
- Data Contract & Loader Behaviour
- Rollout Dynamics & Metrics
- Trainer Settings & Checkpointing
- Built‑In Example Environments
- Minimal Environment Skeleton
- Adapting External Reward Logic (e.g., Verifiers)
- Local Search Tooling (Optional Service)
- Validation and Smoke Tests
- Troubleshooting
- FAQ

---

## Start Here: Launch Command

You must provide the pieces left `null` in `config/grpo.yaml`. The canonical launch sequence lives in `launch_grpo.sh`; invoke the same overrides directly with:

```
torchrun \
  --nproc_per_node=8 \
  -m train.trainer.grpo \
  'data.train_data_path="verifiers:train?limit=32"' \
  'data.test_data_path="verifiers:eval?limit=32"' \
  data.prompts_per_rollout=8 \
  data.responses_per_prompt=4 \
  actor.model_name=Qwen/Qwen2-1.5B-Instruct \
  actor.max_length_per_device=512 \
  rollout.train_sampling_params.max_new_tokens=128 \
  rollout.env_path=environments/verifiers_adapter.py \
  rollout.max_turns=3 \
  adv.estimator=reinforce \
  adv.norm_var=true \
  trainer.project=GRPO \
  trainer.experiment_name=grpo-math-verifiers-integration \
  trainer.use_wandb=true \
  trainer.n_epochs=1 \
  trainer.test_freq=999999 \
  trainer.save_freq=null
```

Override rationale:
- `--nproc_per_node=8`: matches the launch script’s multi-GPU default; set to 1 for single-GPU smoke tests or bump to your local GPU count.
- `data.train_data_path="verifiers:train?limit=32"` / `data.test_data_path="verifiers:eval?limit=32"`: stream prompts from the Verifiers environment with a 32-example cap so you can validate the pipeline quickly; replace with larger limits or local JSONL paths for real runs. Quoting keeps Hydra from mangling the `?limit=` query.
- `data.prompts_per_rollout=8`: sets the rollout batch size high enough to leverage parallelism while staying under typical 24 GB GPU limits for the 1.5B model.
- `data.responses_per_prompt=4`: collects multiple samples per prompt so GRPO’s variance-normalised advantages have signal; must be ≥1 to avoid empty collations.
- `actor.model_name=Qwen/Qwen2-1.5B-Instruct`: chooses the base actor/critic weights; tokenizers follow `${actor.model_name}` automatically.
- `actor.max_length_per_device=512`: shortens packed sequence length to keep memory predictable across eight ranks; raise it once you verify stability.
- `rollout.train_sampling_params.max_new_tokens=128`: bounds completions to 128 tokens so SGLang cannot generate runaway sequences during debugging.
- `rollout.env_path=environments/verifiers_adapter.py`: binds the rollout loop to the Verifiers adapter, aligning environment hooks with streamed data and tool calls.
- `rollout.max_turns=3`: gives the model two tool exchange opportunities plus the final answer, mirroring the `math_python` Verifiers env defaults.
- `adv.estimator=reinforce` and `adv.norm_var=true`: enable GRPO’s REINFORCE-style estimator with variance normalisation—the intended “Dr. GRPO” configuration.
- `trainer.project=GRPO` / `trainer.experiment_name=grpo-math-verifiers-integration`: provide WandB bookkeeping and, via Hydra, expand `trainer.save_dir` to `ckpts/grpo-math-verifiers-integration`.
- `trainer.use_wandb=true`: keeps telemetry identical to the launch script; flip to `false` if you want an offline run.
- `trainer.n_epochs=1`: iterates once over the stateful dataloader—ideal for integration tests before longer campaigns.
- `trainer.test_freq=999999`: effectively disables evaluation because the modulus never hits within short runs (the field must remain a positive integer).
- `trainer.save_freq=null`: turns off periodic checkpointing while still saving the final model at exit.

Export a Hugging Face token if you hit rate limits: `export HF_TOKEN=...`. The launch script also primes HF caches and defaults `VERIFIERS_ENV_ID` to `math_python`; adjust those environment variables as needed before running.

---

## Repository Map & Key Modules

- `config/` — Hydra configs (currently just `grpo.yaml`).
- `train/` — GRPO implementation.
  - `trainer/` orchestrates the loop; `grpo.py` is the entrypoint.
  - `workers/` contain actor, critic, and rollout workers with TP/FSDP setup.
  - `datasets/` provides `RLDataset`, Verifiers streaming hooks, and tokenization utilities.
  - `utils/` handles communication, checkpoints, parallelism, logging, offloading, and math helpers.
- `environments/` — Built-in reward/tool adapters (`eq`, `orz`, `searchr1`, `verifiers_adapter`) and the optional `local_search_service` FastAPI.
- `stub/` — JSONL toy datasets for math/code/language/tool demos.
- `scripts/` — Launch helpers (empty today because the main script lives at root).
- `RL2/` — Upstream RLHF reference copy; not wired into the GRPO trainer but kept for comparison.
- `rubric/`, `planner/` — Mentioned in the README roadmap but not present yet (see “README Claims vs. Reality”).

---

## Full Execution Flow (Call Graph)

```
CLI
└─ python -m train.trainer.grpo --config-name grpo
   └─ hydra.main → train/trainer/grpo.py::main
      └─ GRPOTrainer(config)
         ├─ Trainer.__init__  (resolve config, optional wandb.init)
         ├─ get_dataloader(train/test) → RLDataset → load_dataset
         ├─ Actor(config.actor)              ┐
         │   └─ Worker base → prepare_device_mesh → load tokenizer/model
         ├─ (Critic if adv.estimator == "gae")  │  shared initialization path
         └─ Rollout(config.rollout)              │
             ├─ import env module (config.rollout.env_path)
             └─ start SGLang Engine(model_path)

         └─ train()
            ├─ load_ckpt → restore actor/(critic)/scheduler
            ├─ for batch in train_dataloader:
            │   ├─ tensor_dicts = Rollout.__call__(data_list, train=True, step)
            │   │   ├─ format prompt (tokenizer.apply_chat_template)
            │   │   ├─ Engine.async_generate
            │   │   ├─ env.interact(messages)  # optional tool replies
            │   │   └─ reward = env.reward_fn(messages, answer)
            │   │      tokenize_messages → states/actions/masks + final reward
            │   ├─ optional ref_actor.compute_logps / critic.compute_values
            │   ├─ compute_approx_kl / compute_advantages
            │   ├─ actor.update (and critic.update)
            │   ├─ save_ckpt
            │   └─ Rollout.update → push new weights into engine
            └─ save_model(actor)
```

Keep this mental map handy; the next section walks the same route with commentary and code pointers.

---

## Step‑By‑Step Execution

1. **Hydra boots the trainer** (`train/trainer/grpo.py:161-174`).
   - Resolves the config, prints it once on rank 0, and initialises Weights & Biases if enabled.
   - `trainer.save_dir` is validated immediately; leaving it empty raises before any GPU work starts.

2. **Distributed topology is carved out** (`train/utils/comm.py:7-23`, `train/workers/base.py:34-83`).
   - `initialize_global_process_group` sets up NCCL and pins each rank to its CUDA device.
   - Two meshes are built: `model_device_mesh` over (ddp, fsdp, tp) for sharded parameters and `device_mesh` over (dp, sp, tp) for rollout scattering.

3. **Tokenizers and models load with retry logic** (`train/workers/base.py:14-57`, `train/workers/actor.py:17-65`, `train/workers/critic.py:17-38`).
   - `_load_with_retry` pulls artifacts from Hugging Face, handling rate limits via exponential backoff.
   - `prepare_model_optimizer` applies tensor/sequence parallel transforms (`train/utils/parallelism.py`), wraps the model in FSDP HYBRID_SHARD, enables gradient checkpointing, and optionally offloads model/optimizer states.

4. **Environment module + inference engine come online** (`train/workers/rollout.py:25-118`).
   - `importlib` loads the file pointed to by `rollout.env_path` (default is `environments/verifiers_adapter.py`).
   - SGLang’s `Engine` initialises per rank (TP shard aware) on port `30000 + rank`, ready to answer async generation requests.

5. **Datasets stream prompts** (`train/trainer/grpo.py:49-64`, `train/datasets/base.py:9-68`, `train/datasets/rl.py:4-27`).
   - `RLDataset` wraps `load_dataset`. It supports local files, HF hub, and `verifiers:` streams; each batch is expanded `responses_per_prompt` times during collation.
   - `StatefulDataLoader` keeps shuffle state so checkpoints can restore mid-epoch.

6. **Rollouts build trajectories** (`train/workers/rollout.py:84-188`).
   - Prompts are rendered through `tokenizer.apply_chat_template` (configurable), the engine generates, and assistant messages are appended to the running conversation.
   - If `max_turns > 1`, new assistant turns are inspected for tool triggers; `interact` can emit tool messages (e.g., Verifiers tools or `<search>` replies) before the next generation turn.
   - Rewards are computed once per conversation; `tokenize_messages` converts the transcript to tensors and attaches the reward to the final action token.
   - Metrics (`response_length`, `length_clip_ratio`, `n_turns`, `rewards`, `trajectory_length`) are logged through `wandb`/tqdm.

7. **Policy/Value learning happens** (`train/trainer/grpo.py:107-158`, `train/utils/algorithms.py`).
   - Optional reference logprobs (`ref_actor`) and critic values feed into either REINFORCE (default) or GAE advantage estimators.
   - Actor updates apply PPO-style clipping, entropy bonuses, optional KL penalties, and respect per-update gradient offloading.
   - Critics (if present) run clipped value regression. Aggregations currently support `agg_mode=all_token_mean` only.

8. **Dynamic filtering & sync** (`train/workers/rollout.py:200-224`).
   - When `rollout.dynamic_filtering=true`, prompts whose sampled responses all share identical rewards are dropped to save compute; the drop ratio is logged as `dynamic_filtering_ratio`.
   - Latest actor weights are streamed into the running SGLang engine via `Rollout.update` so the next batch sees the fresh policy.

9. **Checkpointing & teardown** (`train/utils/checkpointing.py`).
   - `save_ckpt`/`load_ckpt` use `torch.distributed.checkpoint` to capture dataloader state, model shards, optimizer, and scheduler.
   - `save_model` dumps the final tokenizer + HF weights to `trainer.save_dir` (or `save_dir/latest` if periodic checkpoints run).
   - WandB sessions and process groups are closed cleanly even if errors bubble up.

---

## Status So Far

**Working today**
- Distributed GRPO trainer with multi-GPU / multi-process support (`torchrun`, NCCL meshes, FSDP HYBRID_SHARD).
- Tensor and sequence parallelism plans in `train/utils/parallelism.py` plus gradient checkpointing and optional offloading.
- Actor / critic / rollout worker stack with live SGLang integration, dynamic filtering, and Verifiers-aware tool loops.
- Multi-turn rollouts via `rollout.max_turns` + environment `interact`, including the default Verifiers adapter pathway.
- Checkpointing, resumption, and WandB logging wired through Hydra config (`trainer.save_dir`, `load_ckpt_from`, `trainer.use_wandb`).

**Planned / in flight**
- Rubric-driven prioritized replay scheduler and rubric runtime (`rubric/`, `planner/`) tracked in `high_level_plan.md` issues #5–#8.
- Single-signal `acc_ema` sampling, contamination detector gate, and multi-domain dataset routing (P0/P1 milestones).
- Zigzag ring attention kernels: wrapper exists, but importing the optimized kernels remains on the backlog.
- Broader continual-learning safeguards (upgrade-mode KL gates, adaptive replay, evaluation adapters) outlined in the roadmap but not yet implemented.

---

## Configuration Quick Reference

Most knobs live under `config/grpo.yaml` and can be overridden at launch.

- **`data.*`**
  - `train_data_path`, `test_data_path` accept local paths, Hugging Face datasets (`split@repo`), or `verifiers:split?limit=...` URIs.
  - `prompts_per_rollout` is the rollout batch size; default `null` forces you to choose based on memory.
  - `responses_per_prompt` controls augmentation per prompt; required to avoid zero-length batches.
- **`actor.*`**
  - `model_name` / `tokenizer_name` (defaults to model) must be set.
  - Parallelism sizes (`ddp`, `tp`, `sp`) factor into mesh assertions.
  - `use_liger_kernel` swaps in Liger kernels when TP is 1.
  - `gradient_checkpointing`, `offload_model`, `offload_optimizer` gate memory trade-offs.
  - `agg_mode` currently supports `all_token_mean` only; other modes raise `NotImplementedError`.
  - `kl.*`, `entropy.coef`, `update_per_rollout`, `freeze_steps` mirror PPO-style controls.
- **`rollout.*`**
  - `model_name` defaults to the actor model; same for tokenizer.
  - `train_sampling_params` / `test_sampling_params` forward to SGLang.
  - `max_turns` ≥2 enables tool turns.
  - `env_path` defaults to the Verifiers adapter; point to custom env modules as needed.
  - `dynamic_filtering` drops zero-variance reward groups to reduce wasted compute.
- **`ref_actor.*`** — Mirrors actor config when `actor.kl.coef > 0`.
- **`critic.*`** — Mirrors actor config, loaded as token classification head when `adv.estimator == "gae"`.
- **`adv.*`**
  - `estimator` is `reinforce` (default) or `gae`.
  - `global_norm` toggles whether the REINFORCE baseline is shared across prompts.
  - `norm_var` matches GRPO’s variance normalization.
- **`trainer.*`**
  - `project`, `experiment_name`, `use_wandb` feed logging.
  - `save_dir` must be non-empty (set automatically if `experiment_name` is provided).
  - `n_epochs`, `save_freq`, `test_freq` control loop cadence. Remember: `test_freq` must be a positive integer; there is no sentinel for “disable”.
  - `load_ckpt_from` resolves checkpoints (`path` or `latest`).

---

## Verifiers Adapter & Dataset Streaming

`environments/verifiers_adapter.py` bridges Infinite’s rollout contract with the open-source Verifiers suite:
- Loads a Verifiers environment once (default `math_python`) using `VERIFIERS_ENV_ID`/`VERIFIERS_ENV_ARGS` env vars.
- Spins up a background asyncio loop so rollout’s synchronous code can await Verifiers coroutines.
- Implements session bookkeeping so multi-turn Verifiers envs keep state per conversation.
- Normalises tool messages (role/content/tool_call_id) before handing them back to the trainer.
- Delegates scoring to the env’s `rubric.score_rollout`, returning a float reward.

Dataset streaming piggybacks on the same adapter: set `data.train_data_path=verifiers:train?limit=128` (and analogous for eval) to iterate the Verifiers environment dataset directly without exporting JSONL. The loader caches rows per split; `limit=` truncates for smoke tests.

If you are not using Verifiers, point `rollout.env_path` elsewhere and swap the dataset paths to local files. The rest of the pipeline is unchanged.

---

## Environment Contract (What You Must Implement)

Create a Python module and point `config/grpo.yaml → rollout.env_path` at it. The module must define:

- `interact(messages: list[dict]) -> list[dict]`
  - Purpose: optionally inject tool outputs between model turns; return `[]` if unused.
  - Input: the running conversation `[ {"role": ..., "content": ...}, ... ]`.
  - Output: zero or more tool messages, e.g., `{ "role": "tool", "content": "..." }`.

- `reward_fn(messages: list[dict], answer: str | list[str]) -> float`
  - Purpose: score the final assistant turn against ground truth (or rubric).
  - Input: full conversation and the `answer` field from the dataset.
  - Output: scalar reward (`float`/`bool`). Most envs attach reward on the last token.

Notes:
- Multi-turn: controlled by `rollout.max_turns`. If > 1, the model can produce tool triggers, your `interact` can respond, then generation continues.
- Keep `reward_fn` stateless when possible. For stateful logic follow the Verifiers adapter pattern: cache per-conversation data keyed by `id(messages)` and clean up once scoring finishes.

---

## Data Contract & Loader Behaviour

Each example must provide a chat history and the supervised answer for scoring.

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "answer": "..."
}
```

- Storage: JSON/JSONL/CSV/Parquet/Arrow are supported. For JSONL, use one record per line.
- Loader: `train/datasets/base.py::load_dataset` resolves based on suffix, Hugging Face repos (`split@name`), or `verifiers:` URIs.
- Repetition: `responses_per_prompt` controls how many independent samples per input are drawn in training.
- Tokenization: `tokenize_messages` asserts that adding a new message never mutates earlier tokenization. If your tokenizer breaks that (e.g., Qwen3 multi-turn), swap to an increasing tokenizer such as `Chenmien/Qwen3-Increasing-Tokenizer`.
- Checkpoints: dataloader RNG/state is checkpointed so resumed runs do not repeat prompts.

---

## Rollout Dynamics & Metrics

- The rollout worker runs on the DP×TP mesh; only TP rank 0 spins up SGLang per process.
- `train_sampling_params` / `test_sampling_params` go straight into `Engine.async_generate`.
- Responses longer than `max_new_tokens` are truncated manually because current SGLang releases can overshoot limits.
- Metrics gathered per step: `response_length`, `length_clip_ratio`, `n_turns`, `rewards`, `trajectory_length`. They are reduced across ranks and logged via `gather_and_log`.
- Dynamic filtering (when enabled) groups tensors per prompt and removes groups whose sampled rewards are identical; the drop ratio is surfaced as `dynamic_filtering_ratio` so you can tune `responses_per_prompt`.
- After each actor update, `Rollout.update` pushes weights into the running Engine without restarting it, keeping generation in sync with training.

---

## Trainer Settings & Checkpointing

- `Trainer.prepare_scheduler` computes warmup/total steps from `n_epochs × len(train_dataloader) × update_per_rollout`.
- `StatefulDataLoader` state is saved alongside model weights, so interrupted runs resume mid-epoch.
- `load_ckpt` / `save_ckpt` rely on `torch.distributed.checkpoint`; ensure `trainer.save_dir` exists on all ranks.
- `save_model` writes the full HF-format model + tokenizer; for reward models set `rm=True` to convert to sequence classification.
- Model/optimizer offloading is opt-in via config; decorators in `train/utils/offloading.py` automatically move tensors between CPU/GPU around forward/backward passes.
- WandB logging is gated by `trainer.use_wandb`. If disabled, a no-op logger is injected so existing calls succeed locally.

---

## Built‑In Example Environments

- **Verifiers Adapter (`environments/verifiers_adapter.py`)**
  - Purpose: bridge to Verifiers multi-turn rubric envs with optional tools.
  - Needs: `verifiers` package + selected environment dependencies.
  - Produces: float rewards, optional tool messages.

- **Equality Match (`environments/eq.py`)**
  - Purpose: normalise and compare the model’s final answer (optionally inside `<answer>...</answer>` tags) to ground truth.
  - Needs: none beyond regex/normalisation.
  - Produces: 0/1 reward.

- **Math Verify (`environments/orz.py`)**
  - Purpose: structured math checking via `math_verify.parse/verify`.
  - Needs: `math_python` dependency pulled in via `requirements.txt`.
  - Produces: boolean correctness reward.

- **Retrieval + QA (`environments/searchr1.py`)**
  - Purpose: allow the model to issue `<search>query</search>`; env calls a local search service and returns a tool message.
  - Needs: the FastAPI service in `environments/local_search_service.py` running.
  - Produces: tool messages during rollout and normalised exact-match reward.

---

## Minimal Environment Skeleton

```python
# environments/verifiers_math.py
import re, string

def interact(messages):
    # No tools for this env; keep it simple.
    return []


def _extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text.strip()


def _normalize(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def reward_fn(messages, answer):
    if not messages:
        return 0.0
    pred = _normalize(_extract_answer(messages[-1]["content"]))
    answers = [answer] if isinstance(answer, str) else list(answer)
    return float(any(pred == _normalize(a) for a in answers))
```

Wire it up by setting in your run command or Hydra overrides:

```
rollout.env_path=environments/verifiers_math.py \
rollout.model_name=<your_model_path_or_name> \
data.train_data_path=stub/data/math/train.jsonl \
data.test_data_path=stub/data/math/test.jsonl \
data.prompts_per_rollout=4 data.responses_per_prompt=2
```

---

## Adapting External Reward Logic (e.g., Verifiers)

- Lift the reward logic you like (e.g., rubric scoring, programmatic verifiers) into a single module that exposes the contract above.
- Keep the dataset format unchanged; only your `reward_fn` needs to normalise the final assistant message the same way.
- If the external system expects tools or additional context, implement that inside `interact` and marshal any session state locally (see Verifiers adapter for a template).
- For Verifiers specifically, expose new environments by exporting `VERIFIERS_ENV_ID`/`VERIFIERS_ENV_ARGS` before launch; no trainer changes required.
- Start with a single-turn env, then expand to tools/multi-turn if needed.

---

## Local Search Tooling (Optional Service)

To support retrieval like `searchr1`, run the local search API:

```bash
python environments/local_search_service.py \
  --model_name <embedding_model_in_engine> \
  --index_path </path/to/faiss.index> \
  --corpus_path </path/to/corpus.jsonl> \
  --top_k 5
```

- Your env will POST `{ "query": "..." }` to `http://localhost:8000/search` and receive concatenated passages.
- Ensure the SGLang engine is serving embeddings at `http://localhost:30000/v1/embeddings` (ports are rank-offset).

---

## Validation and Smoke Tests

1) Single‑GPU quick run on the math stub:

```bash
python -m train.trainer.grpo --config-name grpo \
  rollout.env_path=environments/eq.py \
  rollout.model_name=<your_model> \
  data.train_data_path=stub/data/math/train.jsonl \
  data.test_data_path=stub/data/math/test.jsonl \
  data.prompts_per_rollout=2 data.responses_per_prompt=2 \
  trainer.n_epochs=1 trainer.test_freq=999999 trainer.save_dir=ckpts/math_stub
```

2) Verify logs:
- TQDM prints one sample’s messages; W&B (if enabled) captures `rewards/train`, `trajectory_length/train`, `dynamic_filtering_ratio`, etc.
- If rewards are always 0, confirm `<answer>` normalisation matches your dataset or relax normalisation.
- If you hit `TypeError: unsupported operand type(s) for %: 'int' and 'NoneType'`, set `trainer.test_freq` to a positive integer.

---

## Troubleshooting

- Engine OOM or slow: reduce `rollout.train_sampling_params.max_new_tokens` or `gpu_memory_utilization`.
- No tool messages appear: ensure your `interact` returns a list; with `max_turns=1`, tools never run.
- Rewards always zero: check normalisation/extraction, confirm the assistant outputs the expected format, or disable dynamic filtering.
- Verifiers dataset returns nothing: verify `VERIFIERS_ENV_ID` is installed and the chosen split exists; use `?limit=` for quick checks.
- Tokenizer mismatch: if prompts shrink/grow unexpectedly, see the assertion in `tokenize_messages` regarding increasing tokenizers.
- Checkpoint load fails: ensure `trainer.save_dir` points to a shared filesystem and `load_ckpt_from` references an existing directory (or `latest`).

---

## FAQ

- **Q: Can I keep my env stateful?**
  - A: Prefer pure functions; if you must hold state (e.g., a cache or rubric session), keep it module-local and clean up after scoring, mirroring `verifiers_adapter`.
- **Q: How do I stream dataset rows directly from Verifiers?**
  - A: Set `data.train_data_path=verifiers:train?limit=...` (and the eval analog). Optionally export `VERIFIERS_ENV_ARGS` as JSON to forward environment-specific parameters.
- **Q: Where does the reward get attached?**
  - A: On the final action token only; intermediate tokens receive zero reward by construction.
- **Q: What if I don’t want WandB?**
  - A: Launch with `trainer.use_wandb=false`; logging calls will no-op locally.
- **Q: Can I disable evaluation entirely?**
  - A: There is no sentinel; set `trainer.test_freq` to a very large integer so the modulus condition never triggers.

---
