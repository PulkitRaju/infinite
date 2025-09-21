# Infinite RL Execution Guide

This is the runnable walkthrough: start at the launch command, follow every call the code makes, and see exactly where your environment hooks fire. Read it like an execution trace—because that’s how it’s organized.

## Table of Contents
- Start Here: Launch Command
- Full Execution Flow (Call Graph)
- Step‑By‑Step Execution
- README Claims vs. Reality
- Environment Contract (What You Must Implement)
- Data Contract (What Training Expects)
- Key Configuration (Hydra/YAML)
- Built‑In Example Environments
- Minimal Environment Skeleton
- Adapting External Reward Logic (e.g., Verifiers)
- Local Search Tooling (Optional Service)
- Validation and Smoke Tests
- Troubleshooting
- FAQ

---

## Start Here: Launch Command

You must provide the pieces left `null` in `config/grpo.yaml`. This is the minimal single‑GPU invocation (swap in your own paths/models/env):

```
python -m train.trainer.grpo --config-name grpo \
  data.train_data_path=stub/data/math/train.jsonl \
  data.test_data_path=stub/data/math/test.jsonl \
  data.prompts_per_rollout=4 data.responses_per_prompt=2 \
  actor.model_name=Qwen/Qwen2-1.5B-Instruct \
  rollout.train_sampling_params.max_new_tokens=128 \
  rollout.env_path=environments/eq.py \
  trainer.use_wandb=false
```

Multi‑GPU just wraps the same overrides in `torchrun --nproc_per_node=N`. Without these overrides the run will error before rollouts begin.

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
   - Resolves the config, broadcasts it, and initializes Weights & Biases if enabled.
   - Purpose: lock in hyperparameters and logging before any distributed work.

2. **Distributed topology is carved out** (`train/utils/comm.py:7-23`, `train/workers/base.py:34-83`).
   - `initialize_global_process_group` sets up NCCL and pins each rank to its CUDA device.
   - Two meshes are built: `model_device_mesh` over (ddp, fsdp, tp) for sharded parameters and `device_mesh` over (dp, sp, tp) for rollout scattering.

3. **Tokenizers and models load with retry logic** (`train/workers/base.py:14-57`, `train/workers/actor.py:17-46`, `train/workers/critic.py:17-38`).
   - `_load_with_retry` pulls artifacts from Hugging Face, handling rate limits via exponential backoff.
   - `prepare_model_optimizer` applies tensor/sequence parallel transforms (`train/utils/parallelism.py`) and wraps the result in FSDP with BF16 mixed precision. Optimizers live on CPU unless actively stepping to support offloading.

4. **Environment module + inference engine come online** (`train/workers/rollout.py:25-51,84-145`).
   - `importlib` loads the file pointed to by `rollout.env_path`, expecting `interact`/`reward_fn`.
   - SGLang’s `Engine` initializes per rank (TP shard aware), ready to answer generation requests.

5. **Datasets stream prompts** (`train/trainer/grpo.py:49-64`, `train/datasets/rl.py:4-29`).
   - `RLDataset` reads JSON/JSONL/CSV/Parquet via `datasets.load_dataset` and expands each record `responses_per_prompt` times on the fly for exploration.

6. **Rollouts build trajectories** (`train/workers/rollout.py:92-160`).
   - Prompts are rendered through `tokenizer.apply_chat_template` (configurable), the engine generates, and assistant messages are appended to the running conversation.
   - If `max_turns > 1`, the newest assistant response is scanned for tool triggers; `interact` can emit tool messages (e.g., search results) that get spliced in before the next generation turn.
   - When the loop exits (max turns reached or no tool call), `reward_fn` scores the full transcript and `tokenize_messages` converts it to tensors with the reward attached to the final timestep only.

7. **Policy/Value learning happens** (`train/trainer/grpo.py:107-158`).
   - Optional reference logprobs (`ref_actor`) and critic values feed into either REINFORCE or GAE advantage estimators.
   - Actor updates apply PPO‑style clipping, entropy bonuses, and optional KL penalties before stepping the optimizer/scheduler. Critics (if present) perform clipped MSE updates against returns.

8. **State sync + persistence** (`train/utils/checkpointing.py`, `train/workers/rollout.py:162-220`).
   - Checkpoints capture model/optimizer/scheduler state every iteration; `Rollout.update` broadcasts fresh weights into the SGLang engine so the next rollout queries the latest policy.

This is the loop you are extending when you add a new environment or dataset.

---

## README Claims vs. Reality

- **Distributed training across multiple GPUs** — Implemented (`torchrun`, meshes, NCCL).
- **FSDP** — Implemented via `torch.distributed.fsdp` with HYBRID_SHARD and BF16.
- **Tensor + sequence parallelism** — Implemented using DTensor plans in `train/utils/parallelism.py`.
- **Zigzag ring attention** — Partial. Wrapper enforces shapes but the actual model uses `flash_attention_2`; zigzag scheduling is not imported.
- **Worker architecture (actor/critic/rollout)** — Implemented and actively used.
- **Multi‑turn rollouts** — Implemented via `rollout.max_turns` + env `interact`.
- **Checkpointing** — Implemented (`train/utils/checkpointing.py`).
- **Environment rewards (eq/orz/searchr1)** — Implemented, though `searchr1` assumes the local FastAPI search service is running.
- **Hydra configuration** — Implemented; all knobs live under `config/grpo.yaml`.
- **Rubric‑driven prioritized replay** — UNIMPLEMENTED. No `rubric/` runtime, scheduler still on roadmap (`high_level_plan.md`).
- **Planner / prioritized replay scheduler** — UNIMPLEMENTED. No `planner/` module wired in.
- **Adaptive replay / continual learning claims** — UNIMPLEMENTED beyond the plain GRPO loop.

Keep these gaps in mind when planning future work; the rest of this document focuses on what exists today.

---

## Environment Contract (What You Must Implement)
Create a Python module and point `config/grpo.yaml → rollout.env_path` at it. The module must define:

- `interact(messages: list[dict]) -> list[dict]`
  - Purpose: optionally inject tool outputs between model turns; return `[]` if unused.
  - Input: the running conversation `[ {"role": ..., "content": ...}, ... ]`.
  - Output: zero or more tool messages, e.g., `{ "role": "tool", "content": "..." }`.

- `reward_fn(messages: list[dict], answer: str | list[str]) -> float`
  - Purpose: score the final assistant turn against ground truth.
  - Input: full conversation and the `answer` field from the dataset.
  - Output: scalar reward (`float`/`bool`). Most envs attach reward on the last token.

Notes:
- Multi‑turn: controlled by `rollout.max_turns`. If > 1, the model can produce tool triggers, your `interact` can respond, then generation continues.
- No global state is required; if your env needs services (e.g., search), it can call them inline (see “Local Search Tooling”).

---

## Data Contract (What Training Expects)
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

- Storage: JSON/JSONL/CSV/Parquet/Arrow. For JSONL, use one record per line.
- Loader: `train/datasets/base.py::load_dataset` resolves format based on file extension.
- Repetition: `responses_per_prompt` controls how many samples per prompt are drawn in training.

---

## Key Configuration (Hydra/YAML)
Relevant fields in `config/grpo.yaml`:

- `data.train_data_path`, `data.test_data_path` — dataset file paths.
- `data.prompts_per_rollout` — batch size for rollouts.
- `data.responses_per_prompt` — number of samples per input.
- `rollout.model_name` — path/name passed to SGLang Engine.
- `rollout.tokenizer_name` — tokenizer used for prompts and token accounting.
- `rollout.max_turns` — 1 for single‑turn; >1 enables tool rounds.
- `rollout.env_path` — absolute/relative path to your env module `.py` file.
- `actor.kl.*` — configure KL penalty/estimator if desired.
- `adv.estimator` — `reinforce` (default) or `gae` (with critic).

---

## Built‑In Example Environments
- Equality Match (`environments/eq.py`)
  - Purpose: normalize and compare the model’s final answer (optionally inside `<answer>...</answer>` tags) to ground truth.
  - Needs: none beyond regex/normalization.
  - Produces: 0/1 reward.

- Math Verify (`environments/orz.py`)
  - Purpose: structured math checking via `math_verify.parse/verify`.
  - Needs: `math_verify` package installed and importable.
  - Produces: boolean correctness reward.

- Retrieval + QA (`environments/searchr1.py`)
  - Purpose: allow the model to issue `<search>query</search>`; env calls a local search service and returns a tool message.
  - Needs: the FastAPI service in `environments/local_search_service.py` running.
  - Produces: tool messages during rollout and normalized exact‑match reward.

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
- Lift the reward logic you like (e.g., box-answer parsing or rubric checks) into a single function inside a new env module.
- Keep our dataset format unchanged; only your `reward_fn` needs to normalize the final assistant message the same way.
- If their logic expects “tools” or additional context, implement that in `interact` and/or pre/post-processing helpers inside the module.
- Start with a single-turn env (like the skeleton above), then expand to tools/multi-turn if needed.

### Verifiers Adapter (Bring Your Own Env)
- Path: `environments/verifiers_adapter.py`.
- Configure which Verifiers package to load with env vars:
  - `VERIFIERS_ENV_ID` (defaults to `math_python`).
  - `VERIFIERS_ENV_ARGS` (JSON dict forwarded to `vf.load_environment`).
- Install dependencies: add `verifiers` to your env (`pip install -r requirements.txt`).
- Default install also pulls `math-python` from the Verifiers repo so the stock adapter
  can resolve `VERIFIERS_ENV_ID=math_python` immediately. Override the env var if you
  ship a different package.
- Hydra: set `rollout.env_path=environments/verifiers_adapter.py`; mirror the Verifiers env’s turn budget with `rollout.max_turns`.
- Dataset: continue supplying `messages`/`answer`; optional `info`/`task` can be plumbed by extending the dataloader.
- Runtime: the adapter keeps per-conversation state, calls Verifiers’ rubric for reward, and forwards tool traffic returned by the env. No changes to `train/workers/rollout.py` are required.
- Smoke test: `python -m train.trainer.grpo --config-name grpo rollout.env_path=environments/verifiers_adapter.py rollout.max_turns=<vf_max_turns>` using the stub dataset before scaling up.
- Dataset shortcut: set `data.train_data_path=verifiers:train` (and `test_data_path=verifiers:eval`) to pull prompts directly from the loaded Verifiers environment without exporting JSONL. Append `?limit=128` to sample.

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

- Your env will POST `{"query": "..."}` to `http://localhost:8000/search` and receive concatenated passages.
- Ensure the SGLang engine is serving embeddings at `http://localhost:30000/v1/embeddings` (ports are rank‑offset).

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
  trainer.n_epochs=1 trainer.test_freq=999999
```

2) Verify logs:
- TQDM prints one sample’s messages; W&B captures `rewards/train`, `trajectory_length/train`, etc.
- If rewards are always 0, confirm `<answer>` normalization matches your dataset or relax normalization.

---

## Troubleshooting
- Engine OOM or slow: reduce `rollout.train_sampling_params.max_new_tokens` or `gpu_memory_utilization`.
- No tool messages appear: ensure your `interact` returns a list; with `max_turns=1`, tools never run.
- Rewards always zero: check normalization/extraction and whether the assistant actually outputs `<answer>`.
- External service 404: confirm the FastAPI search is running and reachable.
- Tokenizer mismatch: if prompts shrink/grow unexpectedly, see the assertion in `tokenize_messages` regarding increasing tokenizers.

---

## FAQ
- Q: Can I keep my env stateful?
  - A: Prefer pure functions; if you must hold state (e.g., a cache), keep it module‑local and idempotent across calls.
- Q: How do I add periodic evaluation?
  - A: Set `trainer.test_freq` to a positive integer; the trainer will run test rollouts and log metrics.
- Q: Where does the reward get attached?
  - A: On the final action token only; intermediate tokens receive zero reward by construction.

---

Happy hacking! If you add a new env, consider placing it under `environments/` and documenting any external dependencies inline at the top of the file.
