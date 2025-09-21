# Infinite RL Environments — Integration Guide

This document explains how “environments” plug into Infinite’s GRPO loop, what each stage does, and how to add or adapt a custom environment module. It is written for engineers wiring new reward functions, tool calls, or datasets into the existing training pipeline.

## Table of Contents
- Overview and Purpose
- End‑to‑End Flow (Call Graph)
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

## Overview and Purpose
- Goal: optimize a policy to produce better answers using scalar rewards computed by a simple “environment” module you control.
- The environment module is intentionally lightweight: it can be a single `.py` file exposing two functions. This keeps rewards and tool logic decoupled from the trainer/actor.
- Who consumes it: the Rollout worker imports your module, calls it to optionally inject tool outputs, and asks it to score the final conversation. Those scores drive GRPO updates to the actor (and optionally a critic if you use GAE).

---

## End‑to‑End Flow (Call Graph)

```
CLI
└─ python -m train.trainer.grpo --config-name grpo
   └─ GRPOTrainer.main()  (train/trainer/grpo.py)
      ├─ init dataloaders  → RLDataset(load_dataset(...))
      ├─ init workers      → Actor, (Critic), Rollout
      │   └─ Rollout.__init__
      │       ├─ import env from config.rollout.env_path
      │       └─ start SGLang Engine (model_path=config.rollout.model_name)
      └─ train() loop
          ├─ for batch in dataloader:
          │   └─ tensor_dicts = Rollout.__call__(data_list, train=True, step)
          │       ├─ format prompt (tokenizer.apply_chat_template)
          │       ├─ LLM generate (Engine.async_generate)
          │       ├─ env.interact(messages)    # optional tools
          │       ├─ reward = env.reward_fn(messages, answer)
          │       └─ tokenize_messages(...) → states/actions/masks + final reward
          ├─ compute_approx_kl / compute_advantages
          ├─ actor.update(...) [+ critic.update(...)]
          ├─ save_ckpt(...)
          └─ Rollout.update(...)  # hot‑swap weights into engine
```

What each stage needs and produces:
- Dataloading
  - Needs: dataset files with `messages` and `answer` fields.
  - Produces: lists of examples for rollouts; optionally duplicated per prompt.
- Rollout Generation
  - Needs: tokenizer, model, and your env’s `interact`/`reward_fn`.
  - Produces: trajectories, metrics, and a scalar reward per sample.
- Advantage + KL
  - Needs: tensorized trajectories, chosen estimator (reinforce/gae), KL settings.
  - Produces: advantages and, if configured, KL‑adjusted rewards.
- Updates + Sync
  - Needs: optimizer/scheduler, actor (and critic), state dicts.
  - Produces: improved policy, checkpoints, and refreshed engine weights.

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
- Lift the reward logic you like (e.g., box‑answer parsing or rubric checks) into a single function inside a new env module.
- Keep our dataset format unchanged; only your `reward_fn` needs to normalize the final assistant message the same way.
- If their logic expects “tools” or additional context, implement that in `interact` and/or pre/post‑processing helpers inside the module.
- Start with a single‑turn env (like the skeleton above), then expand to tools/multi‑turn if needed.

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

