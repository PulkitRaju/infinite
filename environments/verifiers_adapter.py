"""Adapter between Infinite RL rollout contract and Verifiers environments.

This module exposes the `interact` and `reward_fn` call signatures that
`train/workers/rollout.py` expects (see docs.md "Environment Contract").
Internally we load a Verifiers environment once and bridge its async rubric
and tool protocol onto the synchronous trainer loop.

Configuration:
- VERIFIERS_ENV_ID: optional env var naming the Verifiers environment package.
  Defaults to "math_python".
- VERIFIERS_ENV_ARGS: optional JSON object with keyword args forwarded to
  `vf.load_environment` (e.g. dataset overrides, tool configs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import verifiers as vf

logger = logging.getLogger(__name__)

_ENV_ID = os.getenv("VERIFIERS_ENV_ID", "math_python")
_ENV_ARGS: Dict[str, Any] = {}
if env_args_raw := os.getenv("VERIFIERS_ENV_ARGS"):
    try:
        parsed = json.loads(env_args_raw)
        if not isinstance(parsed, dict):
            raise TypeError("VERIFIERS_ENV_ARGS must decode to a dict")
        _ENV_ARGS = parsed
    except Exception as exc:  # noqa: BLE001 - surface config errors clearly
        raise RuntimeError(
            "Failed to parse VERIFIERS_ENV_ARGS as JSON object"
        ) from exc

logger.info("Loading Verifiers environment %s with args %s", _ENV_ID, _ENV_ARGS)
_VF_ENV = vf.load_environment(_ENV_ID, **_ENV_ARGS)
logger.info("Loaded Verifiers environment %s", _VF_ENV.__class__.__name__)

# Spin up a dedicated asyncio event loop, so we can schedule Verifiers coroutines
# from the trainer's synchronous rollout loop without blocking.
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, name="verifiers-loop", daemon=True)
_thread.start()


def _await(coro: "asyncio.Future[Any] | asyncio.Awaitable[Any]") -> Any:
    """Synchronously wait for a coroutine on the background event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


class _Session:
    """Per-conversation cache that mirrors Verifiers state expectations."""

    __slots__ = ("prompt", "state")

    def __init__(self, initial_prompt: List[Dict[str, Any]]) -> None:
        self.prompt = initial_prompt
        # seed minimal state structure expected by MultiTurnEnv.
        info: Dict[str, Any] = {}
        if getattr(_VF_ENV, "oai_tools", None):
            info["oai_tools"] = _VF_ENV.oai_tools
        self.state: Dict[str, Any] = {
            "id": id(self),
            "prompt": deepcopy(initial_prompt),
            "completion": [],
            "answer": "",
            "task": "default",
            "info": info,
            "responses": [],
            "turn": 0,
            "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
        }
        # allow envs to mutate state during setup
        self.state = _await(_VF_ENV.setup_state(self.state))


_sessions: Dict[int, _Session] = {}


def _get_session(messages: List[Dict[str, Any]]) -> _Session:
    """Fetch or initialise session for the mutable `messages` object."""
    key = id(messages)
    session = _sessions.get(key)
    if session is None:
        # First assistant turn has already been appended when we are called.
        base_prompt = deepcopy(messages[:-1]) if messages else []
        session = _Session(base_prompt)
        _sessions[key] = session
    return session


def _update_session_on_assistant(session: _Session, messages: List[Dict[str, Any]]) -> None:
    """Record the latest assistant turn into the cached Verifiers state."""
    session.state["completion"] = deepcopy(messages[len(session.prompt) :])
    session.state["responses"].append({"message": deepcopy(messages[-1])})
    session.state["turn"] = session.state.get("turn", 0) + 1


def _cleanup_session(messages: List[Dict[str, Any]]) -> None:
    _sessions.pop(id(messages), None)


def interact(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Produce tool responses after each assistant generation (may be empty)."""
    if not messages:
        return []

    session = _get_session(messages)
    _update_session_on_assistant(session, messages)

    should_stop = _await(_VF_ENV.is_completed(messages, session.state))
    if should_stop:
        return []

    env_messages, new_state = _await(_VF_ENV.env_response(messages, session.state))
    session.state = new_state

    if not env_messages:
        return []

    # Normalise output to the trainer's expected schema.
    normalised: List[Dict[str, Any]] = []
    for msg in env_messages:
        if not isinstance(msg, dict):
            raise TypeError("Verifiers env_response must return dict messages")
        allowed = {k: msg[k] for k in ("role", "content") if k in msg}
        if "tool_call_id" in msg:
            allowed["tool_call_id"] = msg["tool_call_id"]
        normalised.append(allowed)
    return normalised


def reward_fn(messages: List[Dict[str, Any]], answer: Any) -> float:
    """Score the final conversation using the environment's rubric."""
    if not messages:
        return 0.0

    session = _get_session(messages)
    session.state["completion"] = deepcopy(messages[len(session.prompt) :])

    # Normalise answer to a scalar string for rubrics (common Verifiers pattern).
    answer_str = answer if isinstance(answer, str) else (answer[0] if answer else "")
    session.state["answer"] = answer_str

    try:
        score = _await(
            _VF_ENV.rubric.score_rollout(
                prompt=deepcopy(session.prompt),
                completion=deepcopy(messages[len(session.prompt) :]),
                answer=answer_str,
                state=session.state,
            )
        )
        reward = float(score.reward)
    finally:
        _cleanup_session(messages)

    return reward


def get_environment():
    """Expose the loaded Verifiers environment (singleton)."""
    return _VF_ENV


def _resolve_split(split: str) -> str:
    split = (split or "train").lower()
    if split in {"train", "training"}:
        return "dataset"
    if split in {"eval", "evaluation", "test", "validation", "dev"}:
        return "eval_dataset"
    raise ValueError(f"Unsupported Verifiers dataset split: {split}")


def _normalize_answer(row: Dict[str, Any]) -> Any:
    if "answer" in row and row["answer"] is not None:
        return row["answer"]
    info = row.get("info")
    if isinstance(info, dict) and "answer" in info:
        return info["answer"]
    return ""


@lru_cache(maxsize=None)
def _cached_records(split: str) -> tuple:
    attr_name = _resolve_split(split)
    dataset = getattr(_VF_ENV, attr_name, None)
    if dataset is None:
        if attr_name == "eval_dataset":
            dataset = getattr(_VF_ENV, "dataset", None)
        if dataset is None:
            raise ValueError(
                f"Verifiers environment '{_ENV_ID}' does not provide a dataset for split '{split}'"
            )
    records: List[Dict[str, Any]] = []
    for row in dataset:
        prompt = row.get("prompt")
        if prompt is None:
            raise KeyError("Verifiers dataset row missing 'prompt' field")
        record = {
            "messages": prompt,
            "answer": _normalize_answer(row),
        }
        if "info" in row:
            record["info"] = row["info"]
        records.append(record)
    return tuple(records)


def iter_dataset_records(split: str = "train") -> Iterable[Dict[str, Any]]:
    """Yield dataset examples (messages/answer[/info]) from the Verifiers env."""
    for record in _cached_records(split):
        yield {
            "messages": deepcopy(record["messages"]),
            "answer": record["answer"],
            **({"info": record["info"]} if "info" in record else {}),
        }
