import asyncio
import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np


_BATCH_FIELDS = {"batch_images", "instructions", "state"}


@dataclasses.dataclass
class _PendingRequest:
    payload: dict[str, Any]
    future: asyncio.Future


def _as_batch_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"`{key}` must be a list, got {type(value).__name__}")
    return value


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    return left == right


def merge_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge single-client inference payloads into one model batch."""
    if not payloads:
        raise ValueError("Cannot merge an empty payload list")

    merged: dict[str, Any] = {}
    all_keys = set().union(*(payload.keys() for payload in payloads))
    for key in all_keys:
        if key in _BATCH_FIELDS:
            values: list[Any] = []
            present_count = 0
            for payload in payloads:
                if key in payload:
                    present_count += 1
                    values.extend(_as_batch_list(payload, key))
            if present_count not in (0, len(payloads)):
                raise ValueError(f"Batch field `{key}` must be present in every payload or none")
            if present_count:
                merged[key] = values
            continue

        if any(key not in payload for payload in payloads):
            raise ValueError(f"Non-batch field `{key}` must be present in every payload or none")
        first_value = payloads[0][key]
        for payload in payloads[1:]:
            if not _values_match(first_value, payload[key]):
                raise ValueError(f"Non-batch field `{key}` differs across queued requests")
        merged[key] = first_value

    return merged


def _split_value(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, np.ndarray) and value.shape[:1] == (batch_size,):
        return value[index : index + 1]
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index]]
    if isinstance(value, tuple) and len(value) == batch_size:
        return (value[index],)
    return value


def split_batched_output(output: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    """Split one batched model output into per-request response payloads."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    responses: list[dict[str, Any]] = []
    for index in range(batch_size):
        responses.append({key: _split_value(value, index, batch_size) for key, value in output.items()})
    return responses


def merge_outputs(outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-sample model outputs back into a multi-sample client response."""
    if not outputs:
        raise ValueError("Cannot merge an empty output list")
    if len(outputs) == 1:
        return dict(outputs[0])

    merged: dict[str, Any] = {}
    all_keys = set().union(*(output.keys() for output in outputs))
    for key in all_keys:
        if any(key not in output for output in outputs):
            raise ValueError(f"Output field `{key}` must be present in every split output")

        values = [output[key] for output in outputs]
        first_value = values[0]
        if all(isinstance(value, np.ndarray) for value in values):
            merged[key] = np.concatenate(values, axis=0)
        elif all(isinstance(value, list) for value in values):
            merged_value: list[Any] = []
            for value in values:
                merged_value.extend(value)
            merged[key] = merged_value
        elif all(isinstance(value, tuple) for value in values):
            merged_tuple: tuple[Any, ...] = ()
            for value in values:
                merged_tuple += value
            merged[key] = merged_tuple
        else:
            for value in values[1:]:
                if not _values_match(first_value, value):
                    raise ValueError(f"Output field `{key}` differs across split outputs")
            merged[key] = first_value
    return merged


class BatchInferenceQueue:
    """Small async batcher for websocket inference requests.

    The queue accumulates requests for up to `timeout_ms` or until `max_batch_size`
    samples are available, then calls policy.predict_action once and fans results
    back to the waiting websocket handlers.
    """

    def __init__(self, policy: Any, max_batch_size: int = 8, timeout_ms: int = 20) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        self._policy = policy
        self._max_batch_size = max_batch_size
        self._timeout_s = timeout_ms / 1000.0
        self._queue: asyncio.Queue[_PendingRequest | None] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None

    async def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._worker is None or self._worker.done():
            raise RuntimeError("BatchInferenceQueue must be started before infer()")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(_PendingRequest(payload=payload, future=future))
        return await future

    async def _worker_loop(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                break

            batch = [first]
            batch_count = _payload_batch_size(first.payload)
            deadline = asyncio.get_running_loop().time() + self._timeout_s
            while batch_count < self._max_batch_size:
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                if timeout == 0.0:
                    break
                try:
                    request = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if request is None:
                    await self._queue.put(None)
                    break
                request_size = _payload_batch_size(request.payload)
                if batch_count + request_size > self._max_batch_size:
                    await self._queue.put(request)
                    break
                batch.append(request)
                batch_count += request_size

            self._run_batch(batch)

    def _run_batch(self, requests: Sequence[_PendingRequest]) -> None:
        pending = [request for request in requests if not request.future.cancelled()]
        if not pending:
            return

        try:
            payloads = [request.payload for request in pending]
            batch_sizes = [_payload_batch_size(payload) for payload in payloads]
            merged_payload = merge_payloads(payloads)
            output = self._policy.predict_action(**merged_payload)
            split_outputs = split_batched_output(output, sum(batch_sizes))
            offset = 0
            for request, request_size in zip(pending, batch_sizes):
                if request_size == 1:
                    response = split_outputs[offset]
                else:
                    response = merge_outputs(split_outputs[offset : offset + request_size])
                if not request.future.cancelled():
                    request.future.set_result(response)
                offset += request_size
        except Exception as exc:
            logging.exception("Batched policy inference failed")
            for request in pending:
                if not request.future.cancelled():
                    request.future.set_exception(exc)


def _payload_batch_size(payload: dict[str, Any]) -> int:
    batch_images = payload.get("batch_images")
    if not isinstance(batch_images, list) or not batch_images:
        raise ValueError("Each payload must contain a non-empty `batch_images` list")
    return len(batch_images)
