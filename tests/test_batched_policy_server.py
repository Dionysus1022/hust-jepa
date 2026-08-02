import asyncio

import numpy as np


class _FakePolicy:
    def __init__(self):
        self.calls = []

    def predict_action(self, **payload):
        self.calls.append(payload)
        batch_size = len(payload["batch_images"])
        normalized_actions = np.arange(batch_size * 2 * 3, dtype=np.float32).reshape(batch_size, 2, 3)
        return {
            "normalized_actions": normalized_actions,
            "aux": np.arange(batch_size, dtype=np.int64),
            "shared": "ok",
        }


def test_merge_payloads_concatenates_batch_fields_and_preserves_shared_fields():
    from deployment.model_server.tools.batch_inference_queue import merge_payloads

    payloads = [
        {
            "batch_images": [["a0", "a1"]],
            "instructions": ["pick up cup"],
            "state": [np.array([[1, 2, 3]], dtype=np.float32)],
            "unnorm_key": "franka",
            "do_sample": False,
        },
        {
            "batch_images": [["b0", "b1"]],
            "instructions": ["open drawer"],
            "state": [np.array([[4, 5, 6]], dtype=np.float32)],
            "unnorm_key": "franka",
            "do_sample": False,
        },
    ]

    merged = merge_payloads(payloads)

    assert merged["batch_images"] == [["a0", "a1"], ["b0", "b1"]]
    assert merged["instructions"] == ["pick up cup", "open drawer"]
    assert len(merged["state"]) == 2
    np.testing.assert_array_equal(merged["state"][0], payloads[0]["state"][0])
    np.testing.assert_array_equal(merged["state"][1], payloads[1]["state"][0])
    assert merged["unnorm_key"] == "franka"
    assert merged["do_sample"] is False


def test_split_batched_output_returns_one_response_per_request():
    from deployment.model_server.tools.batch_inference_queue import split_batched_output

    output = {
        "normalized_actions": np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3),
        "aux": np.array([10, 11]),
        "shared": "ok",
    }

    responses = split_batched_output(output, batch_size=2)

    assert len(responses) == 2
    np.testing.assert_array_equal(responses[0]["normalized_actions"], output["normalized_actions"][0:1])
    np.testing.assert_array_equal(responses[1]["normalized_actions"], output["normalized_actions"][1:2])
    np.testing.assert_array_equal(responses[0]["aux"], output["aux"][0:1])
    np.testing.assert_array_equal(responses[1]["aux"], output["aux"][1:2])
    assert responses[0]["shared"] == "ok"
    assert responses[1]["shared"] == "ok"


def test_merge_payloads_rejects_partially_present_non_batch_fields():
    from deployment.model_server.tools.batch_inference_queue import merge_payloads

    payloads = [
        {"batch_images": [["a"]], "instructions": ["task a"]},
        {"batch_images": [["b"]], "instructions": ["task b"], "use_ddim": True},
    ]

    try:
        merge_payloads(payloads)
    except ValueError as exc:
        assert "use_ddim" in str(exc)
    else:
        raise AssertionError("Expected partially present non-batch field to fail")


def test_merge_payloads_rejects_non_batch_fields_missing_from_later_payloads():
    from deployment.model_server.tools.batch_inference_queue import merge_payloads

    payloads = [
        {"batch_images": [["a"]], "instructions": ["task a"], "use_ddim": True},
        {"batch_images": [["b"]], "instructions": ["task b"]},
    ]

    try:
        merge_payloads(payloads)
    except ValueError as exc:
        assert "use_ddim" in str(exc)
    else:
        raise AssertionError("Expected missing non-batch field to fail")


def test_batch_inference_queue_batches_concurrent_requests():
    from deployment.model_server.tools.batch_inference_queue import BatchInferenceQueue

    async def run_requests():
        policy = _FakePolicy()
        queue = BatchInferenceQueue(policy=policy, max_batch_size=4, timeout_ms=50)
        await queue.start()
        try:
            results = await asyncio.gather(
                queue.infer({"batch_images": [["a"]], "instructions": ["task a"]}),
                queue.infer({"batch_images": [["b"]], "instructions": ["task b"]}),
            )
        finally:
            await queue.stop()
        return policy, results

    policy, results = asyncio.run(run_requests())

    assert len(policy.calls) == 1
    assert policy.calls[0]["batch_images"] == [["a"], ["b"]]
    assert policy.calls[0]["instructions"] == ["task a", "task b"]
    np.testing.assert_array_equal(
        results[0]["normalized_actions"],
        np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        results[1]["normalized_actions"],
        np.array([[[6, 7, 8], [9, 10, 11]]], dtype=np.float32),
    )


def test_batch_inference_queue_preserves_multi_sample_client_responses():
    from deployment.model_server.tools.batch_inference_queue import BatchInferenceQueue

    async def run_requests():
        policy = _FakePolicy()
        queue = BatchInferenceQueue(policy=policy, max_batch_size=4, timeout_ms=1)
        await queue.start()
        try:
            result = await queue.infer(
                {
                    "batch_images": [["a"], ["b"]],
                    "instructions": ["task a", "task b"],
                }
            )
        finally:
            await queue.stop()
        return result

    result = asyncio.run(run_requests())

    np.testing.assert_array_equal(
        result["normalized_actions"],
        np.array(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
            ],
            dtype=np.float32,
        ),
    )
