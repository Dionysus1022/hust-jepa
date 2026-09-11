from __future__ import annotations

import numpy as np

from starVLA.libero_proprio import (
    canonicalize_axis_angle,
    droid_state_8d_to_normalized_proprio,
    droid_state_8d_to_proprio_7d,
    libero_gripper_qpos_to_state,
    libero_state_8d_to_normalized_proprio,
    libero_zero_pad_normalized_proprio,
    normalize_libero_eef_proprio,
    oxe_bridge_state_8d_to_normalized_proprio,
    oxe_rt1_state_8d_to_normalized_proprio,
)


def test_normalize_libero_eef_proprio_uses_fixed_physical_scales_without_clipping_xyz():
    state = np.asarray([0.0, 0.8, 1.5, np.pi, 0.0, 0.0, 0.25], dtype=np.float32)

    normalized = normalize_libero_eef_proprio(state)

    np.testing.assert_allclose(
        normalized,
        np.asarray([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -0.5], dtype=np.float32),
        atol=1e-6,
    )

    out_of_workspace = np.asarray([1.6, 0.0, 0.0, 2.0 * np.pi, 0.0, 0.0, 1.0], dtype=np.float32)
    normalized_ood = normalize_libero_eef_proprio(out_of_workspace)
    assert normalized_ood[0] > 1.0
    np.testing.assert_allclose(normalized_ood[3:6], [0.0, 0.0, 0.0], atol=1e-6)


def test_canonicalize_axis_angle_uses_shortest_arc_before_scaling():
    axis_angles = np.asarray(
        [
            [np.pi, 0.0, 0.0],
            [2.0 * np.pi, 0.0, 0.0],
            [1.5 * np.pi, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    canonical = canonicalize_axis_angle(axis_angles)

    np.testing.assert_allclose(canonical[0], [np.pi, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(canonical[1], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(canonical[2], [-0.5 * np.pi, 0.0, 0.0], atol=1e-6)


def test_libero_state_8d_to_normalized_proprio_converts_gripper_qpos_to_scalar():
    raw_state = np.asarray([[0.0, 0.0, 0.75, 0.0, 0.0, np.pi, 0.02, -0.02]], dtype=np.float32)

    normalized = libero_state_8d_to_normalized_proprio(raw_state)

    assert normalized.shape == (1, 7)
    np.testing.assert_allclose(normalized[0, :6], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(libero_gripper_qpos_to_state(raw_state[:, 6:8]), [[0.5]], atol=1e-6)
    np.testing.assert_allclose(normalized[0, 6], 0.0, atol=1e-6)


def test_libero_zero_pad_matches_training_time_raw_zero_padding():
    np.testing.assert_allclose(
        libero_zero_pad_normalized_proprio(),
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )


def test_droid_state_8d_to_proprio_7d_drops_pad_and_preserves_gripper_scalar():
    raw_state = np.asarray(
        [[0.52, 0.01, 0.31, 0.3, -0.1, -0.05, 0.0, 1.0]],
        dtype=np.float32,
    )

    proprio = droid_state_8d_to_proprio_7d(raw_state)

    assert proprio.shape == (1, 7)
    np.testing.assert_allclose(
        proprio,
        [[0.52, 0.01, 0.31, 0.3, -0.1, -0.05, 1.0]],
        atol=1e-6,
    )


def test_droid_state_uses_canonical_rotation_scale_and_open_gripper_convention():
    raw_state = np.asarray(
        [
            [0.0, 0.0, 0.75, 0.5 * np.pi, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    normalized = droid_state_8d_to_normalized_proprio(raw_state)

    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(normalized[1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], atol=1e-6)


def test_droid_google_and_widowx_share_the_same_canonical_numeric_semantics():
    droid = np.asarray([0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    google_rt1 = np.asarray([0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    widowx_bridge = np.asarray([0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    expected = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(droid_state_8d_to_normalized_proprio(droid), expected, atol=1e-6)
    np.testing.assert_allclose(oxe_rt1_state_8d_to_normalized_proprio(google_rt1), expected, atol=1e-6)
    np.testing.assert_allclose(oxe_bridge_state_8d_to_normalized_proprio(widowx_bridge), expected, atol=1e-6)
