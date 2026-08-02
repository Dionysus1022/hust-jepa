from __future__ import annotations

import numpy as np


# Conservative physical scale anchors for LIBERO end-effector position.
# These are not dataset min/max statistics and are intentionally not used for
# clipping; out-of-range Robot Initial states should stay distinguishable.
LIBERO_EEF_POS_CENTER = np.asarray([0.0, 0.0, 0.75], dtype=np.float32)
LIBERO_EEF_POS_SCALE = np.asarray([0.8, 0.8, 0.75], dtype=np.float32)


def libero_gripper_qpos_to_state(gripper_qpos: np.ndarray) -> np.ndarray:
    """Convert LIBERO two-finger qpos to a scalar open-state in [0, 1]."""
    qpos = np.asarray(gripper_qpos, dtype=np.float32)
    gripper_state = 1.0 - (np.mean(np.abs(qpos), axis=-1, keepdims=True) / 0.04)
    return np.clip(gripper_state, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_libero_eef_proprio(proprio: np.ndarray) -> np.ndarray:
    """Normalize 7D LIBERO proprio [eef_xyz, axis_angle, gripper] using fixed physical scales."""
    arr = np.asarray(proprio, dtype=np.float32)
    if arr.shape[-1] != 7:
        raise ValueError(f"Expected last dimension 7 for LIBERO proprio, got shape {arr.shape}")

    normalized = np.empty_like(arr, dtype=np.float32)
    normalized[..., :3] = (arr[..., :3] - LIBERO_EEF_POS_CENTER) / LIBERO_EEF_POS_SCALE
    normalized[..., 3:6] = canonicalize_axis_angle(arr[..., 3:6]) / np.pi
    normalized[..., 6:7] = 2.0 * np.clip(arr[..., 6:7], 0.0, 1.0) - 1.0
    return normalized


def canonicalize_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Map axis-angle vectors to the equivalent shortest-arc representation."""
    vec = np.asarray(axis_angle, dtype=np.float32)
    theta = np.linalg.norm(vec, axis=-1, keepdims=True)
    safe_theta = np.where(theta > 0.0, theta, 1.0)
    wrapped_theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
    wrapped_theta = np.where(np.isclose(wrapped_theta, -np.pi), np.pi, wrapped_theta)
    return vec * (wrapped_theta / safe_theta)


def libero_state_8d_to_normalized_proprio(state: np.ndarray) -> np.ndarray:
    """Convert raw LIBERO 8D EEF+gripper state to normalized 7D proprio."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for raw LIBERO state, got shape {arr.shape}")
    gripper_state = libero_gripper_qpos_to_state(arr[..., 6:8])
    proprio = np.concatenate([arr[..., :6], gripper_state], axis=-1)
    return normalize_libero_eef_proprio(proprio)


def droid_state_8d_to_proprio_7d(state: np.ndarray) -> np.ndarray:
    """Convert DROID [eef_xyz, eef_rpy, pad, gripper] state to 7D proprio."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for raw DROID state, got shape {arr.shape}")
    return np.concatenate([arr[..., :6], arr[..., 7:8]], axis=-1)


def libero_zero_pad_normalized_proprio() -> np.ndarray:
    """Return the normalized 7D state produced by training-time raw zero padding."""
    return libero_state_8d_to_normalized_proprio(np.zeros(8, dtype=np.float32))
