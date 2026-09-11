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


def quaternion_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to shortest-arc axis-angle vectors."""
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected last dimension 4 for XYZW quaternion, got shape {quat.shape}")

    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = np.zeros_like(quat, dtype=np.float32)
    identity[..., 3] = 1.0
    quat = np.where(norm > 1e-8, quat / np.maximum(norm, 1e-8), identity)

    # q and -q encode the same rotation. Keeping w non-negative selects the
    # shortest rotation and avoids discontinuities around 2*pi.
    quat = np.where(quat[..., 3:4] < 0.0, -quat, quat)
    sin_half = np.linalg.norm(quat[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quat[..., 3:4], 0.0, 1.0))
    scale = np.where(sin_half > 1e-8, angle / np.maximum(sin_half, 1e-8), 2.0)
    return canonicalize_axis_angle(quat[..., :3] * scale)


def euler_rpy_to_axis_angle(euler_rpy: np.ndarray) -> np.ndarray:
    """Convert extrinsic XYZ roll-pitch-yaw angles to axis-angle vectors."""
    rpy = np.asarray(euler_rpy, dtype=np.float32)
    if rpy.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3 for RPY angles, got shape {rpy.shape}")

    half = 0.5 * rpy
    sr, sp, sy = np.sin(half[..., 0]), np.sin(half[..., 1]), np.sin(half[..., 2])
    cr, cp, cy = np.cos(half[..., 0]), np.cos(half[..., 1]), np.cos(half[..., 2])
    quat_xyzw = np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        axis=-1,
    )
    return quaternion_xyzw_to_axis_angle(quat_xyzw)


def oxe_bridge_state_8d_to_normalized_proprio(state: np.ndarray) -> np.ndarray:
    """Convert Bridge [xyz, RPY, pad, gripper] state to canonical normalized 7D."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for Bridge state, got shape {arr.shape}")
    rotation = euler_rpy_to_axis_angle(arr[..., 3:6])
    proprio = np.concatenate([arr[..., :3], rotation, arr[..., 7:8]], axis=-1)
    return normalize_libero_eef_proprio(proprio)


def oxe_rt1_state_8d_to_normalized_proprio(state: np.ndarray) -> np.ndarray:
    """Convert RT-1 [xyz, quaternion_xyzw, gripper_closed] to normalized open-state 7D."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for RT-1 state, got shape {arr.shape}")
    rotation = quaternion_xyzw_to_axis_angle(arr[..., 3:7])
    # Fractal exposes observation.gripper_closed (0=open, 1=closed), while the
    # canonical VLA-JEPA proprio convention is gripper_open (0=closed, 1=open).
    gripper_open = 1.0 - np.clip(arr[..., 7:8], 0.0, 1.0)
    proprio = np.concatenate([arr[..., :3], rotation, gripper_open], axis=-1)
    return normalize_libero_eef_proprio(proprio)


def libero_state_8d_to_normalized_proprio(state: np.ndarray) -> np.ndarray:
    """Convert raw LIBERO 8D EEF+gripper state to normalized 7D proprio."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for raw LIBERO state, got shape {arr.shape}")
    gripper_state = libero_gripper_qpos_to_state(arr[..., 6:8])
    proprio = np.concatenate([arr[..., :6], gripper_state], axis=-1)
    return normalize_libero_eef_proprio(proprio)


def droid_state_8d_to_proprio_7d(state: np.ndarray) -> np.ndarray:
    """Legacy raw DROID conversion that only removes the padding slot."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for raw DROID state, got shape {arr.shape}")
    return np.concatenate([arr[..., :6], arr[..., 7:8]], axis=-1)


def droid_state_8d_to_normalized_proprio(state: np.ndarray) -> np.ndarray:
    """Convert DROID [xyz, RPY, pad, gripper_closed] to canonical normalized 7D.

    DROID reports gripper position as ``1 - width / max_width`` (0=open,
    1=closed).  VLA-JEPA's cross-embodiment convention stores gripper_open.
    """
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dimension 8 for raw DROID state, got shape {arr.shape}")
    rotation = euler_rpy_to_axis_angle(arr[..., 3:6])
    gripper_open = 1.0 - np.clip(arr[..., 7:8], 0.0, 1.0)
    proprio = np.concatenate([arr[..., :3], rotation, gripper_open], axis=-1)
    return normalize_libero_eef_proprio(proprio)


def libero_zero_pad_normalized_proprio() -> np.ndarray:
    """Return normalized raw-zero state for legacy evaluation compatibility."""
    return libero_state_8d_to_normalized_proprio(np.zeros(8, dtype=np.float32))
