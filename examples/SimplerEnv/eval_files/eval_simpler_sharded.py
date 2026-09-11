from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os
import pathlib
from typing import Any

import imageio
import numpy as np
from scipy.spatial.transform import Rotation
from sapien.core import Pose
from transforms3d.euler import euler2quat

from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from starVLA.libero_proprio import normalize_libero_eef_proprio


URDF_VERSIONS = (
    None,
    "recolor_tabletop_visual_matching_1",
    "recolor_tabletop_visual_matching_2",
    "recolor_cabinet_visual_matching_1",
)
DRAWER_URDF_VERSIONS = (
    "recolor_cabinet_visual_matching_1",
    "recolor_tabletop_visual_matching_1",
    "recolor_tabletop_visual_matching_2",
    None,
)


@dataclasses.dataclass(frozen=True)
class EvalCase:
    env_name: str
    scene_name: str
    robot: str
    policy_setup: str
    control_freq: int
    sim_freq: int
    max_episode_steps: int
    rgb_overlay_path: str
    robot_init_x: float
    robot_init_y: float
    robot_init_rpy: tuple[float, float, float]
    obj_variation_mode: str
    obj_init_x: float | None = None
    obj_init_y: float | None = None
    obj_episode_id: int | None = None
    enable_raytracing: bool = False
    additional_env_build_kwargs: tuple[tuple[str, Any], ...] = ()

    def env_build_kwargs(self) -> dict[str, Any]:
        return dict(self.additional_env_build_kwargs)

    def summary(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    task_suite_name: str = "simpler_env"
    benchmark_mode: str = "simpler_env"
    num_trials_per_task: int = 1
    max_steps: int | None = None
    # Deprecated compatibility option. Official VLA-JEPA SimplerEnv evaluation
    # always replans from the newest observation at every environment step.
    replan_steps: int | None = None
    unnorm_key: str | None = None
    gripper_encoding: str = "zero_one"
    category_value: str = "pick_coke_can"
    video_out_path: str = "results/simpler_env"
    seed: int = 7
    pretrained_path: str = ""
    with_state: str = "true"
    reset_state_history_on_subtask_change: bool = True
    save_video: bool = False
    task_start: int | None = None
    task_end: int | None = None
    shard_index: int = 0
    num_shards: int = 1


def _overlay(simpler_env_path: pathlib.Path, name: str) -> str:
    return str(simpler_env_path / "ManiSkill2_real2sim" / "data" / "real_inpainting" / name)


def _kwargs(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(values.items())


def _drawer_robot_poses(simpler_env_path: pathlib.Path) -> tuple[tuple[str, float, float, float], ...]:
    return (
        (_overlay(simpler_env_path, "open_drawer_a0.png"), 0.644, -0.179, -0.03),
        (_overlay(simpler_env_path, "open_drawer_a1.png"), 0.765, -0.182, -0.02),
        (_overlay(simpler_env_path, "open_drawer_a2.png"), 0.889, -0.203, -0.06),
        (_overlay(simpler_env_path, "open_drawer_b0.png"), 0.652, 0.009, 0.0),
        (_overlay(simpler_env_path, "open_drawer_b1.png"), 0.752, 0.009, 0.0),
        (_overlay(simpler_env_path, "open_drawer_b2.png"), 0.851, 0.035, 0.0),
        (_overlay(simpler_env_path, "open_drawer_c0.png"), 0.665, 0.224, 0.0),
        (_overlay(simpler_env_path, "open_drawer_c1.png"), 0.765, 0.222, -0.025),
        (_overlay(simpler_env_path, "open_drawer_c2.png"), 0.865, 0.222, -0.025),
    )


def build_eval_cases(category_value: str, simpler_env_path: pathlib.Path) -> list[EvalCase]:
    cases: list[EvalCase] = []

    if category_value == "bridge_put_on":
        for env_name in (
            "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
            "PutCarrotOnPlateInScene-v0",
            "PutSpoonOnTableClothInScene-v0",
        ):
            for episode_id in range(24):
                cases.append(
                    EvalCase(
                        env_name=env_name,
                        scene_name="bridge_table_1_v1",
                        robot="widowx",
                        policy_setup="widowx_bridge",
                        control_freq=5,
                        sim_freq=500,
                        max_episode_steps=120,
                        rgb_overlay_path=_overlay(simpler_env_path, "bridge_real_eval_1.png"),
                        robot_init_x=0.147,
                        robot_init_y=0.028,
                        robot_init_rpy=(0.0, 0.0, 0.0),
                        obj_variation_mode="episode",
                        obj_episode_id=episode_id,
                    )
                )
        for episode_id in range(24):
            cases.append(
                EvalCase(
                    env_name="PutEggplantInBasketScene-v0",
                    scene_name="bridge_table_1_v2",
                    robot="widowx_sink_camera_setup",
                    policy_setup="widowx_bridge",
                    control_freq=5,
                    sim_freq=500,
                    max_episode_steps=120,
                    rgb_overlay_path=_overlay(simpler_env_path, "bridge_sink.png"),
                    robot_init_x=0.127,
                    robot_init_y=0.06,
                    robot_init_rpy=(0.0, 0.0, 0.0),
                    obj_variation_mode="episode",
                    obj_episode_id=episode_id,
                )
            )

    elif category_value == "drawer":
        for urdf_version in DRAWER_URDF_VERSIONS:
            build_kwargs = _kwargs(
                station_name="mk_station_recolor",
                light_mode="simple",
                disable_bad_material=True,
                urdf_version=urdf_version,
            )
            for env_name in (
                "OpenTopDrawerCustomInScene-v0",
                "OpenMiddleDrawerCustomInScene-v0",
                "OpenBottomDrawerCustomInScene-v0",
                "CloseTopDrawerCustomInScene-v0",
                "CloseMiddleDrawerCustomInScene-v0",
                "CloseBottomDrawerCustomInScene-v0",
            ):
                for overlay_path, robot_x, robot_y, yaw in _drawer_robot_poses(simpler_env_path):
                    cases.append(
                        EvalCase(
                            env_name=env_name,
                            scene_name="dummy_drawer",
                            robot="google_robot_static",
                            policy_setup="google_robot",
                            control_freq=3,
                            sim_freq=513,
                            max_episode_steps=113,
                            rgb_overlay_path=overlay_path,
                            robot_init_x=robot_x,
                            robot_init_y=robot_y,
                            robot_init_rpy=(0.0, 0.0, yaw),
                            obj_variation_mode="xy",
                            obj_init_x=0.0,
                            obj_init_y=0.0,
                            enable_raytracing=True,
                            additional_env_build_kwargs=build_kwargs,
                        )
                    )

    elif category_value == "move_near":
        for urdf_version in URDF_VERSIONS:
            for episode_id in range(60):
                cases.append(
                    EvalCase(
                        env_name="MoveNearGoogleBakedTexInScene-v0",
                        scene_name="google_pick_coke_can_1_v4",
                        robot="google_robot_static",
                        policy_setup="google_robot",
                        control_freq=3,
                        sim_freq=513,
                        max_episode_steps=80,
                        rgb_overlay_path=_overlay(simpler_env_path, "google_move_near_real_eval_1.png"),
                        robot_init_x=0.35,
                        robot_init_y=0.21,
                        robot_init_rpy=(0.0, 0.0, -0.09),
                        obj_variation_mode="episode",
                        obj_episode_id=episode_id,
                        additional_env_build_kwargs=_kwargs(urdf_version=urdf_version),
                    )
                )

    elif category_value == "pick_coke_can":
        coke_can_options = (
            _kwargs(lr_switch=True),
            _kwargs(upright=True),
            _kwargs(laid_vertically=True),
        )
        for urdf_version in URDF_VERSIONS:
            for option in coke_can_options:
                build_kwargs = dict(option)
                build_kwargs["urdf_version"] = urdf_version
                for obj_x in np.linspace(-0.35, -0.12, 5):
                    for obj_y in np.linspace(-0.02, 0.42, 5):
                        cases.append(
                            EvalCase(
                                env_name="GraspSingleOpenedCokeCanInScene-v0",
                                scene_name="google_pick_coke_can_1_v4",
                                robot="google_robot_static",
                                policy_setup="google_robot",
                                control_freq=3,
                                sim_freq=513,
                                max_episode_steps=80,
                                rgb_overlay_path=_overlay(
                                    simpler_env_path, "google_coke_can_real_eval_1.png"
                                ),
                                robot_init_x=0.35,
                                robot_init_y=0.20,
                                robot_init_rpy=(0.0, 0.0, 0.0),
                                obj_variation_mode="xy",
                                obj_init_x=float(obj_x),
                                obj_init_y=float(obj_y),
                                additional_env_build_kwargs=tuple(build_kwargs.items()),
                            )
                        )

    elif category_value == "long_horizon_apple_in_drawer":
        drawer_poses = (
            (_overlay(simpler_env_path, "open_drawer_a0.png"), 0.644, -0.179, -0.03),
            (_overlay(simpler_env_path, "open_drawer_b0.png"), 0.652, 0.009, 0.0),
            (_overlay(simpler_env_path, "open_drawer_c0.png"), 0.665, 0.224, 0.0),
        )
        for urdf_version in DRAWER_URDF_VERSIONS:
            build_kwargs = _kwargs(
                station_name="mk_station_recolor",
                light_mode="simple",
                disable_bad_material=True,
                urdf_version=urdf_version,
                model_ids="baked_apple_v2",
            )
            for overlay_path, robot_x, robot_y, yaw in drawer_poses:
                for obj_x in np.linspace(-0.08, -0.02, 3):
                    for obj_y in np.linspace(-0.02, 0.08, 3):
                        cases.append(
                            EvalCase(
                                env_name="PlaceIntoClosedTopDrawerCustomInScene-v0",
                                scene_name="dummy_drawer",
                                robot="google_robot_static",
                                policy_setup="google_robot",
                                control_freq=3,
                                sim_freq=513,
                                max_episode_steps=200,
                                rgb_overlay_path=overlay_path,
                                robot_init_x=robot_x,
                                robot_init_y=robot_y,
                                robot_init_rpy=(0.0, 0.0, yaw),
                                obj_variation_mode="xy",
                                obj_init_x=float(obj_x),
                                obj_init_y=float(obj_y),
                                enable_raytracing=True,
                                additional_env_build_kwargs=build_kwargs,
                            )
                        )
    else:
        raise ValueError(
            f"Unknown SimplerEnv category {category_value!r}; choose from "
            "bridge_put_on, drawer, move_near, pick_coke_can, "
            "long_horizon_apple_in_drawer"
        )

    return cases


def resolve_task_ids(num_tasks: int, args: Args) -> range:
    if args.task_start is not None or args.task_end is not None:
        start = 0 if args.task_start is None else args.task_start
        end = num_tasks if args.task_end is None else min(args.task_end, num_tasks)
        if start < 0 or end < start:
            raise ValueError(f"Invalid task range: start={start}, end={end}, num_tasks={num_tasks}")
        return range(start, end)
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard_index must be in [0, {args.num_shards}), got {args.shard_index}")
    return range(args.shard_index, num_tasks, args.num_shards)


def _robot_init_quat(rpy: tuple[float, float, float]) -> np.ndarray:
    return (Pose(q=euler2quat(*rpy)) * Pose(q=[0, 0, 0, 1])).q


def _state_history_array(history: collections.deque[np.ndarray]) -> np.ndarray:
    states = list(history)
    if not states:
        raise ValueError("Cannot build proprio history before observing the current state")
    if len(states) < 8:
        states = [states[0]] * (8 - len(states)) + states
    return np.stack(states[-8:], axis=0).astype(np.float32)


def _simpler_gripper_open(env) -> float:
    gripper_closedness = float(env.unwrapped.agent.get_gripper_closedness())
    return float(np.clip(1.0 - gripper_closedness, 0.0, 1.0))


def _simpler_proprio(env) -> np.ndarray:
    unwrapped = env.unwrapped
    tcp_pose_at_base = unwrapped.agent.robot.pose.inv().transform(unwrapped.tcp.pose)
    quat_xyzw = np.asarray(tcp_pose_at_base.q, dtype=np.float32)[[1, 2, 3, 0]]
    axis_angle = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    gripper_open = _simpler_gripper_open(env)
    proprio = np.concatenate(
        [np.asarray(tcp_pose_at_base.p, dtype=np.float32), axis_angle, [gripper_open]]
    )
    return normalize_libero_eef_proprio(proprio)


def run_episode(
    model: ModelClient, case: EvalCase, args: Args
) -> tuple[bool, bool, int, list[np.ndarray]]:
    build_kwargs = case.env_build_kwargs()
    if case.enable_raytracing:
        build_kwargs = {"shader_dir": "rt", **build_kwargs}
    control_mode = get_robot_control_mode(case.robot, "rt1")
    env = build_maniskill2_env(
        case.env_name,
        obs_mode="rgbd",
        robot=case.robot,
        sim_freq=case.sim_freq,
        control_mode=control_mode,
        control_freq=case.control_freq,
        max_episode_steps=args.max_steps or case.max_episode_steps,
        scene_name=case.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=case.rgb_overlay_path,
        **build_kwargs,
    )
    reset_options: dict[str, Any] = {
        "robot_init_options": {
            "init_xy": np.asarray([case.robot_init_x, case.robot_init_y]),
            "init_rot_quat": _robot_init_quat(case.robot_init_rpy),
        }
    }
    if case.obj_variation_mode == "episode":
        reset_options["obj_init_options"] = {"episode_id": case.obj_episode_id}
    else:
        reset_options["obj_init_options"] = {
            "init_xy": np.asarray([case.obj_init_x, case.obj_init_y])
        }

    try:
        obs, _ = env.reset(options=reset_options)
        task_description = env.get_language_instruction()
        model.reset(task_description)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        replay_images = [image] if args.save_video else []
        state_history: collections.deque[np.ndarray] = collections.deque(maxlen=8)
        any_success = False
        final_success = False
        first_success_step = 0
        step = 0
        truncated = False
        while not truncated:
            state = None
            if args.with_state == "true":
                state_history.append(_simpler_proprio(env))
                state = _state_history_array(state_history)
            elif args.with_state == "zero":
                # Inference-time proprio ablation: preserve all state-conditioned
                # model paths and token shapes while removing state information.
                state = np.zeros((8, 7), dtype=np.float32)
            _, action = model.step(image, task_description, state=state, step=step)
            was_final_subtask = bool(env.unwrapped.is_final_subtask())
            obs, _, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            step += 1
            # Keep both evaluation conventions. SimplerEnv's official evaluator
            # reports the final step's `done`, while the looser metric latches
            # whether success occurred at any point during the episode.
            final_success = bool(done)
            if final_success and not any_success:
                any_success = True
                first_success_step = step

            transition_reason = None
            if (
                args.category_value == "long_horizon_apple_in_drawer"
                and not was_final_subtask
            ):
                is_final_subtask = bool(env.unwrapped.is_final_subtask())
                episode_stats = info.get("episode_stats", {})
                drawer_open = bool(episode_stats.get("is_drawer_open", False))
                if drawer_open and not is_final_subtask:
                    env.unwrapped.advance_to_next_subtask()
                    transition_reason = "drawer success"
                    logging.info(
                        "Advancing to placement after drawer success at step=%s, qpos=%s",
                        step,
                        episode_stats.get("qpos"),
                    )
                elif is_final_subtask:
                    transition_reason = "100-step fallback"

            new_task_description = env.get_language_instruction()
            if new_task_description != task_description:
                previous_task_description = task_description
                task_description = new_task_description
                gripper_open_state = _simpler_gripper_open(env)
                model.reset(
                    task_description,
                    gripper_open_state=gripper_open_state,
                )
                if args.reset_state_history_on_subtask_change:
                    state_history.clear()
                    state_history_action = "cleared"
                else:
                    state_history_action = "kept"
                logging.info(
                    "Subtask changed from %r to %r via %s; reset action ensemble, "
                    "synced gripper_open=%.4f, and %s proprio history",
                    previous_task_description,
                    task_description,
                    transition_reason or "environment transition",
                    gripper_open_state,
                    state_history_action,
                )
            image = get_image_from_maniskill2_obs_dict(env, obs)
            if args.save_video:
                replay_images.append(image)
        return final_success, any_success, first_success_step, replay_images
    finally:
        env.close()


def eval_simpler(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=4))
    if args.benchmark_mode != "simpler_env":
        raise ValueError(f"benchmark_mode must be 'simpler_env', got {args.benchmark_mode!r}")
    if args.with_state not in ("true", "false", "zero"):
        raise ValueError("with_state must be 'true', 'false', or 'zero'")
    logging.info("State input mode: %s", args.with_state)
    logging.info(
        "Proprio history on subtask change: %s",
        "reset" if args.reset_state_history_on_subtask_change else "keep",
    )
    np.random.seed(args.seed)
    simpler_env_path = pathlib.Path(os.environ.get("SimplerEnv_PATH", "/home/WangBizi/SimplerEnv"))
    cases = build_eval_cases(args.category_value, simpler_env_path)
    task_ids = resolve_task_ids(len(cases), args)
    logging.info(
        "Task shard: %s/%s, resolved=start=%s, stop=%s, step=%s, count=%s, total_tasks=%s",
        args.shard_index,
        args.num_shards,
        task_ids.start,
        task_ids.stop,
        task_ids.step,
        len(task_ids),
        len(cases),
    )
    output_path = pathlib.Path(args.video_out_path)
    output_path.mkdir(parents=True, exist_ok=True)
    first_case = cases[0]
    requested_unnorm_key = None if args.unnorm_key in (None, "", "auto") else args.unnorm_key
    if args.replan_steps is not None:
        logging.warning(
            "Ignoring deprecated replan_steps=%s: official VLA-JEPA SimplerEnv "
            "execution replans every step and uses adaptive action ensembling.",
            args.replan_steps,
        )
    model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=requested_unnorm_key,
        policy_setup=first_case.policy_setup,
        host=args.host,
        port=args.port,
        gripper_encoding=args.gripper_encoding,
        action_ensemble=True,
    )
    logging.info(
        "Action execution: official closed-loop, infer_every_step=true, "
        "adaptive_ensemble=true, ensemble_horizon=%s",
        model.action_ensemble_horizon,
    )

    total_episodes = 0
    total_final_successes = 0
    total_any_successes = 0
    total_cases = len(task_ids)
    total_planned_episodes = total_cases * args.num_trials_per_task
    completed_cases = 0
    any_success_first_steps: list[int] = []
    task_results = []
    for task_id in task_ids:
        case = cases[task_id]
        task_final_successes = 0
        task_any_successes = 0
        for episode_idx in range(args.num_trials_per_task):
            logging.info("Starting task %s, episode %s", task_id, episode_idx + 1)
            final_success, any_success, first_success_step, replay_images = run_episode(
                model, case, args
            )
            total_episodes += 1
            task_final_successes += int(final_success)
            task_any_successes += int(any_success)
            total_final_successes += int(final_success)
            total_any_successes += int(any_success)
            if any_success:
                any_success_first_steps.append(first_success_step)
            if args.save_video:
                suffix = (
                    f"final_{'success' if final_success else 'failure'}_"
                    f"any_{'success' if any_success else 'failure'}"
                )
                imageio.mimwrite(
                    output_path / f"rollout_task{task_id}_episode{episode_idx}_{suffix}.mp4",
                    replay_images,
                    fps=5,
                )
        completed_cases += 1
        case_final_success_rate = task_final_successes / args.num_trials_per_task
        case_any_success_rate = task_any_successes / args.num_trials_per_task
        shard_final_success_rate = total_final_successes / total_episodes
        shard_any_success_rate = total_any_successes / total_episodes
        logging.info(
            "Completed case task_id=%s | cases=%s/%s | "
            "case_final_successes=%s/%s (%.4f) | case_any_successes=%s/%s (%.4f) | "
            "shard_episodes=%s/%s | shard_final_successes=%s (%.4f) | "
            "shard_any_successes=%s (%.4f)",
            task_id,
            completed_cases,
            total_cases,
            task_final_successes,
            args.num_trials_per_task,
            case_final_success_rate,
            task_any_successes,
            args.num_trials_per_task,
            case_any_success_rate,
            total_episodes,
            total_planned_episodes,
            total_final_successes,
            shard_final_success_rate,
            total_any_successes,
            shard_any_success_rate,
        )
        task_results.append(
            {
                "task_id": task_id,
                "task": case.summary(),
                "episodes": args.num_trials_per_task,
                # Backward-compatible generic fields use the official final-state metric.
                "successes": task_final_successes,
                "success_rate": case_final_success_rate,
                "final_successes": task_final_successes,
                "final_success_rate": case_final_success_rate,
                "any_successes": task_any_successes,
                "any_success_rate": case_any_success_rate,
            }
        )

    avg_any_success_first_step = (
        float(np.mean(any_success_first_steps)) if any_success_first_steps else None
    )
    summary = {
        "task_suite_name": args.task_suite_name,
        "benchmark_mode": args.benchmark_mode,
        "category_value": args.category_value,
        "state_input_mode": args.with_state,
        "seed": args.seed,
        "pretrained_path": args.pretrained_path,
        "reset_state_history_on_subtask_change": (
            args.reset_state_history_on_subtask_change
        ),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "task_start": task_ids.start,
        "task_end": task_ids.stop,
        "task_step": task_ids.step,
        "task_ids": list(task_ids),
        "total_episodes": total_episodes,
        # Backward-compatible generic fields use the official final-state metric.
        "total_successes": total_final_successes,
        "success_rate": total_final_successes / total_episodes if total_episodes else 0.0,
        "total_final_successes": total_final_successes,
        "final_success_rate": total_final_successes / total_episodes if total_episodes else 0.0,
        "total_any_successes": total_any_successes,
        "any_success_rate": total_any_successes / total_episodes if total_episodes else 0.0,
        "transient_only_successes": total_any_successes - total_final_successes,
        "avg_any_success_first_step": avg_any_success_first_step,
        "any_success_first_steps": any_success_first_steps,
        "tasks": task_results,
    }
    summary_path = output_path / "result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote shard result summary to %s", summary_path)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args.host", dest="host", default="127.0.0.1")
    parser.add_argument("--args.port", dest="port", type=int, default=10093)
    parser.add_argument("--args.task-suite-name", dest="task_suite_name", default="simpler_env")
    parser.add_argument("--args.benchmark-mode", dest="benchmark_mode", default="simpler_env")
    parser.add_argument("--args.num-trials-per-task", dest="num_trials_per_task", type=int, default=1)
    parser.add_argument("--args.max-steps", dest="max_steps", type=int)
    parser.add_argument(
        "--args.replan-steps",
        dest="replan_steps",
        type=int,
        help="Deprecated and ignored; retained only for compatibility with older commands.",
    )
    parser.add_argument("--args.unnorm-key", dest="unnorm_key")
    parser.add_argument("--args.gripper-encoding", dest="gripper_encoding", default="zero_one")
    parser.add_argument("--args.category-value", dest="category_value", default="pick_coke_can")
    parser.add_argument("--args.video-out-path", dest="video_out_path", default="results/simpler_env")
    parser.add_argument("--args.seed", dest="seed", type=int, default=7)
    parser.add_argument("--args.pretrained-path", dest="pretrained_path", required=True)
    parser.add_argument(
        "--args.with-state",
        dest="with_state",
        choices=("true", "false", "zero"),
        default="true",
        help="Use real proprio, omit proprio, or inject an all-zero [8, 7] history.",
    )
    parser.add_argument(
        "--args.reset-state-history-on-subtask-change",
        dest="reset_state_history_on_subtask_change",
        action="store_true",
        help="Clear proprio history when the environment changes instruction (default).",
    )
    parser.add_argument(
        "--args.keep-state-history-on-subtask-change",
        dest="reset_state_history_on_subtask_change",
        action="store_false",
        help="Keep continuous proprio history when the environment changes instruction.",
    )
    parser.add_argument("--args.save-video", dest="save_video", action="store_true")
    parser.add_argument("--args.no-save-video", dest="save_video", action="store_false")
    parser.set_defaults(save_video=False, reset_state_history_on_subtask_change=True)
    parser.add_argument("--args.task-start", dest="task_start", type=int)
    parser.add_argument("--args.task-end", dest="task_end", type=int)
    parser.add_argument("--args.shard-index", dest="shard_index", type=int, default=0)
    parser.add_argument("--args.num-shards", dest="num_shards", type=int, default=1)
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    eval_simpler(parse_args())
