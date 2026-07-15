from __future__ import annotations

import collections
import dataclasses
import json
import logging
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark

from examples.LIBERO.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    Args as BaseArgs,
    M1Inference,
    _binarize_gripper_open,
    _get_libero_env,
    _quat2axisangle,
    _validate_benchmark_mode,
    short_name,
)


@dataclasses.dataclass
class Args(BaseArgs):
    save_video: bool = False
    task_start: int | None = None
    task_end: int | None = None
    shard_index: int = 0
    num_shards: int = 1


def resolve_task_range(num_tasks: int, args: Args) -> range:
    """Resolve the task IDs this eval worker should run."""
    if num_tasks < 0:
        raise ValueError("num_tasks must be non-negative")
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


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name == "libero_mix":
        task_suite = benchmark_dict[args.task_suite_name](category_value=args.category_value)
    else:
        task_suite = benchmark_dict[args.task_suite_name]()
    _validate_benchmark_mode(task_suite, args)
    num_tasks_in_suite = task_suite.n_tasks
    task_range = resolve_task_range(num_tasks_in_suite, args)
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(
        "Task shard: %s/%s, explicit range=(%s, %s), resolved=start=%s, stop=%s, step=%s, count=%s, total_tasks=%s",
        args.shard_index,
        args.num_shards,
        args.task_start,
        args.task_end,
        task_range.start,
        task_range.stop,
        task_range.step,
        len(task_range),
        num_tasks_in_suite,
    )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 250
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10" or args.task_suite_name == "libero_mix":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps is not None:
        max_steps = args.max_steps
    logging.info("Max policy steps per episode: %s", max_steps)

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        replan_steps=args.replan_steps,
    )

    total_episodes, total_successes = 0, 0
    successful_completion_steps = []
    for task_id in tqdm.tqdm(task_range):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            model.reset(task_description=task_description)
            env.reset()

            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            step = 0
            replay_images = []
            full_actions = []
            done = False

            logging.info(f"Starting task {task_id}, episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if args.save_video:
                    replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {
                    "observation.primary": np.expand_dims(img, axis=0),
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }
                obs_input = {
                    "images": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "task_description": observation["instruction"][0],
                    "step": step,
                }
                if args.with_state == "true":
                    obs_input["state"] = observation["observation.state"]

                start_time = time.time()
                response = model.step(**obs_input)
                end_time = time.time()
                logging.debug("policy step time: %.4fs", end_time - start_time)

                raw_action = response["raw_action"]
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )

                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                full_actions.append(delta_action)
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    completion_steps = step + 1
                    successful_completion_steps.append(completion_steps)
                    avg_completion_steps = float(np.mean(successful_completion_steps))
                    logging.info(
                        "Successful episode completed in %d policy steps; "
                        "running successful-episode average: %.2f policy steps over %d successes",
                        completion_steps,
                        avg_completion_steps,
                        len(successful_completion_steps),
                    )
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = short_name(task_description.replace(" ", "_"))
            if args.save_video:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_task{task_id}_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            if full_actions:
                full_actions = np.stack(full_actions)

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if successful_completion_steps:
                logging.info(
                    "Average successful completion steps so far: %.2f over %d successes",
                    float(np.mean(successful_completion_steps)),
                    len(successful_completion_steps),
                )

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes == 0:
        logging.warning("No episodes were evaluated for this shard.")
        return
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if successful_completion_steps:
        logging.info(
            "Final average successful completion steps: %.2f over %d successes",
            float(np.mean(successful_completion_steps)),
            len(successful_completion_steps),
        )


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    import os

    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
