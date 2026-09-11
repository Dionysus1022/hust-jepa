from collections import deque
from typing import Optional, Sequence
import json
import os
import warnings
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from typing import Dict
import numpy as np
from pathlib import Path


from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler



class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble = True,
        adaptive_ensemble_alpha = 0.1,
        gripper_encoding: str = "zero_one",
        host="0.0.0.0",
        port=10093,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if gripper_encoding not in ("zero_one", "minus_one_one"):
            raise ValueError(
                "gripper_encoding must be 'zero_one' or 'minus_one_one', "
                f"got {gripper_encoding!r}"
            )
        self.gripper_encoding = gripper_encoding

        if policy_setup == "widowx_bridge":
            default_unnorm_key = "oxe_bridge"
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            default_unnorm_key = "oxe_rt1"
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = self.resolve_unnorm_key(
            unnorm_key if unnorm_key is not None else default_unnorm_key,
            policy_ckpt_path,
        )

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {self.unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps


        self.cfg_scale = cfg_scale # 1.5

        self.image_size = image_size
        self.action_scale = action_scale # 1.0
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(
        self,
        task_description: str,
        *,
        gripper_open_state: Optional[float] = None,
    ) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        if gripper_open_state is None:
            self.previous_gripper_action = None
        else:
            # Google-robot policy outputs an absolute binary open-state, while
            # SimplerEnv consumes relative gripper commands.  At a subtask
            # boundary, anchor that conversion to the physical gripper instead
            # of treating the first prediction as the current gripper state.
            self.previous_gripper_action = np.asarray(
                [np.clip(float(gripper_open_state), 0.0, 1.0)],
                dtype=np.float64,
            )

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))
        # image: Image.Image = Image.fromarray(image)

        image = self._resize_image(image)
        vla_input = {
            "batch_images": [[image]],
            "instructions": [self.task_description],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        if state is not None:
            vla_input["state"] = [state]

        # Match the official VLA-JEPA SimplerEnv rollout: infer from the newest
        # observation at every environment step, ensemble overlapping chunks,
        # and execute only the ensembled action for the current step.
        response = self.client.infer(vla_input)
        if not response.get("ok", response.get("status") == "ok"):
            raise RuntimeError(f"Policy server inference failed: {response.get('error', response)}")
        if "data" not in response:
            raise RuntimeError(f"Policy server response missing `data`: {response}")
        normalized_actions = response["data"]["normalized_actions"][0]
        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions,
            action_norm_stats=self.action_norm_stats,
            gripper_encoding=self.gripper_encoding,
            # OXE Bridge and RT-1 actions are normalized with q01/q99.
            use_quantiles=True,
        )
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        if self.policy_setup in ("google_robot", "widowx_bridge"):
            # Both Fractal/RT-1 and Bridge actions store rotation deltas as
            # roll, pitch, yaw; SimplerEnv expects an axis-angle vector.
            roll, pitch, yaw = action_rotation_delta
            axes, angles = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = axes * angles
        else:
            raise NotImplementedError(f"Unsupported policy setup: {self.policy_setup}")
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        gripper_encoding: str = "zero_one",
        use_quantiles: bool = True,
    ) -> np.ndarray:
        low_key, high_key = ("q01", "q99") if use_quantiles else ("min", "max")
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats[low_key], dtype=bool))
        action_high, action_low = np.array(action_norm_stats[high_key]), np.array(action_norm_stats[low_key])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        if gripper_encoding == "zero_one":
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        elif gripper_encoding == "minus_one_one":
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.0, 1, 0)
        else:
            raise ValueError(f"Unsupported gripper_encoding: {gripper_encoding!r}")
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        if not policy_ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {policy_ckpt_path}")

        dataset_statistics_path = policy_ckpt_path.parents[1] / "dataset_statistics.json"
        if not dataset_statistics_path.is_file():
            raise FileNotFoundError(
                f"Missing dataset statistics next to checkpoint: {dataset_statistics_path}"
            )

        with dataset_statistics_path.open("r", encoding="utf-8") as file:
            norm_stats = json.load(file)

        # unnorm_key = baseframework._check_unnorm_key(norm_stats, unnorm_key) # 其实也是很环境 specific 的
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def resolve_unnorm_key(unnorm_key: str | None, policy_ckpt_path) -> str:
        norm_stats = ModelClient._read_dataset_statistics(policy_ckpt_path)
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "Checkpoint contains multiple dataset statistics keys; choose one of "
                    f"{sorted(norm_stats)} with unnorm_key."
                )
            return next(iter(norm_stats))
        if unnorm_key not in norm_stats and len(norm_stats) == 1:
            fallback_key = next(iter(norm_stats))
            warnings.warn(
                f"Requested unnorm_key {unnorm_key!r} is unavailable; "
                f"falling back to the checkpoint's only key {fallback_key!r}."
            )
            return fallback_key
        if unnorm_key not in norm_stats:
            raise ValueError(f"Unknown unnorm_key {unnorm_key!r}; choose from {sorted(norm_stats)}")
        return unnorm_key

    @staticmethod
    def _read_dataset_statistics(policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        dataset_statistics_path = policy_ckpt_path.parents[1] / "dataset_statistics.json"
        if not dataset_statistics_path.is_file():
            raise FileNotFoundError(
                f"Missing dataset statistics next to checkpoint: {dataset_statistics_path}"
            )
        with dataset_statistics_path.open("r", encoding="utf-8") as file:
            return json.load(file)



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
