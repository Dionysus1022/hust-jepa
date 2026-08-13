from __future__ import annotations
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

_VJ_PROCESSOR = None
_VJ_PROCESSOR_PATH = None
_QWEN_PROCESSOR = None
_QWEN_PROCESSOR_PATH = None


def _sample_range(value, default):
    if value is None:
        return default
    if len(value) == 1:
        delta = float(value[0])
        return float(np.random.uniform(1.0 - delta, 1.0 + delta))
    return float(np.random.uniform(float(value[0]), float(value[1])))


def _sample_absolute_range(value, default):
    if value is None:
        return default
    if len(value) == 1:
        return float(value[0])
    return float(np.random.uniform(float(value[0]), float(value[1])))


def _augment_video_frames(video, augmentation):
    if not augmentation or not augmentation.get("enabled", False):
        return video
    order = list(augmentation.get("augment_order", []))
    if not order:
        return video

    from torchvision.transforms import InterpolationMode, RandomResizedCrop
    from torchvision.transforms import functional as TVF

    frames = video
    is_video = frames.ndim == 4
    def to_uint8(frame):
        if frame.dtype == np.uint8:
            return frame
        frame = np.asarray(frame)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        return np.clip(frame, 0, 255).astype(np.uint8)

    pil_frames = [Image.fromarray(to_uint8(frame)) for frame in (frames if is_video else [frames])]
    out_h, out_w = pil_frames[0].height, pil_frames[0].width

    crop_params = None
    rrc = augmentation.get("random_resized_crop", None)
    if "random_resized_crop" in order and rrc is not None:
        scale = tuple(rrc.get("scale", (0.9, 1.0)))
        ratio = tuple(rrc.get("ratio", (1.0, 1.0)))
        crop_params = RandomResizedCrop.get_params(pil_frames[0], scale=scale, ratio=ratio)

    brightness = _sample_range(augmentation.get("random_brightness", None), 1.0)
    contrast = _sample_range(augmentation.get("random_contrast", None), 1.0)
    saturation = _sample_range(augmentation.get("random_saturation", None), 1.0)
    hue_cfg = augmentation.get("random_hue", None)
    hue = 0.0
    if hue_cfg:
        if len(hue_cfg) == 1:
            hue = float(np.random.uniform(-float(hue_cfg[0]), float(hue_cfg[0])))
        else:
            hue = float(np.random.uniform(float(hue_cfg[0]), float(hue_cfg[1])))
        hue = float(np.clip(hue, -0.5, 0.5))
    gamma = _sample_absolute_range(augmentation.get("random_gamma", None), 1.0)
    exposure_ev_cfg = augmentation.get("random_exposure_ev", None)
    exposure_ev = 0.0
    if exposure_ev_cfg:
        if len(exposure_ev_cfg) == 1:
            ev = float(exposure_ev_cfg[0])
            exposure_ev = float(np.random.uniform(-ev, ev))
        else:
            exposure_ev = float(np.random.uniform(float(exposure_ev_cfg[0]), float(exposure_ev_cfg[1])))
    gaussian_noise_std = float(augmentation.get("gaussian_noise_std", 0.0) or 0.0)
    rotation_degrees = augmentation.get("random_rotation_degrees", None)
    rotation_angle = 0.0
    if rotation_degrees:
        if len(rotation_degrees) == 1:
            degrees = float(rotation_degrees[0])
            rotation_angle = float(np.random.uniform(-degrees, degrees))
        else:
            rotation_angle = float(np.random.uniform(float(rotation_degrees[0]), float(rotation_degrees[1])))

    def apply_float_image_op(frame, op):
        arr = np.asarray(frame).astype(np.float32) / 255.0
        arr = op(arr)
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).round().astype(np.uint8))

    out = []
    for frame in pil_frames:
        for op in order:
            if op == "random_resized_crop" and crop_params is not None:
                i, j, h, w = crop_params
                frame = TVF.resized_crop(frame, i, j, h, w, size=(out_h, out_w))
            elif op == "random_brightness":
                frame = TVF.adjust_brightness(frame, brightness)
            elif op == "random_contrast":
                frame = TVF.adjust_contrast(frame, contrast)
            elif op == "random_saturation":
                frame = TVF.adjust_saturation(frame, saturation)
            elif op == "random_hue":
                frame = TVF.adjust_hue(frame, hue)
            elif op == "random_gamma":
                frame = apply_float_image_op(frame, lambda arr: np.power(arr, gamma))
            elif op == "random_exposure_ev":
                frame = apply_float_image_op(frame, lambda arr: arr * (2.0 ** exposure_ev))
            elif op == "gaussian_noise" and gaussian_noise_std > 0:
                frame = apply_float_image_op(
                    frame,
                    lambda arr: arr + np.random.normal(0.0, gaussian_noise_std, arr.shape).astype(np.float32),
                )
            elif op == "random_rotation" and rotation_degrees:
                fill = tuple(np.asarray(frame, dtype=np.uint8).reshape(-1, 3).mean(axis=0).round().astype(np.uint8).tolist())
                frame = TVF.rotate(
                    frame,
                    rotation_angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=fill,
                )
        out.append(np.asarray(frame, dtype=np.uint8))
    out = np.stack(out, axis=0)
    return out if is_video else out[0]


def _select_augmented_image_indices(augmentation, num_images):
    policy = augmentation.get("camera_aug_policy", "all")
    if policy in (None, "all"):
        return set(range(num_images))
    if policy != "third_wrist_quarters":
        raise ValueError(f"Unsupported camera_aug_policy: {policy}")

    third_camera_index = int(augmentation.get("third_camera_index", 0))
    wrist_camera_index = int(augmentation.get("wrist_camera_index", 1))
    selected_by_bucket = [
        [],
        [third_camera_index],
        [wrist_camera_index],
        [third_camera_index, wrist_camera_index],
    ]
    bucket = int(np.random.randint(4))
    return {idx for idx in selected_by_bucket[bucket] if 0 <= idx < num_images}


def _augment_example(example, augmentation):
    if not augmentation or not augmentation.get("enabled", False):
        return example["video"], example["image"]

    videos = np.asarray(example["video"])
    images = []
    augmented_indices = _select_augmented_image_indices(augmentation, len(example["image"]))
    for image_idx, image in enumerate(example["image"]):
        if image_idx in augmented_indices:
            aug_image = _augment_video_frames(np.asarray(image), augmentation)
            images.append(Image.fromarray(aug_image.astype(np.uint8)))
        else:
            images.append(image)
    return videos, images


def _get_vj_processor(processor_path):
    global _VJ_PROCESSOR, _VJ_PROCESSOR_PATH
    if _VJ_PROCESSOR is None or _VJ_PROCESSOR_PATH != processor_path:
        from transformers import AutoVideoProcessor

        _VJ_PROCESSOR = AutoVideoProcessor.from_pretrained(processor_path)
        _VJ_PROCESSOR_PATH = processor_path
    return _VJ_PROCESSOR


def _get_qwen_processor(model_path, action_tokens, embodied_action_token, future_tokens=None):
    global _QWEN_PROCESSOR, _QWEN_PROCESSOR_PATH
    if _QWEN_PROCESSOR is None or _QWEN_PROCESSOR_PATH != model_path:
        from transformers import AutoProcessor

        _QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_path)
        _QWEN_PROCESSOR.tokenizer.padding_side = "left"
        for token in action_tokens:
            if token not in _QWEN_PROCESSOR.tokenizer.get_vocab():
                _QWEN_PROCESSOR.tokenizer.add_tokens([token], special_tokens=True)
        for token in future_tokens or []:
            if token not in _QWEN_PROCESSOR.tokenizer.get_vocab():
                _QWEN_PROCESSOR.tokenizer.add_tokens([token], special_tokens=True)
        if embodied_action_token not in _QWEN_PROCESSOR.tokenizer.get_vocab():
            _QWEN_PROCESSOR.tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        _QWEN_PROCESSOR_PATH = model_path
    return _QWEN_PROCESSOR


def _process_vj_videos(processor, videos):
    try:
        return processor(videos=videos, return_tensors="pt")["pixel_values_videos"]
    except Exception:
        return torch.cat(
            [
                processor(videos=videos[i], return_tensors="pt")["pixel_values_videos"]
                for i in range(videos.shape[0])
            ],
            dim=0,
        )


def collate_fn(
    batch,
    vj_processor_path=None,
    preprocess_vj_inputs=False,
    qwen_processor_path=None,
    preprocess_qwen_inputs=False,
    qwen_prompt_template="",
    qwen_replace_prompt="",
    qwen_embodied_replace_prompt="",
    qwen_action_tokens=None,
    qwen_embodied_action_token="<|embodied_action|>",
    qwen_future_tokens=None,
    augmentation=None,
):
    if augmentation and augmentation.get("enabled", False):
        augmented = [_augment_example(example, augmentation) for example in batch]
        batch_images = [item[1] for item in augmented]
        videos_np = np.stack([item[0] for item in augmented])  # [B, V, T, H, W, C]
    else:
        batch_images = [example["image"] for example in batch]
        videos_np = np.stack([example["video"] for example in batch])  # [B, V, T, H, W, C]
    collated = {
        "image": batch_images,
        "lang": [example["lang"] for example in batch],
        "video": torch.from_numpy(videos_np),
        "action": torch.from_numpy(np.stack([example["action"] for example in batch])),
    }
    if "state" in batch[0]:
        collated["state"] = torch.from_numpy(np.stack([example["state"] for example in batch]))

    if preprocess_qwen_inputs:
        if not qwen_processor_path:
            raise ValueError("qwen_processor_path is required when preprocess_qwen_inputs=True")
        qwen_action_tokens = qwen_action_tokens or []
        processor = _get_qwen_processor(
            qwen_processor_path,
            qwen_action_tokens,
            qwen_embodied_action_token,
            future_tokens=qwen_future_tokens,
        )
        messages = []
        for imgs, instruction in zip(collated["image"], collated["lang"]):
            prompt = qwen_prompt_template.replace("{instruction}", instruction)
            prompt = prompt.replace("{actions}", qwen_replace_prompt)
            prompt = prompt.replace("{e_actions}", qwen_embodied_replace_prompt)
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": img} for img in imgs] + [{"type": "text", "text": prompt}],
                    }
                ]
            )
        collated["qwen_inputs"] = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    if preprocess_vj_inputs:
        if not vj_processor_path:
            raise ValueError("vj_processor_path is required when preprocess_vj_inputs=True")
        processor = _get_vj_processor(vj_processor_path)
        videos = videos_np.transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, C, H, W]
        B, V, T, C, H, W = videos.shape
        videos = videos.reshape(B * V, T, C, H, W)
        collated["vj_pixel_values_videos"] = _process_vj_videos(processor, videos)
        collated["video_shape"] = (B, V, T, C, H, W)
        collated["video"] = None

    return collated

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,  # 新增参数
    delete_pause_frame: bool = False,
    action_horizon: int = 7,
    video_horizon: int = 16,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    data_config_cls = ROBOT_TYPE_CONFIG_MAP[robot_type]
    data_config = data_config_cls(
        observation_indices=list(range(video_horizon)),
        action_indices=list(range(action_horizon))
    )
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = Path(data_name)
    if not dataset_path.is_absolute():
        dataset_path = Path(data_root_dir) / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="torchvision_av",
        delete_pause_frame=delete_pause_frame,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool = True,
    action_horizon: int = 7,
    video_horizon: int = 16,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), 
                                                          d_name, 
                                                          robot_type, 
                                                          delete_pause_frame=delete_pause_frame, 
                                                          action_horizon=action_horizon,
                                                          video_horizon=video_horizon), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        with_state=data_cfg.get("with_state", False),
        resolution_size=data_cfg.get("resolution_size", 224),
        video_resolution_size=data_cfg.get("video_resolution_size", 256),
        seed=seed,
        **kwargs,
    )

if __name__ == "__main__":
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    from tqdm import tqdm
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        pass
