from __future__ import annotations
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf
import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

_VJ_PROCESSOR = None
_VJ_PROCESSOR_PATH = None
_QWEN_PROCESSOR = None
_QWEN_PROCESSOR_PATH = None


def _get_vj_processor(processor_path):
    global _VJ_PROCESSOR, _VJ_PROCESSOR_PATH
    if _VJ_PROCESSOR is None or _VJ_PROCESSOR_PATH != processor_path:
        from transformers import AutoVideoProcessor

        _VJ_PROCESSOR = AutoVideoProcessor.from_pretrained(processor_path)
        _VJ_PROCESSOR_PATH = processor_path
    return _VJ_PROCESSOR


def _get_qwen_processor(model_path, action_tokens, embodied_action_token):
    global _QWEN_PROCESSOR, _QWEN_PROCESSOR_PATH
    if _QWEN_PROCESSOR is None or _QWEN_PROCESSOR_PATH != model_path:
        from transformers import AutoProcessor

        _QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_path)
        _QWEN_PROCESSOR.tokenizer.padding_side = "left"
        for token in action_tokens:
            if token not in _QWEN_PROCESSOR.tokenizer.get_vocab():
                _QWEN_PROCESSOR.tokenizer.add_tokens([token], special_tokens=True)
        if embodied_action_token not in _QWEN_PROCESSOR.tokenizer.get_vocab():
            _QWEN_PROCESSOR.tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        _QWEN_PROCESSOR_PATH = model_path
    return _QWEN_PROCESSOR


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
):
    videos_np = np.stack([example["video"] for example in batch])  # [B, V, T, H, W, C]
    collated = {
        "image": [example["image"] for example in batch],
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
        processor = _get_qwen_processor(qwen_processor_path, qwen_action_tokens, qwen_embodied_action_token)
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
        processed = [
            processor(videos=videos[i], return_tensors="pt")["pixel_values_videos"]
            for i in range(B * V)
        ]
        collated["vj_pixel_values_videos"] = torch.cat(processed, dim=0)

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
    dataset_path = data_root_dir / data_name
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
