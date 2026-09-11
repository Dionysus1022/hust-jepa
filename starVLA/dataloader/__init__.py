from __future__ import annotations
import json
import os
from accelerate.logging import get_logger
from functools import partial
import torch
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def _worker_init_fn(worker_id):
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass


def _dataloader_kwargs(data_cfg, default_num_workers):
    num_workers = int(data_cfg.get("num_workers", default_num_workers))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
        kwargs["worker_init_fn"] = _worker_init_fn
    logger.info(f"DataLoader kwargs: {kwargs}")
    return kwargs


def _build_jepa_prompt_tokens(cfg, tubelet_size):
    action_tokens = [
        cfg.framework.vj2_model.special_action_token.format(i)
        for i in range(cfg.framework.action_model.action_horizon * 4)
    ]
    num_future_steps = cfg.framework.vj2_model.num_frames // tubelet_size - 1
    use_future_tokens = bool(cfg.framework.vj2_model.get("enable_future_tokens", False))
    future_special_token = cfg.framework.vj2_model.get("special_future_token", "<|future_{}|>")
    future_tokens = [future_special_token.format(i + 1) for i in range(num_future_steps)]
    if use_future_tokens:
        replace_prompt = "".join(
            action_tokens[i] * cfg.framework.vj2_model.num_action_tokens_per_timestep + future_tokens[i]
            for i in range(num_future_steps)
        )
    else:
        replace_prompt = "".join(
            action_tokens[i] * cfg.framework.vj2_model.num_action_tokens_per_timestep
            for i in range(num_future_steps)
        )
    return action_tokens, future_tokens if use_future_tokens else [], replace_prompt

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        from transformers import AutoConfig

        vla_dataset_cfg = cfg.datasets.vla_data
        vj_config = AutoConfig.from_pretrained(cfg.framework.vj2_model.base_encoder)
        tubelet_size = getattr(vj_config, "tubelet_size", 1)
        action_tokens, future_tokens, replace_prompt = _build_jepa_prompt_tokens(cfg, tubelet_size)
        embodied_action_token = cfg.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        embodied_replace_prompt = (
            embodied_action_token * cfg.framework.vj2_model.num_embodied_action_tokens_per_instruction
        )
        custom_collate_fn = partial(
            collate_fn,
            vj_processor_path=cfg.framework.vj2_model.base_encoder,
            preprocess_vj_inputs=vla_dataset_cfg.get("preprocess_vj_inputs_in_collate", True),
            qwen_processor_path=cfg.framework.qwenvl.base_vlm,
            preprocess_qwen_inputs=vla_dataset_cfg.get("preprocess_qwen_inputs_in_collate", False),
            qwen_prompt_template=vla_dataset_cfg.get("CoT_prompt", ""),
            qwen_replace_prompt=replace_prompt,
            qwen_embodied_replace_prompt=embodied_replace_prompt,
            qwen_action_tokens=action_tokens,
            qwen_embodied_action_token=embodied_action_token,
            qwen_future_tokens=future_tokens,
            augmentation=vla_dataset_cfg.get("augmentation", None),
        )

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            delete_pause_frame=bool(vla_dataset_cfg.get("delete_pause_frame", True)),
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames)

        shuffle = bool(vla_dataset_cfg.get("shuffle", False))
        dataloader_generator = None
        if shuffle:
            dataloader_generator = torch.Generator()
            dataloader_generator.manual_seed(int(cfg.seed))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=custom_collate_fn,
            shuffle=shuffle,
            generator=dataloader_generator,
            **_dataloader_kwargs(vla_dataset_cfg, default_num_workers=8),
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
    elif dataset_py == "lerobot_v3_datasets":
        from starVLA.dataloader.lerobot_v3_datasets import get_lerobot_v3_datasets, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_lerobot_v3_datasets(data_cfg=vla_dataset_cfg)

        custom_collate_fn = partial(collate_fn, 
            img_keys=cfg.datasets.vla_data.img_keys,
            state_key=cfg.datasets.vla_data.state_key if "state_key" in cfg.datasets.vla_data else None,
            action_key=cfg.datasets.vla_data.action_key if cfg.datasets.vla_data.action_key else None,
            task_key=cfg.datasets.vla_data.task_key if cfg.datasets.vla_data.task_key else None,
            resize_size=cfg.datasets.vla_data.resize_size)



        train_sampler = torch.utils.data.distributed.DistributedSampler(vla_dataset, shuffle=True)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=custom_collate_fn,
            sampler=train_sampler,
            **_dataloader_kwargs(vla_dataset_cfg, default_num_workers=16),
        )      
        #if dist.get_rank() == 0: 
        #    for batch in vla_train_dataloader:
        #        print(batch)
        #        for k, v in batch.items():
        #            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        #        break
        return vla_train_dataloader
    elif dataset_py == "video_datasets":
        from starVLA.dataloader.video_datasets import VideoFolderDataset, collate_fn
        from transformers import AutoConfig

        video_dataset_cfg = cfg.datasets.video_data
        vj_config = AutoConfig.from_pretrained(cfg.framework.vj2_model.base_encoder)
        tubelet_size = getattr(vj_config, "tubelet_size", 1)
        action_tokens, future_tokens, replace_prompt = _build_jepa_prompt_tokens(cfg, tubelet_size)
        embodied_action_token = cfg.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")

        video_dataset = VideoFolderDataset(
            video_dir=video_dataset_cfg.video_dir,
            text_file=video_dataset_cfg.text_file,
            n_frames=cfg.framework.vj2_model.num_frames,
            extensions=tuple(video_dataset_cfg.extensions),
            crop_h_size=video_dataset_cfg.video_resolution_size,
            crop_w_size=video_dataset_cfg.video_resolution_size,
            max_retry=10,
            decode_threads=video_dataset_cfg.get("decode_threads", 1),
        )

        video_collate_fn = partial(collate_fn, 
            n_views=2,
            resolution_size=video_dataset_cfg.resolution_size,
            vj_processor_path=cfg.framework.vj2_model.base_encoder,
            preprocess_vj_inputs=video_dataset_cfg.get("preprocess_vj_inputs_in_collate", False),
            qwen_processor_path=cfg.framework.qwenvl.base_vlm,
            preprocess_qwen_inputs=video_dataset_cfg.get("preprocess_qwen_inputs_in_collate", False),
            qwen_prompt_template=video_dataset_cfg.get("CoT_prompt", ""),
            qwen_replace_prompt=replace_prompt,
            qwen_action_tokens=action_tokens,
            qwen_embodied_action_token=embodied_action_token,
            qwen_future_tokens=future_tokens)

        # Accelerator shards the prepared DataLoader across processes.  Adding a
        # DistributedSampler here would shard SSV once before Accelerator shards
        # it again, so a 4-rank job would expose only about 1/4 of the dataset per
        # logical epoch.
        shuffle = bool(video_dataset_cfg.get("shuffle", True))
        dataloader_generator = None
        if shuffle:
            dataloader_generator = torch.Generator()
            dataloader_generator.manual_seed(int(cfg.seed))

        video_train_dataloader = DataLoader(
            video_dataset,
            batch_size=video_dataset_cfg.per_device_batch_size,
            collate_fn=video_collate_fn,
            shuffle=shuffle,
            generator=dataloader_generator,
            **_dataloader_kwargs(video_dataset_cfg, default_num_workers=16),
        )        
        return video_train_dataloader
