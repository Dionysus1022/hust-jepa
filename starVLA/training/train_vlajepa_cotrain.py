from __future__ import annotations
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np

# Third-Party Libraries
import torch
import torch.distributed as dist
#import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

def create_accelerator(cfg) -> Accelerator:
    grad_accum_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    grad_accum_plugin = GradientAccumulationPlugin(
        num_steps=grad_accum_steps,
        sync_each_batch=False,
    )
    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=grad_accum_plugin,
    )
    accelerator.print(accelerator.state)
    accelerator.print(f"Using gradient_accumulation_steps={accelerator.gradient_accumulation_steps}")
    return accelerator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    video_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.video_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader, video_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and learning rate scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group information
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, video_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.video_train_dataloader = video_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.run_root_dir, cfg.run_id, "tensorboard"))  # 保存目录


        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def _move_batch_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.accelerator.device, non_blocking=True)
        if batch.__class__.__name__ == "BatchFeature":
            return batch.to(self.accelerator.device)
        if isinstance(batch, dict):
            has_preprocessed_vj = "vj_pixel_values_videos" in batch
            has_preprocessed_qwen = "qwen_inputs" in batch
            return {
                k: v
                if (has_preprocessed_vj and k == "video") or (has_preprocessed_qwen and k in {"image", "lang"})
                else self._move_batch_to_device(v)
                for k, v in batch.items()
            }
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(v) for v in batch)
        if isinstance(batch, list):
            return [self._move_batch_to_device(v) for v in batch]
        return batch

    def _as_collated_dict(self, batch):
        if isinstance(batch, dict):
            return batch
        videos_np = np.stack([example["video"] for example in batch])
        collated = {
            "image": [example["image"] for example in batch],
            "lang": [example["lang"] for example in batch],
            "video": torch.from_numpy(videos_np),
        }
        if "action" in batch[0]:
            collated["action"] = torch.from_numpy(np.stack([example["action"] for example in batch]))
        if "state" in batch[0]:
            collated["state"] = torch.from_numpy(np.stack([example["state"] for example in batch]))
        return collated

    def _get_qwen_pad_token_id(self):
        model = getattr(self.model, "module", self.model)
        qwen = getattr(model, "qwen_vl_interface", None)
        processor = getattr(qwen, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        return 0 if pad_token_id is None else pad_token_id

    def _concat_preprocessed_qwen_inputs(self, qwen_inputs_list):
        qwen_inputs_list = [item for item in qwen_inputs_list if item is not None]
        if len(qwen_inputs_list) <= 1:
            return qwen_inputs_list[0] if qwen_inputs_list else None

        batch_sizes = [inputs["input_ids"].shape[0] for inputs in qwen_inputs_list]
        max_seq_len = max(inputs["input_ids"].shape[1] for inputs in qwen_inputs_list)
        pad_token_id = self._get_qwen_pad_token_id()

        merged = {}
        keys = set().union(*(inputs.keys() for inputs in qwen_inputs_list))
        for key in keys:
            values = [inputs[key] for inputs in qwen_inputs_list if key in inputs]
            if not values or not torch.is_tensor(values[0]):
                continue

            if values[0].dim() == 2 and all(v.shape[0] == b for v, b in zip(values, batch_sizes)):
                pad_value = pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
                padded_values = []
                for value in values:
                    pad_len = max_seq_len - value.shape[1]
                    if pad_len > 0:
                        pad = value.new_full((value.shape[0], pad_len), pad_value)
                        value = torch.cat([pad, value], dim=1)
                    padded_values.append(value)
                merged[key] = torch.cat(padded_values, dim=0)
            else:
                merged[key] = torch.cat(values, dim=0)
        return merged

    def _merge_cotrain_batches(self, batch_vla, batch_vlm):
        batch_vla = self._as_collated_dict(batch_vla)
        batch_vlm = self._as_collated_dict(batch_vlm)
        vla_video = batch_vla["video"]
        vlm_video = batch_vlm["video"]

        mixed = {
            "image": batch_vla["image"] + batch_vlm["image"],
            "lang": batch_vla["lang"] + batch_vlm["lang"],
            "video": torch.cat([vla_video, vlm_video], dim=0),
            "action": batch_vla["action"],
            "vla_batch_size": len(batch_vla["lang"]),
            "vlm_batch_size": len(batch_vlm["lang"]),
        }
        if "state" in batch_vla:
            mixed["state"] = batch_vla["state"]
        qwen_inputs = [batch.get("qwen_inputs", None) for batch in (batch_vla, batch_vlm)]
        if any(item is not None for item in qwen_inputs):
            mixed["qwen_inputs"] = (
                self._concat_preprocessed_qwen_inputs(qwen_inputs)
                if all(item is not None for item in qwen_inputs)
                else qwen_inputs
            )
        vj_inputs = [batch.get("vj_pixel_values_videos", None) for batch in (batch_vla, batch_vlm)]
        if any(item is not None for item in vj_inputs):
            mixed["vj_pixel_values_videos"] = vj_inputs
        return mixed

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            ignore_mismatched_sizes = bool(
                getattr(self.config.trainer, "ignore_mismatched_pretrained", False)
            )
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print trainable parameters of the model
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader, self.video_train_dataloader = (
            self.setup_distributed_training(
                self.accelerator, self.model, self.optimizer, self.vla_train_dataloader, self.video_train_dataloader
            )
        )

        #self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # resume training state
        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if (
            self.completed_steps % self.config.trainer.logging_frequency == 0
        ):  # some parameters should be initialized for the class
            if dist.get_rank() == 0:
                # calculate gradient norm
                # total_norm = 0.0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         total_norm += p.grad.data.norm(2).item() ** 2
                # metrics["grad_norm"] = total_norm ** 0.5

                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch information
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                #wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.video_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            # check if there is self.vla_epoch_count
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        try:
            batch_vlm = next(self.vlm_iter)
        except StopIteration:
            if not hasattr(self, "vlm_epoch_count"):
                self.vlm_epoch_count = 0
            self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(self.video_train_dataloader, self.vlm_epoch_count)
            batch_vlm = next(self.vlm_iter)

        return batch_vla, batch_vlm

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t_end_fetch = time.perf_counter()
            t_start_h2d = time.perf_counter()
            batch_mixed = self._merge_cotrain_batches(batch_vla, batch_vlm)
            batch_mixed = self._move_batch_to_device(batch_mixed)
            if self.config.trainer.get("enable_detailed_timing", False) and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_mixed)
            t_end_model = time.perf_counter()

            # update progress
            is_update_step = step_metrics.pop("is_update_step", True)
            if is_update_step:
                self.lr_scheduler.step()
                progress_bar.update(1)
                self.completed_steps += 1

            if is_update_step:
                # evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                # record metrics
                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["data_fetch_time"] = t_end_fetch - t_start_data
                step_metrics["batch_h2d_time"] = t_end_data - t_start_h2d
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                # save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

                    dist.barrier()  # ensure all processes are synchronized, avoid timeout

            # check termination condition
            if self.accelerator.is_local_main_process:
                postfix = {
                    "data_times": f"{t_end_data - t_start_data:.3f}",
                    "model_times": f"{t_end_model - t_start_model:.3f}",
                }
                if self.config.trainer.get("enable_detailed_timing", False):
                    postfix.update(
                        {
                            "mix_fwd": f"{step_metrics.get('timing/mixed_forward_time', 0.0):.3f}",
                            "bwd": f"{step_metrics.get('timing/backward_grad_sync_time', 0.0):.3f}",
                            "ds_step": f"{step_metrics.get('timing/deepspeed_step_time', 0.0):.3f}",
                            "h2d": f"{t_end_data - t_start_h2d:.3f}",
                        }
                    )
                progress_bar.set_postfix(postfix)

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step
    
    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if self.accelerator.is_main_process:

            examples, vlm_data = self._get_next_batch()

            score = 0.0
            if isinstance(examples, dict):
                batch_images = examples["image"]
                instructions = examples["lang"]
                actions = examples["action"].detach().cpu().numpy() if torch.is_tensor(examples["action"]) else examples["action"]
            else:
                batch_images = [example["image"] for example in examples]
                instructions = [example["lang"] for example in examples]  # [B, str]
                actions = [example["action"] for example in examples]  # label
            num_samples = len(instructions)

            # Predict actions using the model
            output_dict = self.model.predict_action(
                batch_images=batch_images, instructions=instructions, use_ddim=True, num_ddim_steps=20
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            mae_score = np.mean(np.abs(normalized_actions - actions))

            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score
            step_metrics["mae_score"] = mae_score
            self.writer.add_scalar("mae_score", step_metrics["mae_score"], self.completed_steps)
            self.writer.add_scalar("mse_score", step_metrics["mse_score"], self.completed_steps)
        
        pass
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f" Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_mixed):
        """execute single training step"""
        log_dict = {}
        detailed_timing = bool(self.config.trainer.get("enable_detailed_timing", False))
        timing = {}

        def sync_cuda():
            if detailed_timing and torch.cuda.is_available():
                torch.cuda.synchronize()

        def timed(name, fn):
            if not detailed_timing:
                return fn()
            sync_cuda()
            start = time.perf_counter()
            result = fn()
            sync_cuda()
            timing[name] = time.perf_counter() - start
            return result

        def get_forward_timing(prefix):
            if not detailed_timing:
                return {}
            wrapped_model = getattr(self.model, "module", self.model)
            return {
                f"{prefix}/{k}": v
                for k, v in getattr(wrapped_model, "last_forward_timing", {}).items()
            }

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = timed("timing/mixed_forward_time", lambda: self.model.forward(batch_mixed))
            mixed_forward_timing = get_forward_timing("mixed")

        total_loss = timed("timing/loss_sum_time", lambda: sum(output_dict.values()))

        is_update_step = True
        if hasattr(self.model, "backward") and hasattr(self.model, "step"):
            timed("timing/backward_grad_sync_time", lambda: self.model.backward(total_loss))
            if hasattr(self.model, "is_gradient_accumulation_boundary"):
                is_update_step = bool(self.model.is_gradient_accumulation_boundary())
            timed("timing/deepspeed_step_time", self.model.step)
        else:
            with self.accelerator.accumulate(self.model):
                timed("timing/backward_grad_sync_time", lambda: self.accelerator.backward(total_loss))
                if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                timed("timing/optimizer_step_time", self.optimizer.step)
                self.optimizer.zero_grad()
                is_update_step = bool(self.accelerator.sync_gradients)

        for k, v in output_dict.items():
            log_dict[k] = v.item()
        log_dict["loss"] = total_loss.item()
        log_dict["is_update_step"] = is_update_step
        if detailed_timing:
            log_dict.update(mixed_forward_timing)
            log_dict.update(timing)
        return log_dict

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        #if self.accelerator.is_main_process:
        #    wandb.finish()

        self.accelerator.wait_for_everyone()


from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


def main(cfg, accelerator) -> None:
    logger.info("VLA Training :: Warming Up")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)

    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader, video_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        video_train_dataloader=video_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print(
            "🔍 Rank 0 waiting for debugger attach on port 10092..."
        )  # you may ask chatGPT what is debugger attach in vscode
        debugpy.wait_for_client()

    accelerator = create_accelerator(cfg)
    main(cfg, accelerator)
