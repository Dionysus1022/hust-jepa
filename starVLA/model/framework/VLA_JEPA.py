from __future__ import annotations
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
import time
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoConfig, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


class SoftQueryConnector(nn.Module):
    def __init__(self, hidden_size: int, depth: int = 2, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.norm(self.decoder(query, context))

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        trainer_cfg = getattr(self.config, "trainer", {})
        self.enable_gradient_checkpointing = bool(trainer_cfg.get("enable_gradient_checkpointing", False))
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        self.use_future_tokens = bool(self.config.framework.vj2_model.get("enable_future_tokens", False))
        future_special_token = self.config.framework.vj2_model.get("special_future_token", "<|future_{}|>")
        vj_encoder_config = AutoConfig.from_pretrained(self.config.framework.vj2_model.base_encoder)
        tubelet_size = vj_encoder_config.tubelet_size
        self.num_future_tokens = self.config.framework.vj2_model.num_frames // tubelet_size - 1
        action_tokens, self.action_token_ids, self.embodied_action_token_id, self.future_token_ids = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token,
            special_future_token=future_special_token,
            num_future_tokens=self.num_future_tokens if self.use_future_tokens else 0,
        )
        self.register_buffer(
            "_action_token_ids_tensor",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_embodied_action_token_id_tensor",
            torch.tensor([self.embodied_action_token_id], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_future_token_ids_tensor",
            torch.tensor(self.future_token_ids or [-1], dtype=torch.long),
            persistent=False,
        )
        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用
        if self.enable_gradient_checkpointing and hasattr(getattr(self.action_model, "model", None), "gradient_checkpointing"):
            self.action_model.model.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing for action DiT")

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_encoder.requires_grad_(False)
        self.vj_encoder.eval()
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
            use_activation_checkpointing=self.enable_gradient_checkpointing,
        )
        self.future_projector = None
        if self.use_future_tokens:
            self.future_projector = nn.Linear(
                self.qwen_vl_interface.model.config.hidden_size,
                self.vj_encoder.config.hidden_size * 2,
            )
        vlanext_cfg = self.config.framework.get("vlanext_conditioning", {})
        vl_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.use_proprio_input_vlm = bool(vlanext_cfg.get("use_proprio_input_vlm", False))
        self.use_soft_connector = bool(vlanext_cfg.get("use_soft_connector", False))
        self.action_dct_loss_weight = float(vlanext_cfg.get("dct_loss_weight", 0.0))
        self.dct_freq_split = float(vlanext_cfg.get("dct_freq_split", 0.125))
        self.dct_low_freq_weight = float(vlanext_cfg.get("dct_low_freq_weight", 5.0))
        self.dct_high_freq_weight = float(vlanext_cfg.get("dct_high_freq_weight", 0.0))
        self.proprio_projector = None
        if self.use_proprio_input_vlm:
            self.proprio_projector = nn.Sequential(
                nn.Linear(self.config.framework.action_model.state_dim, vl_hidden_size),
                nn.LayerNorm(vl_hidden_size),
                nn.SiLU(),
                nn.Linear(vl_hidden_size, vl_hidden_size),
            )
        self.soft_connector = None
        if self.use_soft_connector:
            self.soft_connector = SoftQueryConnector(
                hidden_size=vl_hidden_size,
                depth=int(vlanext_cfg.get("connector_depth", 2)),
                num_heads=int(vlanext_cfg.get("connector_num_heads", 4)),
                mlp_ratio=float(vlanext_cfg.get("connector_mlp_ratio", 4.0)),
            )
        self._maybe_compile_modules()
        future_tokens = [future_special_token.format(i + 1) for i in range(self.num_future_tokens)]
        if self.use_future_tokens:
            self.replace_prompt = "".join(
                action_tokens[i] * self.config.framework.vj2_model.num_action_tokens_per_timestep + future_tokens[i]
                for i in range(self.num_future_tokens)
            )
        else:
            self.replace_prompt = "".join(
                [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
                 action_tokens[:self.num_future_tokens]]
            )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])
        logger.info(
            f"Qwen prompt placeholders: action_chars={len(self.replace_prompt)}, "
            f"embodied_chars={len(self.embodied_replace_prompt)}, "
            f"num_action_token_ids={len(self.action_token_ids)}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "vj_encoder"):
            self.vj_encoder.eval()
        return self

    def _compute_dct_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, T, D = pred.shape
        n = torch.arange(T, device=pred.device, dtype=pred.dtype)
        k = torch.arange(T, device=pred.device, dtype=pred.dtype)
        dct = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
        dct[0, :] *= 1.0 / np.sqrt(T)
        if T > 1:
            dct[1:, :] *= np.sqrt(2.0 / T)

        split_idx = max(1, int(T * self.dct_freq_split))
        weights = torch.ones(T, device=pred.device, dtype=pred.dtype)
        weights[:split_idx] = self.dct_low_freq_weight
        weights[split_idx:] = self.dct_high_freq_weight
        weights = weights.view(1, T, 1)

        pred_dct = torch.matmul(pred.permute(0, 2, 1), dct.t()).permute(0, 2, 1)
        target_dct = torch.matmul(target.permute(0, 2, 1), dct.t()).permute(0, 2, 1)
        return (((pred_dct - target_dct) ** 2) * weights).mean()

    def _maybe_compile_modules(self):
        """Compile stable compute-heavy submodules without touching dynamic HF processors."""
        trainer_cfg = getattr(self.config, "trainer", {})
        if not trainer_cfg.get("enable_torch_compile", False):
            return
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile is not available in this PyTorch build; skipping compile.")
            return

        mode = trainer_cfg.get("torch_compile_mode", "reduce-overhead")
        backend = trainer_cfg.get("torch_compile_backend", "inductor")
        fullgraph = bool(trainer_cfg.get("torch_compile_fullgraph", True))
        dynamic = bool(trainer_cfg.get("torch_compile_dynamic", False))
        compile_threads = int(trainer_cfg.get("torch_compile_threads", 4))
        suppress_errors = bool(trainer_cfg.get("torch_compile_suppress_errors", True))
        try:
            import torch._dynamo.config as dynamo_config
            import torch._inductor.config as inductor_config

            dynamo_config.suppress_errors = suppress_errors
            inductor_config.compile_threads = compile_threads
        except Exception as exc:
            logger.warning(f"Could not configure torch.compile options: {exc}")
        modules = trainer_cfg.get("torch_compile_modules", ["vj_predictor", "action_dit"])
        logger.info(
            f"Enabling torch.compile for modules={list(modules)}, backend={backend}, mode={mode}, "
            f"fullgraph={fullgraph}, dynamic={dynamic}, compile_threads={compile_threads}, "
            f"suppress_errors={suppress_errors}"
        )

        if "vj_predictor" in modules:
            self.vj_predictor = torch.compile(
                self.vj_predictor,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
        if "action_dit_blocks" in modules and hasattr(self.action_model, "model"):
            transformer_blocks = getattr(self.action_model.model, "transformer_blocks", None)
            if transformer_blocks is None:
                logger.warning("Requested action_dit_blocks compile, but action_model.model has no transformer_blocks.")
            else:
                for idx, block in enumerate(transformer_blocks):
                    transformer_blocks[idx] = torch.compile(
                        block,
                        backend=backend,
                        mode=mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
        if "action_dit_inner" in modules and hasattr(self.action_model, "model"):
            transformer_blocks = getattr(self.action_model.model, "transformer_blocks", None)
            if transformer_blocks is None:
                logger.warning("Requested action_dit_inner compile, but action_model.model has no transformer_blocks.")
            else:
                for block in transformer_blocks:
                    if hasattr(block, "attn1"):
                        block.attn1 = torch.compile(
                            block.attn1,
                            backend=backend,
                            mode=mode,
                            fullgraph=fullgraph,
                            dynamic=dynamic,
                        )
                    if hasattr(block, "ff"):
                        block.ff = torch.compile(
                            block.ff,
                            backend=backend,
                            mode=mode,
                            fullgraph=fullgraph,
                            dynamic=dynamic,
                        )
        if "action_dit" in modules and hasattr(self.action_model, "model"):
            self.action_model.model = torch.compile(
                self.action_model.model,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>",
                         special_future_token: str = "<|future_{}|>",
                         num_future_tokens: int = 0):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        future_token_ids = []
        for i in range(num_future_tokens):
            future_token = special_future_token.format(i + 1)
            if future_token not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([future_token], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) future_token: {future_token}.")
            future_token_ids.append(tokenizer.convert_tokens_to_ids(future_token))

        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id, future_token_ids

    def _concat_qwen_inputs(self, qwen_inputs_list):
        batch_sizes = [inputs["input_ids"].shape[0] for inputs in qwen_inputs_list]
        max_seq_len = max(inputs["input_ids"].shape[1] for inputs in qwen_inputs_list)
        pad_token_id = self.qwen_vl_interface.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        merged = {}
        keys = set().union(*(inputs.keys() for inputs in qwen_inputs_list))
        for key in keys:
            values = [inputs[key] for inputs in qwen_inputs_list if key in inputs]
            if not values or not torch.is_tensor(values[0]):
                continue
            if values[0].dim() == 2 and all(v.shape[0] == b for v, b in zip(values, batch_sizes)):
                pad_value = 0
                if key == "input_ids":
                    pad_value = pad_token_id
                elif key == "labels":
                    pad_value = IGNORE_INDEX
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

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
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

        is_collated_batch = isinstance(examples, dict)
        if is_collated_batch:
            batch_images = examples["image"]
            batch_videos = examples["video"]
            video_shape = examples.get("video_shape", None)
            instructions = examples["lang"]
            actions = examples.get("action", None)
            state = examples.get("state", None)
            vj_pixel_values_videos = examples.get("vj_pixel_values_videos", None)
            qwen_inputs = examples.get("qwen_inputs", None)
            vla_batch_size = examples.get("vla_batch_size", None)
        else:
            batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
            batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
            state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
            vj_pixel_values_videos = None
            qwen_inputs = None
            vla_batch_size = None
            video_shape = None
        is_mixed_cotrain_batch = vla_batch_size is not None

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        def stack_videos():
            if batch_videos is None:
                if video_shape is None:
                    raise RuntimeError("Missing raw video and video_shape for VLA_JEPA forward.")
                return None
            if torch.is_tensor(batch_videos):
                return batch_videos.permute(0, 1, 2, 5, 3, 4).contiguous()
            stacked = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
            return stacked.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        batch_videos = timed("forward/video_numpy_stack_time", stack_videos)
        if batch_videos is not None:
            video_shape = tuple(batch_videos.shape)
        else:
            video_shape = tuple(int(x) for x in video_shape)

        # Step 1: QWenVL input format
        def build_qwen_inputs():
            if qwen_inputs is not None and not is_mixed_cotrain_batch:
                return qwen_inputs
            if is_mixed_cotrain_batch:
                if (
                    qwen_inputs is not None
                    and not isinstance(qwen_inputs, list)
                    and "input_ids" in qwen_inputs
                    and qwen_inputs["input_ids"].shape[0] == len(instructions)
                ):
                    return qwen_inputs
                vla_bs = int(vla_batch_size)
                qwen_input_parts = qwen_inputs if isinstance(qwen_inputs, list) else [qwen_inputs, None]
                qwen_parts = []
                if qwen_input_parts[0] is not None:
                    qwen_parts.append(qwen_input_parts[0])
                else:
                    qwen_parts.append(
                        self.qwen_vl_interface.build_qwenvl_inputs(
                            images=batch_images[:vla_bs],
                            instructions=instructions[:vla_bs],
                            prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
                            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
                        )
                    )
                if qwen_input_parts[1] is not None:
                    qwen_parts.append(qwen_input_parts[1])
                else:
                    qwen_parts.append(
                        self.qwen_vl_interface.build_qwenvl_inputs(
                            images=batch_images[vla_bs:],
                            instructions=instructions[vla_bs:],
                            prompt_replace_dict={"{actions}": self.replace_prompt},
                            prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""),
                        )
                    )
                return self._concat_qwen_inputs(qwen_parts)
            if actions is not None:
                return self.qwen_vl_interface.build_qwenvl_inputs(
                    images=batch_images,
                    instructions=instructions,
                    prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                    prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""))
            return self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))

        qwen_inputs = timed("forward/qwen_build_inputs_h2d_time", build_qwen_inputs)
        if detailed_timing:
            input_ids = qwen_inputs["input_ids"]
            attention_mask = qwen_inputs.get("attention_mask", None)
            timing["forward/qwen_seq_len"] = float(input_ids.shape[1])
            if attention_mask is not None:
                valid_tokens = attention_mask.sum(dim=1).float()
                timing["forward/qwen_valid_tokens_mean"] = valid_tokens.mean().item()
                timing["forward/qwen_valid_tokens_max"] = valid_tokens.max().item()
                timing["forward/qwen_padding_ratio"] = (
                    1.0 - valid_tokens.mean().item() / max(float(input_ids.shape[1]), 1.0)
                )
            image_grid_thw = qwen_inputs.get("image_grid_thw", None)
            if image_grid_thw is not None:
                grid_tokens = image_grid_thw.prod(dim=1).float()
                timing["forward/qwen_num_images"] = float(image_grid_thw.shape[0])
                timing["forward/qwen_image_tokens_total"] = grid_tokens.sum().item()
                timing["forward/qwen_image_tokens_per_sample"] = grid_tokens.sum().item() / max(float(input_ids.shape[0]), 1.0)

        action_indices = torch.isin(
            qwen_inputs['input_ids'],
            self._action_token_ids_tensor.to(qwen_inputs['input_ids'].device),
        )
        action_indices = action_indices.nonzero(as_tuple=True)
        if detailed_timing:
            timing["forward/qwen_action_token_count"] = float(action_indices[0].numel())

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(
            qwen_inputs['input_ids'],
            self._embodied_action_token_id_tensor.to(qwen_inputs['input_ids'].device),
        )
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        if detailed_timing:
            timing["forward/qwen_embodied_token_count"] = float(embodied_action_indices[0].numel())

        future_indices = None
        if self.use_future_tokens:
            future_indices = torch.isin(
                qwen_inputs["input_ids"],
                self._future_token_ids_tensor.to(qwen_inputs["input_ids"].device),
            )
            future_indices = future_indices.nonzero(as_tuple=True)
            if detailed_timing:
                timing["forward/qwen_future_token_count"] = float(future_indices[0].numel())
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            def qwen_forward():
                return self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )

            qwenvl_outputs = timed("forward/qwen_forward_time", qwen_forward)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = getattr(qwenvl_outputs, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = qwenvl_outputs.hidden_states[-1]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_batch_size = int(vla_batch_size) if is_mixed_cotrain_batch else B
            embodied_action_tokens = last_hidden[
                embodied_action_indices[0], embodied_action_indices[1], :
            ].view(embodied_batch_size, -1, H)  # [B_vla, action_len, H]
            state_tensor = None
            proprio_condition_tokens = None
            if self.use_proprio_input_vlm and self.proprio_projector is not None and state is not None:
                if torch.is_tensor(state):
                    state_tensor = state.to(last_hidden.device, dtype=last_hidden.dtype, non_blocking=True)
                else:
                    state_tensor = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                if state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(1)
                proprio_condition_tokens = timed(
                    "forward/proprio_projector_time",
                    lambda: self.proprio_projector(state_tensor),
                )

            if self.use_soft_connector and self.soft_connector is not None:
                soft_context = last_hidden[:embodied_batch_size]
                if proprio_condition_tokens is not None:
                    soft_context = torch.cat([proprio_condition_tokens, soft_context], dim=1)
                embodied_action_tokens = timed(
                    "forward/soft_connector_time",
                    lambda: self.soft_connector(embodied_action_tokens, soft_context),
                )

            action_condition_parts = []
            if proprio_condition_tokens is not None:
                action_condition_parts.append(proprio_condition_tokens)
            action_condition_parts.append(embodied_action_tokens)
            action_condition_tokens = torch.cat(action_condition_parts, dim=1)
            future_token_states = None
            if self.use_future_tokens and future_indices is not None:
                expected_future_tokens = B * self.num_future_tokens
                actual_future_tokens = future_indices[0].numel()
                if actual_future_tokens != expected_future_tokens:
                    raise RuntimeError(
                        f"Expected {expected_future_tokens} future tokens "
                        f"({B} batch * {self.num_future_tokens} tokens), got {actual_future_tokens}. "
                        "Check that dataloader and model add special tokens in the same order."
                    )
                future_token_states = last_hidden[
                    future_indices[0], future_indices[1], :
                ].view(B, self.num_future_tokens, H)
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = video_shape
            if batch_videos is not None:
                batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]

            def prepare_vj_videos():
                expected_videos = B * V
                if isinstance(vj_pixel_values_videos, list):
                    ready_videos = [
                        item.to(self.vj_encoder.device, non_blocking=True)
                        for item in vj_pixel_values_videos
                        if item is not None
                    ]
                    ready_count = sum(item.shape[0] for item in ready_videos)
                    if ready_count == expected_videos:
                        return torch.cat(ready_videos, dim=0)
                    if batch_videos is None:
                        raise RuntimeError(
                            "Raw videos are required when only part of vj_pixel_values_videos is preprocessed."
                        )
                    input_videos = ready_videos
                    for i in range(ready_count, expected_videos):
                        input_videos.append(self.vj_processor(
                            videos=batch_videos[i], return_tensors="pt"
                        )["pixel_values_videos"].to(self.vj_encoder.device))
                    return torch.cat(input_videos, dim=0)
                if vj_pixel_values_videos is not None and vj_pixel_values_videos.shape[0] == expected_videos:
                    return vj_pixel_values_videos.to(self.vj_encoder.device, non_blocking=True)
                if vj_pixel_values_videos is not None and is_mixed_cotrain_batch:
                    if batch_videos is None:
                        raise RuntimeError(
                            "Raw videos are required when mixed cotrain VJ inputs are only partially preprocessed."
                        )
                    vla_vj = vj_pixel_values_videos.to(self.vj_encoder.device, non_blocking=True)
                    input_videos = [vla_vj]
                    for i in range(vla_vj.shape[0], expected_videos):
                        input_videos.append(self.vj_processor(
                            videos=batch_videos[i], return_tensors="pt"
                        )["pixel_values_videos"].to(self.vj_encoder.device))
                    return torch.cat(input_videos, dim=0)
                if batch_videos is None:
                    raise RuntimeError("Raw videos are required when vj_pixel_values_videos is not provided.")
                input_videos = []
                for i in range(B*V):
                    input_videos.append(self.vj_processor(
                        videos=batch_videos[i], return_tensors="pt"
                    )["pixel_values_videos"].to(self.vj_encoder.device))
                return torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]

            input_videos = timed("forward/vj_processor_h2d_time", prepare_vj_videos)
            with torch.no_grad():
                video_embeddings = timed(
                    "forward/vj_encoder_time",
                    lambda: self.vj_encoder.get_vision_features(pixel_values_videos=input_videos),
                )
                encoded_videos, num_video_tokens, video_embed_dim = video_embeddings.shape
                expected_videos = B * V
                if encoded_videos != expected_videos:
                    raise RuntimeError(
                        f"Expected V-JEPA encoder to return {expected_videos} videos "
                        f"({B} batch * {V} views), got {encoded_videos}."
                    )
                # VJ inputs are flattened sample-major: [b0v0, b0v1, b1v0, b1v1, ...].
                video_embeddings = (
                    video_embeddings.reshape(B, V, num_video_tokens, video_embed_dim)
                    .transpose(1, 2)
                    .reshape(B, num_video_tokens, V * video_embed_dim)
                )
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            tokens_per_frame = video_embeddings.shape[1] // T
            input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1),:]  # [B, (T-1)*dim_per_frame, V*embed_dim]
            gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]
            #print(input_states.shape, action_tokens.shape)
            #exit()
            predicted_states = timed(
                "forward/vj_predictor_time",
                lambda: self.vj_predictor(
                    input_states,
                    action_tokens
                ),
            )

            teacher_forcing_wm_loss = timed(
                "forward/wm_loss_time",
                lambda: F.l1_loss(
                    predicted_states,
                    gt_states,
                    reduction="mean"
                ),
            )
            future_token_loss = None
            if self.use_future_tokens and future_token_states is not None and self.future_projector is not None:
                gt_future_summary = gt_states.view(B, T - 1, tokens_per_frame, -1).mean(dim=2)
                future_token_loss = timed(
                    "forward/future_token_loss_time",
                    lambda: F.l1_loss(
                        self.future_projector(future_token_states).to(gt_future_summary.dtype),
                        gt_future_summary.detach(),
                        reduction="mean",
                    ),
                )
        
        if actions is None:
            self.last_forward_timing = timing
            result = {"wm_loss": teacher_forcing_wm_loss}
            if future_token_loss is not None:
                future_loss_scale = float(self.config.trainer.get("loss_scale", {}).get("future", 1.0))
                result["future_token_loss"] = future_token_loss * future_loss_scale
            return result

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            def move_actions():
                if torch.is_tensor(actions):
                    return actions.to(last_hidden.device, dtype=last_hidden.dtype, non_blocking=True)
                return torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)

            actions = timed("forward/action_label_h2d_time", move_actions)  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            action_condition_repeated = action_condition_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                if state_tensor is None:
                    def move_state():
                        if torch.is_tensor(state):
                            return state.to(last_hidden.device, dtype=last_hidden.dtype, non_blocking=True)
                        return torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)

                    state_tensor = timed("forward/state_h2d_time", move_state)
                #print(state.shape)
                state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

            #print(embodied_action_repeated.shape, actions_target_repeated.shape, state_repeated.shape) if state_repeated is not None else print("No state for action model")
            #exit()
            action_dct_loss = None
            if self.action_dct_loss_weight > 0:
                action_details = timed(
                    "forward/action_head_loss_time",
                    lambda: self.action_model(
                        action_condition_repeated,
                        actions_target_repeated,
                        state_repeated,
                        return_details=True,
                    ),
                )
                action_loss = action_details["loss"]
                action_dct_loss = self._compute_dct_loss(
                    action_details["pred_actions"].float(),
                    actions_target_repeated.float(),
                ) * self.action_dct_loss_weight
                action_loss = action_loss + action_dct_loss
            else:
                action_loss = timed(
                    "forward/action_head_loss_time",
                    lambda: self.action_model(action_condition_repeated, actions_target_repeated, state_repeated),
                )  # (B, chunk_len, action_dim)

        self.last_forward_timing = timing
        result = {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}
        if action_dct_loss is not None:
            result["action_dct_loss"] = action_dct_loss
        if future_token_loss is not None:
            future_loss_scale = float(self.config.trainer.get("loss_scale", {}).get("future", 1.0))
            result["future_token_loss"] = future_token_loss * future_loss_scale
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = getattr(qwenvl_outputs, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = qwenvl_outputs.hidden_states[-1]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        if state is not None and state.dim() == 2:
            state = state.unsqueeze(1)
        proprio_condition_tokens = None
        if self.use_proprio_input_vlm and self.proprio_projector is not None and state is not None:
            proprio_condition_tokens = self.proprio_projector(state)
        if self.use_soft_connector and self.soft_connector is not None:
            soft_context = last_hidden
            if proprio_condition_tokens is not None:
                soft_context = torch.cat([proprio_condition_tokens, soft_context], dim=1)
            embodied_action_tokens = self.soft_connector(embodied_action_tokens, soft_context)
        action_condition_parts = []
        if proprio_condition_tokens is not None:
            action_condition_parts.append(proprio_condition_tokens)
        action_condition_parts.append(embodied_action_tokens)
        action_condition_tokens = torch.cat(action_condition_parts, dim=1)
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(action_condition_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": action_condition_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
