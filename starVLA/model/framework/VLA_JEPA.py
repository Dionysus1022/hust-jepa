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
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

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
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
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
        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self._maybe_compile_modules()
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])
        logger.info(
            f"Qwen prompt placeholders: action_chars={len(self.replace_prompt)}, "
            f"embodied_chars={len(self.embodied_replace_prompt)}, "
            f"num_action_token_ids={len(self.action_token_ids)}"
        )

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
                         embodied_action_token: str = "<|embodied_action|>"):
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
        return action_tokens, action_token_ids, embodied_action_token_id

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
            instructions = examples["lang"]
            actions = examples.get("action", None)
            state = examples.get("state", None)
            vj_pixel_values_videos = examples.get("vj_pixel_values_videos", None)
            qwen_inputs = examples.get("qwen_inputs", None)
        else:
            batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
            batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
            state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
            vj_pixel_values_videos = None
            qwen_inputs = None

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
            if torch.is_tensor(batch_videos):
                return batch_videos.permute(0, 1, 2, 5, 3, 4).contiguous()
            stacked = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
            return stacked.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        batch_videos = timed("forward/video_numpy_stack_time", stack_videos)

        # Step 1: QWenVL input format
        def build_qwen_inputs():
            if qwen_inputs is not None:
                return qwen_inputs
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
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]

            def prepare_vj_videos():
                if vj_pixel_values_videos is not None:
                    return vj_pixel_values_videos.to(self.vj_encoder.device, non_blocking=True)
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
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
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
        
        if actions is None:
            self.last_forward_timing = timing
            return {"wm_loss": teacher_forcing_wm_loss}

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
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                def move_state():
                    if torch.is_tensor(state):
                        return state.to(last_hidden.device, dtype=last_hidden.dtype, non_blocking=True)
                    return torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)

                state = timed("forward/state_h2d_time", move_state)
                #print(state.shape)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            #print(embodied_action_repeated.shape, actions_target_repeated.shape, state_repeated.shape) if state_repeated is not None else print("No state for action model")
            #exit()
            action_loss = timed(
                "forward/action_head_loss_time",
                lambda: self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated),
            )  # (B, chunk_len, action_dim)

        self.last_forward_timing = timing
        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

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
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



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
