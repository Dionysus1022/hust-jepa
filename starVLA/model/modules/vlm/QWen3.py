from __future__ import annotations
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
import json
from pathlib import Path
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# [151936, 153984]
_ACTION_TOKEN_MIN = 151936 # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
_ACTION_TOKEN_MAX = 153984 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


def _read_local_hf_config_name(model_id: str) -> str:
    config_path = Path(model_id).expanduser() / "config.json"
    if not config_path.is_file():
        return ""

    try:
        with config_path.open("r") as f:
            hf_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    names = []
    for key in ("model_type", "architectures"):
        value = hf_config.get(key)
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, list):
            names.extend(str(item) for item in value)
    return " ".join(names)


def _get_qwen3_model_class(model_id: str):
    model_name = f"{model_id} {_read_local_hf_config_name(model_id)}".lower()
    compact_name = model_name.replace("_", "").replace("-", "").replace(".", "")

    if "qwen35" in compact_name:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "base_vlm is Qwen3.5, but the current transformers package does not support "
                "`Qwen3_5ForConditionalGeneration`. Upgrade transformers in the VLA_JEPA "
                "environment, for example: "
                "`/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/pip install -U transformers`."
            ) from exc
        return Qwen3_5ForConditionalGeneration
    return Qwen3VLForConditionalGeneration


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        model_cls = _get_qwen3_model_class(model_id)
        logger.info(f"Loading Qwen VLM from {model_id} with class {model_cls.__name__}")
        model = model_cls.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            labels = kwargs.get("labels", None)
            skip_lm_head = self.config.trainer.get("skip_qwen_lm_head", True)
            if skip_lm_head and labels is None:
                outputs = self.model.model(**kwargs)
            else:
                outputs = self.model(
                    **kwargs,
                )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        prompt_replace_dict=None,
        prompt_template=None,
        device="model",
        **kwargs,
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if prompt_template is None:
                if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                    CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                    prompt = CoT_prompt.replace("{instruction}", instruction)
                    if prompt_replace_dict is not None:
                        for k, v in prompt_replace_dict.items():
                            prompt = prompt.replace(k, v)
                else:
                    prompt = instruction
            else:
                prompt = prompt_template.replace("{instruction}", instruction)
                if prompt_replace_dict is not None:
                    for k, v in prompt_replace_dict.items():
                        prompt = prompt.replace(k, v)

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            #print(msg)
            #exit()

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
        )

        #for k, v in batch_inputs.items():
        #    print(k, v.shape if isinstance(v, torch.Tensor) else v)
        #exit()

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now. 
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning (f"action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md.")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        if device == "model":
            device = self.model.device
        if device is not None:
            return batch_inputs.to(device)
        return batch_inputs




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
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
