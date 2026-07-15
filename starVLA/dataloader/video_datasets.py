from __future__ import annotations
import os
import random
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from transformers import VJEPA2VideoProcessor

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

def random_crop_or_pad(video, target_h, target_w, pad_value=0):
    """
    video: np.ndarray [T, H, W, 3]
    return: np.ndarray [T, target_h, target_w, 3]
    """
    T, H, W, C = video.shape
    assert C == 3

    # 1️⃣ 随机 crop 起点（如果原图更大）
    top = random.randint(0, H - target_h) if H > target_h else 0
    left = random.randint(0, W - target_w) if W > target_w else 0

    cropped = video[
        :,
        top : top + min(H, target_h),
        left : left + min(W, target_w),
        :
    ]

    # 2️⃣ padding（如果原图更小）
    out = np.full(
        (T, target_h, target_w, 3),
        pad_value,
        dtype=video.dtype
    )

    h, w = cropped.shape[1:3]
    out[:, :h, :w, :] = cropped

    return out

def resize_video(video, target_h, target_w):
    """
    video: np.ndarray [T, H, W, 3]
    return: np.ndarray [T, target_h, target_w, 3]
    """
    T, H, W, C = video.shape
    assert C == 3

    out = np.empty((T, target_h, target_w, 3), dtype=video.dtype)

    for t in range(T):
        out[t] = cv2.resize(
            video[t],
            (target_w, target_h),  # 注意：cv2 是 (W, H)
            interpolation=cv2.INTER_AREA  # 下采样最稳
        )

    return out

def collate_fn(
    batch,
    n_views=2,
    resolution_size=224,
    vj_processor_path=None,
    preprocess_vj_inputs=False,
    qwen_processor_path=None,
    preprocess_qwen_inputs=False,
    qwen_prompt_template="",
    qwen_replace_prompt="",
    qwen_action_tokens=None,
    qwen_embodied_action_token="<|embodied_action|>",
    qwen_future_tokens=None,
):
    images = []
    videos = []
    instructions = []
    for b in batch:
        video, instruction = b[0], b[1]
        images.append([Image.fromarray(video[0]).resize((resolution_size, resolution_size))])
        videos.append(np.stack([video.copy() for _ in range(n_views)], axis=0))  # [V, T, H, W, C]
        instructions.append(instruction)

    videos_np = np.stack(videos)  # [B, V, T, H, W, C]
    collated = {
        "image": images,
        "video": torch.from_numpy(videos_np),
        "lang": instructions,
    }

    if preprocess_qwen_inputs:
        if not qwen_processor_path:
            raise ValueError("qwen_processor_path is required when preprocess_qwen_inputs=True")
        processor = _get_qwen_processor(
            qwen_processor_path,
            qwen_action_tokens or [],
            qwen_embodied_action_token,
            future_tokens=qwen_future_tokens,
        )
        messages = []
        for imgs, instruction in zip(images, instructions):
            prompt = qwen_prompt_template.replace("{instruction}", instruction)
            prompt = prompt.replace("{actions}", qwen_replace_prompt)
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
        videos_for_processor = videos_np.transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, C, H, W]
        B, V, T, C, H, W = videos_for_processor.shape
        videos_for_processor = videos_for_processor.reshape(B * V, T, C, H, W)
        processed = [
            processor(videos=videos_for_processor[i], return_tensors="pt")["pixel_values_videos"]
            for i in range(B * V)
        ]
        collated["vj_pixel_values_videos"] = torch.cat(processed, dim=0)

    return collated

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        text_file: str,
        n_frames: int,
        extensions=(".mp4", ".avi", ".webm"),
        crop_h_size=420,
        crop_w_size=240,
        max_retry: int = 10,
    ):
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.max_retry = max_retry
        self.crop_h_size = crop_h_size
        self.crop_w_size = crop_w_size

        # 只扫描文件名
        self.video_files = [
            f for f in os.listdir(video_dir)
            if f.lower().endswith(extensions)
        ]
        df = pd.read_csv(text_file, sep=";")
        self.id2text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        
        for each in self.video_files:
            file_idx = int(each.split(".")[0])
            if file_idx not in self.id2text:
                self.id2text[file_idx] = "Completing something that humans might want to do."

        if len(self.video_files) == 0:
            raise RuntimeError(f"No video files found in {video_dir}")

    def __len__(self):
        return len(self.video_files)
    
    def _load_video(self, idx):
        file_idx = int(self.video_files[idx].split(".")[0])
        video_path = os.path.join(self.video_dir, self.video_files[idx])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count < self.n_frames:
            raise ValueError(f"Video {video_path} has only {frame_count} frames, which is less than the required {self.n_frames} frames.")

        start = random.randint(0, frame_count - self.n_frames)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_offset in range(self.n_frames):
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {start + frame_offset}")
            frames.append(frame)
        cap.release()
        #frames = random_crop_or_pad(
        #    np.array(frames),
        #    target_h=self.crop_h_size,
        #    target_w=self.crop_w_size,
        #    pad_value=0)
        frames = resize_video(
            np.array(frames),
            target_h=self.crop_h_size,
            target_w=self.crop_w_size)

        #print(frames.shape, video_path, file_idx, file_idx in self.id2text.keys())

        return [frames, self.id2text[file_idx]]

    def __getitem__(self, idx):
        for _ in range(self.max_retry):
            try:
                return self._load_video(idx)
            except Exception as e:
                idx = random.randint(0, len(self.video_files) - 1)

        return self._load_video(2)
