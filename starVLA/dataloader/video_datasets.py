from __future__ import annotations
import os
import random
import time
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
        videos.append(video)
        instructions.append(instruction)

    single_view_videos_np = np.stack(videos)  # [B, T, H, W, C]
    B, T, H, W, C = single_view_videos_np.shape
    collated = {
        "image": images,
        "video": None,
        "video_shape": (B, n_views, T, C, H, W),
        "lang": instructions,
    }
    if not preprocess_vj_inputs:
        videos_np = np.repeat(single_view_videos_np[:, None], n_views, axis=1)  # [B, V, T, H, W, C]
        collated["video"] = torch.from_numpy(videos_np)

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
        videos_for_processor = single_view_videos_np.transpose(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        processed = _process_vj_videos(processor, videos_for_processor)
        if not torch.is_tensor(processed) or processed.shape[0] != B:
            actual_shape = getattr(processed, "shape", None)
            raise RuntimeError(
                "SSV V-JEPA preprocessing must preserve one physical video per sample; "
                f"expected first dimension {B}, got {actual_shape}."
            )
        # SSV has one physical camera stream. Both logical JEPA views are the
        # same video, so encode each sample once and duplicate the frozen
        # encoder features later instead of running V-JEPA twice.
        collated["vj_pixel_values_videos"] = processed
        collated["vj_feature_expand_indices"] = torch.arange(
            B, dtype=torch.long
        ).repeat_interleave(n_views)

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
        decode_threads: int = 1,
    ):
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.max_retry = max_retry
        self.crop_h_size = crop_h_size
        self.crop_w_size = crop_w_size
        self.decode_threads = max(int(decode_threads), 0)
        self.slow_video_warn_sec = float(os.environ.get("VLAJEPA_SLOW_VIDEO_WARN_SEC", "3.0"))
        self.skip_slow_videos = os.environ.get("VLAJEPA_SKIP_SLOW_VIDEO", "1") != "0"
        self._slow_video_files = set()

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

    def _open_video_capture(self, video_path):
        if self.decode_threads > 0 and hasattr(cv2, "CAP_PROP_N_THREADS"):
            try:
                cap = cv2.VideoCapture(
                    video_path,
                    cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_N_THREADS, self.decode_threads],
                )
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception:
                pass
        return cv2.VideoCapture(video_path)
    
    def _load_video(self, idx):
        load_start = time.perf_counter()
        file_idx = int(self.video_files[idx].split(".")[0])
        video_path = os.path.join(self.video_dir, self.video_files[idx])

        cap = self._open_video_capture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video: {video_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count < self.n_frames:
                raise ValueError(f"Video {video_path} has only {frame_count} frames, which is less than the required {self.n_frames} frames.")

            start = random.randint(0, frame_count - self.n_frames)

            frames = []
            seek_start = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            seek_time = time.perf_counter() - seek_start
            read_start = time.perf_counter()
            for frame_offset in range(self.n_frames):
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Unable to read frame at index {start + frame_offset} from {video_path}")
                # OpenCV decodes frames as BGR, while PIL/Qwen/V-JEPA processors expect RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            read_time = time.perf_counter() - read_start
        finally:
            cap.release()

        resize_start = time.perf_counter()
        frames = resize_video(
            np.array(frames),
            target_h=self.crop_h_size,
            target_w=self.crop_w_size)
        resize_time = time.perf_counter() - resize_start
        total_time = time.perf_counter() - load_start
        if total_time >= self.slow_video_warn_sec:
            print(
                f"[slow_video_load] {total_time:.3f}s "
                f"seek={seek_time:.3f}s read={read_time:.3f}s resize={resize_time:.3f}s "
                f"frames={frame_count} start={start} path={video_path}",
                flush=True,
            )
            if self.skip_slow_videos:
                self._slow_video_files.add(self.video_files[idx])

        #print(frames.shape, video_path, file_idx, file_idx in self.id2text.keys())

        return [frames, self.id2text[file_idx]]

    def __getitem__(self, idx):
        last_error = None
        for _ in range(self.max_retry):
            try:
                if self.skip_slow_videos and self.video_files[idx] in self._slow_video_files:
                    idx = random.randint(0, len(self.video_files) - 1)
                    continue
                return self._load_video(idx)
            except Exception as e:
                last_error = e
                idx = random.randint(0, len(self.video_files) - 1)

        if last_error is not None:
            print(f"[video_load_retry_exhausted] last_error={last_error}", flush=True)
        return self._load_video(2)
