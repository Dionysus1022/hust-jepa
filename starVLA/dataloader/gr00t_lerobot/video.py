from __future__ import annotations
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import av
import cv2
import numpy as np

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip

# Import decord with graceful fallback
try:
    import decord  # noqa: F401

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict): Backend options. ``torchvision_av`` accepts
            ``num_threads`` and defaults it to 2 to keep dataloader workers bounded.
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        target_ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if target_ts.size == 0:
            raise ValueError("timestamps must contain at least one value")
        if not np.isfinite(target_ts).all():
            raise ValueError("timestamps must contain only finite values")

        # All requested frames in LeRobot are from a short temporal window. Seek
        # once to the keyframe preceding the earliest request, then decode that
        # window sequentially. The previous implementation sought and decoded
        # the same GOP once per timestamp, which made an 8-frame sample perform
        # eight largely redundant AV1 decodes.
        decoded_frames = []
        decoded_ts = []
        max_target_ts = float(target_ts.max())
        decode_threads = int(video_backend_kwargs.get("num_threads", 2))
        if decode_threads < 1:
            raise ValueError(f"num_threads must be at least 1, got {decode_threads}")
        reader = None
        codec_context = None
        try:
            reader = torchvision.io.VideoReader(video_path, "video")
            # torchvision's PyAV backend ignores VideoReader(num_threads=...).
            # Its default codec thread_count=0 asks FFmpeg to create an automatic
            # thread pool (up to roughly one thread per host CPU) for every
            # dataloader worker. Configure the underlying codec before decoding
            # to prevent thousands of lingering threads under multi-worker load.
            codec_context = reader.container.streams.video[0].codec_context
            codec_context.thread_count = decode_threads
            reader.seek(float(target_ts.min()), keyframes_only=True)

            for frame in reader:
                current_ts = float(frame["pts"])
                decoded_ts.append(current_ts)
                decoded_frames.append(frame["data"])

                # The first frame after the latest request is sufficient to
                # decide its nearest neighbour. It also preserves the old
                # implementation's preference for the earlier frame on ties.
                if current_ts > max_target_ts:
                    break
        finally:
            # Close the PyAV container once per requested window. Avoid a forced
            # full-process gc.collect() in this hot dataloader path.
            if reader is not None:
                if hasattr(reader, "_c"):
                    reader._c = None
                container = getattr(reader, "container", None)
                try:
                    if codec_context is not None:
                        codec_context.close()
                finally:
                    if container is not None:
                        container.close()
                        reader.container = None

        if not decoded_frames:
            raise RuntimeError(f"Unable to decode frames from video: {video_path}")

        decoded_ts_array = np.asarray(decoded_ts, dtype=np.float64)
        closest_indices = np.abs(decoded_ts_array[:, None] - target_ts[None, :]).argmin(axis=0)
        selected_frames = []
        for index in closest_indices:
            frame_data = decoded_frames[int(index)]
            if isinstance(frame_data, torch.Tensor):
                frame_data = frame_data.cpu().numpy()
            selected_frames.append(frame_data)

        frames = np.stack(selected_frames, axis=0)
        return frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
