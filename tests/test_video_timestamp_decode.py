from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.dataloader.gr00t_lerobot import video


class _FakeContainer:
    def __init__(self):
        self.closed = False
        self.streams = SimpleNamespace(video=[SimpleNamespace(codec_context=_FakeCodecContext())])

    def close(self):
        self.closed = True


class _FakeCodecContext:
    def __init__(self):
        self.thread_count = 0
        self.closed = False

    def close(self):
        self.closed = True


class _FakeVideoReader:
    def __init__(self, frames):
        self.frames = frames
        self.seek_calls = []
        self.yielded_pts = []
        self.container = _FakeContainer()
        self._c = object()

    def seek(self, timestamp, keyframes_only=False):
        self.seek_calls.append((timestamp, keyframes_only))
        return self

    def __iter__(self):
        for frame in self.frames:
            self.yielded_pts.append(frame["pts"])
            yield frame


def _frame(timestamp, value):
    return {
        "pts": timestamp,
        "data": torch.full((3, 2, 2), value, dtype=torch.uint8),
    }


def test_torchvision_av_decodes_window_with_one_seek(monkeypatch):
    reader = _FakeVideoReader(
        [
            _frame(0.0, 0),
            _frame(0.1, 1),
            _frame(0.2, 2),
            _frame(0.3, 3),
            _frame(0.4, 4),
            _frame(0.5, 5),
            _frame(0.6, 6),
        ]
    )
    codec_context = reader.container.streams.video[0].codec_context
    monkeypatch.setattr(video.torchvision, "set_video_backend", lambda backend: None)
    monkeypatch.setattr(video.torchvision.io, "VideoReader", lambda path, stream: reader)

    frames = video.get_frames_by_timestamps(
        "fake.mp4",
        [0.18, 0.02, 0.18, 0.49],
        video_backend="torchvision_av",
    )

    assert reader.seek_calls == [(0.02, True)]
    assert codec_context.thread_count == 2
    assert codec_context.closed
    assert reader.container is None
    assert reader.yielded_pts == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert frames.shape == (4, 2, 2, 3)
    assert [int(frame[0, 0, 0]) for frame in frames] == [2, 0, 2, 5]
    assert reader._c is None


def test_torchvision_av_prefers_earlier_frame_on_tie(monkeypatch):
    reader = _FakeVideoReader([_frame(0.0, 0), _frame(0.25, 1), _frame(0.5, 2)])
    monkeypatch.setattr(video.torchvision, "set_video_backend", lambda backend: None)
    monkeypatch.setattr(video.torchvision.io, "VideoReader", lambda path, stream: reader)

    frames = video.get_frames_by_timestamps(
        "fake.mp4",
        [0.125, 0.125],
        video_backend="torchvision_av",
    )

    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 0]


def test_torchvision_av_uses_configured_decoder_thread_count(monkeypatch):
    reader = _FakeVideoReader([_frame(0.0, 0), _frame(0.1, 1)])
    codec_context = reader.container.streams.video[0].codec_context
    monkeypatch.setattr(video.torchvision, "set_video_backend", lambda backend: None)
    monkeypatch.setattr(video.torchvision.io, "VideoReader", lambda path, stream: reader)

    video.get_frames_by_timestamps(
        "fake.mp4",
        [0.0],
        video_backend="torchvision_av",
        video_backend_kwargs={"num_threads": 3},
    )

    assert codec_context.thread_count == 3
    assert codec_context.closed


@pytest.mark.parametrize("timestamps", [[], [0.0, np.nan], [np.inf]])
def test_torchvision_av_rejects_invalid_timestamps(timestamps):
    with pytest.raises(ValueError):
        video.get_frames_by_timestamps(
            "fake.mp4",
            timestamps,
            video_backend="torchvision_av",
        )


def test_torchvision_av_rejects_zero_decoder_threads():
    with pytest.raises(ValueError, match="num_threads"):
        video.get_frames_by_timestamps(
            "fake.mp4",
            [0.0],
            video_backend="torchvision_av",
            video_backend_kwargs={"num_threads": 0},
        )
