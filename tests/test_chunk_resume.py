from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from seedvr2_runtime import InProcessTaskManager, SafeStop, _video_frame_count


class FakeSeedVR2CLI:
    """Small CPU-only stand-in for deterministic checkpoint/merge tests."""

    __file__ = str(Path(__file__).resolve())
    DEFAULT_VAE = "fake_vae.safetensors"
    SEEDVR2_FOLDER_NAME = "SEEDVR2"
    torch = torch

    @staticmethod
    def _read_frames_from_cap(cap, max_frames):
        frames = []
        for _ in range(int(max_frames)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            return None
        return torch.from_numpy(np.stack(frames)).to(torch.float32) / 255.0

    @staticmethod
    def _process_frames_core(frames_tensor, args, device_id, debug,
                             runner_cache=None, progress_callback=None):
        if progress_callback:
            progress_callback(1, 1, int(frames_tensor.shape[0]), "fake phase")
        return frames_tensor.to(torch.float32)

    @staticmethod
    def save_frames_to_video(frames_tensor, output_path, fps=12.0,
                             writer=None, video_backend="opencv", use_10bit=False):
        data = (frames_tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        h, w = data.shape[1:3]
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError("Unable to create fake test video writer")
        try:
            for frame in data:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return None

    @staticmethod
    def clear_memory(**kwargs):
        return None

    @staticmethod
    def get_input_type(path):
        return "video"


def _make_video(path: Path, frames: int = 25, fps: float = 12.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64)
    )
    assert writer.isOpened()
    try:
        for i in range(frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 7) % 255
            frame[:, :, 1] = (i * 11) % 255
            frame[:, :, 2] = (i * 13) % 255
            writer.write(frame)
    finally:
        writer.release()


def test_safe_stop_and_resume_commits_and_merges_all_chunks(tmp_path: Path):
    source = tmp_path / "source.mp4"
    _make_video(source)

    manager = InProcessTaskManager(history_path=tmp_path / "history.json")
    cli = FakeSeedVR2CLI()
    manager._engine = cli
    manager._engine_file_identity = {"fake": "test"}

    params = {
        "chunk_size": 10,
        "temporal_overlap": 2,
        "skip_first_frames": 0,
        "load_cap": 0,
        "prepend_frames": 0,
        "dit_model": "fake_dit.safetensors",
        "vae_model": "fake_vae.safetensors",
        "model_dir": str(tmp_path / "models"),
    }
    args = SimpleNamespace(
        chunk_size=10,
        temporal_overlap=2,
        skip_first_frames=0,
        load_cap=0,
        cuda_device="0",
        prepend_frames=0,
        output_format="mp4",
        video_backend="opencv",
        use_10bit=False,
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    logs: list[str] = []
    log = logs.append

    manager.safe_stop_requested = True
    with pytest.raises(SafeStop):
        manager._process_video_resumable(
            cli, "test-job", source, output_dir, params, args, object(), log
        )

    manifest_path = output_dir / "checkpoint" / "checkpoint.json"
    assert manifest_path.is_file()
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["completed_chunks"]) == 1
    first_chunk = output_dir / "checkpoint" / manifest["completed_chunks"][0]["file"]
    assert first_chunk.is_file()
    assert _video_frame_count(first_chunk) == 10

    manager.safe_stop_requested = False
    final_path, completed, total = manager._process_video_resumable(
        cli, "test-job", source, output_dir, params, args, object(), log
    )
    assert completed == total == 3
    final_path = Path(final_path)
    assert final_path.is_file()
    assert _video_frame_count(final_path) == 25
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert len(manifest["completed_chunks"]) == 3
    assert any("Chunk 2/3" in line for line in logs)
