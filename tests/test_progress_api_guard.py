from __future__ import annotations

from types import SimpleNamespace

import pytest

from seedvr2_runtime import InProcessTaskManager


def _with_callback(*args, progress_callback=None, **kwargs):
    return None


def _without_callback(*args, **kwargs):
    return None


def test_progress_api_guard_accepts_matching_engine():
    fake = SimpleNamespace(**{
        "encode_all_batches": _with_callback,
        "upscale_all_batches": _with_callback,
        "decode_all_batches": _with_callback,
        "postprocess_all_batches": _with_callback,
        "_process_frames_core": _with_callback,
    })
    InProcessTaskManager._validate_progress_api(fake)


def test_progress_api_guard_rejects_engine_without_cooperative_callback():
    fake = SimpleNamespace(**{
        "encode_all_batches": _with_callback,
        "upscale_all_batches": _with_callback,
        "decode_all_batches": _with_callback,
        "postprocess_all_batches": _without_callback,
        "_process_frames_core": _with_callback,
    })
    with pytest.raises(RuntimeError, match="postprocess_all_batches 不支持 progress_callback"):
        InProcessTaskManager._validate_progress_api(fake)
