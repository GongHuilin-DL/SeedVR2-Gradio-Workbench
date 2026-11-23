#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process SeedVR2 runtime with a persistent model cache and resumable chunks.

This module imports the supplied inference_cli.py as a Python module. It does
not launch inference_cli.py as a separate CLI process for the single-GPU path.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib
import inspect
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
HISTORY_PATH = APP_DIR / "seedvr2_gradio_history.json"
MAX_HISTORY = 300
MAX_LOG_LINES = 600
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class TaskAborted(Exception):
    """Cooperative immediate-stop exception raised at an inference batch boundary."""


class SafeStop(Exception):
    """Internal marker: stop only after the current output Chunk is committed."""


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _source_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:80] or "media"


def _is_valid_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
    finally:
        cap.release()
    # OpenCV builds on Windows may not decode an x265 10-bit stream even though
    # the MP4 container is valid. In that case use ffprobe for stream validation.
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        import subprocess
        try:
            cp = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=codec_type,width,height",
                                 "-of", "json", str(path)], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=15)
            if cp.returncode == 0:
                data = json.loads(cp.stdout or "{}")
                streams = data.get("streams") or []
                return bool(streams and streams[0].get("codec_type") == "video"
                            and int(streams[0].get("width", 0)) > 0
                            and int(streams[0].get("height", 0)) > 0)
        except Exception:
            pass
    return False



def _video_frame_count(path: Path) -> Optional[int]:
    """Return decoded video frame count when ffprobe/OpenCV can determine it."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        import subprocess
        try:
            cp = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames,nb_frames", "-of", "json", str(path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            if cp.returncode == 0:
                streams = (json.loads(cp.stdout or "{}").get("streams") or [])
                if streams:
                    for key in ("nb_read_frames", "nb_frames"):
                        value = streams[0].get(key)
                        if value not in (None, "N/A", ""):
                            try:
                                return int(value)
                            except (TypeError, ValueError):
                                pass
        except Exception:
            pass
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported > 0:
            return reported
        count = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
        return count or None
    finally:
        cap.release()


def _verify_video_frames(path: Path, expected: int) -> bool:
    if not _is_valid_video(path):
        return False
    actual = _video_frame_count(path)
    return actual is None or actual == int(expected)

def _quote_concat_path(path: Path) -> str:
    # FFmpeg concat-demuxer escaping for single-quoted paths.
    value = str(path.resolve()).replace("\\", "/")
    return "file '" + value.replace("'", "'\\''") + "'"


def _merge_chunks(chunks: List[Path], output: Path, fps: float, backend: str, logger,
                  expected_frames: Optional[int] = None) -> None:
    if not chunks:
        raise RuntimeError("没有可合并的 Chunk 文件。")
    output.parent.mkdir(parents=True, exist_ok=True)
    assembling = output.with_name(output.stem + ".assembling" + output.suffix)
    if assembling.exists():
        assembling.unlink()

    def opencv_merge() -> None:
        first = cv2.VideoCapture(str(chunks[0]))
        try:
            width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            first.release()
        writer = cv2.VideoWriter(str(assembling), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建最终视频：{assembling}")
        try:
            for chunk in chunks:
                cap = cv2.VideoCapture(str(chunk))
                if not cap.isOpened():
                    raise RuntimeError(f"无法读取 Chunk：{chunk}")
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(frame)
                finally:
                    cap.release()
        finally:
            writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        concat_list = output.parent / "concat_list.txt"
        concat_list.write_text("\n".join(_quote_concat_path(p) for p in chunks), encoding="utf-8")
        import subprocess
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", str(concat_list), "-c", "copy", "-movflags", "+faststart", str(assembling)]
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if cp.returncode != 0:
            assembling.unlink(missing_ok=True)
            if backend != "opencv":
                raise RuntimeError("FFmpeg 合并 Chunk 失败：" + (cp.stderr.strip()[-3000:] or f"exit={cp.returncode}"))
            logger("FFmpeg 无损拼接失败，改用 OpenCV 逐帧解码/重新编码：" + cp.stderr.strip()[-800:])
            opencv_merge()
        else:
            logger("FFmpeg 已无损拼接全部 Chunk（stream copy）。")
    else:
        if backend == "ffmpeg":
            raise RuntimeError("当前选择 FFmpeg 编码，但系统 PATH 中找不到 ffmpeg，无法合并 Chunk。")
        opencv_merge()
        logger("未检测到 FFmpeg，已使用 OpenCV 逐帧合并 Chunk（会重新编码）。")

    if not assembling.exists() or assembling.stat().st_size < 1024:
        raise RuntimeError("合并输出不存在或文件大小异常。")
    if not _is_valid_video(assembling):
        raise RuntimeError("合并后的视频无法通过容器/首帧读取验证；保留 assembling 文件供排查。")
    if expected_frames is not None and not _verify_video_frames(assembling, expected_frames):
        actual = _video_frame_count(assembling)
        raise RuntimeError(f"合并后视频帧数校验失败：期望 {expected_frames}，实际 {actual}；保留 assembling 文件供排查。")
    os.replace(assembling, output)
    try:
        (output.parent / "concat_list.txt").unlink(missing_ok=True)
    except Exception:
        pass


class _TaskDebug:
    """Debug adapter: preserve SeedVR2 console logs and mirror them to task history."""
    def __init__(self, base, emit):
        self._base = base
        self._emit = emit
        self.enabled = getattr(base, "enabled", False)
        original = base.log

        def wrapped(message, *args, **kwargs):
            result = original(message, *args, **kwargs)
            try:
                self._emit(str(message))
            except Exception:
                pass
            return result
        base.log = wrapped

    def __getattr__(self, item):
        return getattr(self._base, item)


class InProcessTaskManager:
    """Serial queue; the imported SeedVR2 engine and its model cache live in this process."""
    ACTIVE = {"queued", "running", "paused", "pause_pending", "cancelling", "safe_stopping"}
    TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}

    def __init__(self, history_path: Path = HISTORY_PATH):
        self.history_path = Path(history_path)
        self.cv = threading.Condition(threading.RLock())
        self.records = self._load_history()
        self.by_id = {r["id"]: r for r in self.records}
        self.pending: collections.deque[str] = collections.deque()
        self.current_id: Optional[str] = None
        self.pause_requested = False
        self.paused = False
        self.stop_requested = False
        self.safe_stop_requested = False
        self._last_submit_sig = ""
        self._last_submit_time = 0.0
        self._runner_cache: Dict[str, Any] = {}
        self._runtime_signature: Optional[str] = None
        self._engine = None
        self._debug = None
        self._engine_file_identity = None
        for rec in self.records:
            if rec.get("status") in self.ACTIVE:
                rec["status"] = "interrupted"
                rec["finished_at"] = _now()
                rec["error"] = "GUI 上次退出时任务未完成；可重新提交相同输入和参数以从已提交的 Chunk 检查点恢复。"
        self._persist_locked()
        self.worker = threading.Thread(target=self._worker_loop, name="SeedVR2-InProcess-Worker", daemon=True)
        self.worker.start()

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
            return [x for x in raw if isinstance(x, dict) and x.get("id")] if isinstance(raw, list) else []
        except Exception:
            backup = self.history_path.with_name(self.history_path.name + f".corrupt-{int(time.time())}")
            try:
                shutil.copy2(self.history_path, backup)
            except Exception:
                pass
            return []

    def _persist_locked(self):
        active = [r for r in self.records if r.get("status") in self.ACTIVE]
        terminal = [r for r in self.records if r.get("status") not in self.ACTIVE]
        self.records = sorted(terminal[-MAX_HISTORY:] + active, key=lambda x: x.get("created_at", ""))
        self.by_id = {r["id"]: r for r in self.records}
        try:
            _atomic_json(self.history_path, self.records)
        except Exception as exc:
            print(f"[SeedVR2 GUI] history save failed: {exc}")

    def _log_locked(self, rec: Dict[str, Any], message: str):
        line = str(message).replace("\x1b", "").rstrip()
        if line:
            rec.setdefault("logs", []).append({"time": _now(), "text": line})
            rec["logs"] = rec["logs"][-MAX_LOG_LINES:]
            self._persist_locked()

    def log(self, job_id: str, message: str):
        with self.cv:
            rec = self.by_id.get(job_id)
            if rec:
                self._log_locked(rec, message)

    def submit(self, paths: List[str], params: Dict[str, Any], cli_path: str, output_root: str) -> Tuple[int, str]:
        signature = _hash_json({"paths": paths, "params": params, "cli": cli_path, "out": output_root})
        now = time.monotonic()
        with self.cv:
            if signature == self._last_submit_sig and now - self._last_submit_time < 2:
                return 0, "检测到短时间内重复点击，已忽略重复提交。"
            self._last_submit_sig, self._last_submit_time = signature, now
            added = 0
            for raw in paths:
                path = str(Path(raw).expanduser().resolve())
                job_id = uuid.uuid4().hex[:12]
                rec = {"id": job_id, "name": Path(path).name or path, "input": path,
                       "cli_path": str(Path(cli_path).expanduser().resolve()),
                       "output_root": str(Path(output_root).expanduser().resolve()),
                       "output_path": "", "status": "queued", "progress": 0,
                       "created_at": _now(), "started_at": None, "finished_at": None,
                       "elapsed_seconds": 0.0, "return_code": None, "error": "",
                       "params": dict(params), "logs": [], "command": [], "checkpoint": ""}
                self.records.append(rec)
                self.by_id[job_id] = rec
                self.pending.append(job_id)
                self._log_locked(rec, f"任务已加入队列：{path}")
                added += 1
            self._persist_locked()
            self.cv.notify_all()
        return added, f"已提交 {added} 个任务；队列由同一个 Python 进程串行执行。"

    def _cancel_pending_locked(self, reason: str):
        while self.pending:
            jid = self.pending.popleft()
            rec = self.by_id.get(jid)
            if rec and rec.get("status") == "queued":
                rec["status"] = "cancelled"
                rec["finished_at"] = _now()
                rec["error"] = reason
                self._log_locked(rec, reason)

    def pause(self) -> str:
        with self.cv:
            if self.stop_requested or self.safe_stop_requested:
                return "任务正在停止，不能暂停。"
            if self.pause_requested or self.paused:
                return "已经请求暂停或处于暂停状态。"
            if not self.current_id and not self.pending:
                return "当前没有可暂停的任务。"
            self.pause_requested = True
            rec = self.by_id.get(self.current_id) if self.current_id else None
            if rec:
                rec["status"] = "pause_pending"
                self._log_locked(rec, "收到暂停请求；将在下一个模型 Batch 回调边界暂停。")
            self.cv.notify_all()
        return "已请求暂停；会在推理内部的下一个 Batch 回调边界暂停，GPU 已分配内存不会自动释放。"

    def resume(self) -> str:
        with self.cv:
            if not self.pause_requested and not self.paused:
                return "当前没有暂停任务。"
            self.pause_requested = False
            self.paused = False
            rec = self.by_id.get(self.current_id) if self.current_id else None
            if rec and rec.get("status") in {"paused", "pause_pending"}:
                rec["status"] = "running"
                self._log_locked(rec, "已恢复推理。")
            self.cv.notify_all()
        return "已恢复推理。"

    def stop_all(self) -> str:
        with self.cv:
            if self.stop_requested:
                return "立即停止请求已提交。"
            if not self.current_id and not self.pending:
                return "当前没有运行中或等待中的任务。"
            self.stop_requested = True
            self.safe_stop_requested = False
            self.pause_requested = False
            self.paused = False
            self._cancel_pending_locked("用户请求立即停止，等待任务已取消。")
            rec = self.by_id.get(self.current_id) if self.current_id else None
            if rec:
                rec["status"] = "cancelling"
                self._log_locked(rec, "已请求尽快终止；不会强杀 Python 进程，推理将在下一个 Batch 回调/安全边界退出。当前未提交 Chunk 不会加入检查点。")
            self.cv.notify_all()
        return "已请求立即停止。当前 Python/CUDA 调用无法安全地从线程外强杀；将在推理回调边界停止。"

    def safe_stop(self) -> str:
        with self.cv:
            if self.safe_stop_requested:
                return "安全停止请求已提交。"
            if not self.current_id and not self.pending:
                return "当前没有运行中或等待中的任务。"
            self.safe_stop_requested = True
            self.stop_requested = False
            self.pause_requested = False
            self.paused = False
            self._cancel_pending_locked("用户请求安全停止，等待任务已取消。")
            rec = self.by_id.get(self.current_id) if self.current_id else None
            if rec:
                rec["status"] = "safe_stopping"
                self._log_locked(rec, "安全停止已请求：当前 Chunk 完成推理、写入、关闭并提交检查点后停止。")
            self.cv.notify_all()
        return "已请求安全停止；当前 Chunk 会完整写入并记录检查点，然后停止后续 Chunk。"

    def _control_hook(self, job_id: str):
        with self.cv:
            while self.pause_requested and not self.stop_requested:
                self.paused = True
                rec = self.by_id.get(job_id)
                if rec:
                    rec["status"] = "paused"
                self.cv.notify_all()
                self.cv.wait(timeout=0.5)
            self.paused = False
            if self.stop_requested:
                raise TaskAborted("用户请求立即停止；当前尚未提交的 Chunk 已放弃。")

    def _load_engine(self, cli_path: str):
        cli_file = Path(cli_path).resolve()
        if not cli_file.is_file():
            raise FileNotFoundError(f"找不到推理核心：{cli_file}")
        if str(cli_file.parent) not in os.sys.path:
            os.sys.path.insert(0, str(cli_file.parent))
        if self._engine is None:
            existing = os.sys.modules.get("inference_cli")
            existing_path = getattr(existing, "__file__", None) if existing is not None else None
            if existing_path and Path(existing_path).resolve() != cli_file:
                raise RuntimeError(
                    "当前 Python 进程已经从另一个目录导入 inference_cli.py；"
                    "为避免混用 SeedVR2 源码，请在独立的 Conda 进程中启动本 GUI。"
                )
            self._engine = importlib.import_module("inference_cli")
        elif Path(getattr(self._engine, "__file__", cli_file)).resolve() != cli_file:
            raise RuntimeError("当前 GUI 实例已绑定另一份 inference_cli.py；请重启 GUI 后切换 SeedVR2 项目目录。")
        self._validate_progress_api(self._engine)
        return self._engine

    @staticmethod
    def _validate_progress_api(cli) -> None:
        """Fail before model loading if the installed SeedVR2 phases cannot cooperate."""
        required = ("encode_all_batches", "upscale_all_batches", "decode_all_batches",
                    "postprocess_all_batches", "_process_frames_core")
        incompatible = []
        for name in required:
            function = getattr(cli, name, None)
            if not callable(function):
                incompatible.append(f"{name} 不存在")
                continue
            try:
                parameters = inspect.signature(function).parameters
            except (TypeError, ValueError):
                incompatible.append(f"无法读取 {name} 的函数签名")
                continue
            if "progress_callback" not in parameters:
                incompatible.append(f"{name} 不支持 progress_callback")
        if incompatible:
            details = "；".join(incompatible)
            raise RuntimeError(
                "当前 SeedVR2 核心版本不兼容同进程安全停止/Chunk 检查点模式：" + details +
                "。请使用本项目配套的 inference_cli.py，以及支持 progress_callback 的 SeedVR2 src/core/generation_phases.py。"
            )

    def _model_path_overrides(self, cli, p: Dict[str, Any]):
        """Install one persistent resolver wrapper for explicitly selected model files."""
        overrides = {}
        dit_path = str(p.get("dit_model_path", "") or "").strip()
        vae_path = str(p.get("vae_model", "") or "").strip()
        if dit_path:
            path = Path(dit_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"DiT 模型文件不存在：{path}")
            overrides[path.name] = str(path)
        if vae_path:
            candidate = Path(vae_path).expanduser()
            # A plain registry filename is resolved by SeedVR2's original resolver.
            if candidate.is_absolute() or candidate.parent != Path("."):
                path = candidate.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"VAE 模型文件不存在：{path}")
                overrides[path.name] = str(path)
        if not hasattr(cli, "_gui_model_file_overrides"):
            cli._gui_model_file_overrides = {}
        cli._gui_model_file_overrides.clear()
        cli._gui_model_file_overrides.update(overrides)
        if getattr(cli, "_gui_model_resolver_installed", False):
            return

        import sys
        modules = [m for m in list(sys.modules.values()) if m is not None and
                   getattr(m, "__name__", "").startswith("src.")]
        targets = []
        for module in modules:
            for name, value in list(vars(module).items()):
                if name == "find_model_file" and callable(value) and not getattr(value, "_seedvr2_gui_wrapper", False):
                    targets.append((module, name, value))
        # Also inspect prepare_runner's own global namespace; imported function refs
        # may not be the same object as the resolver exposed by the module.
        for name, value in list(getattr(cli.prepare_runner, "__globals__", {}).items()):
            if name == "find_model_file" and callable(value) and not getattr(value, "_seedvr2_gui_wrapper", False):
                if not any(v is value for _, _, v in targets):
                    targets.append((cli.prepare_runner.__globals__, name, value))
        if not targets and overrides:
            raise RuntimeError("当前 SeedVR2 版本没有找到 find_model_file；请把模型放入 model_dir 并在界面填写模型文件名。")
        for owner, name, original in targets:
            def make_wrapper(original_func):
                def wrapped(filename, base_cache_dir=None, *args, **kwargs):
                    key = Path(str(filename)).name
                    selected = cli._gui_model_file_overrides.get(key)
                    if selected:
                        return selected
                    try:
                        candidate = Path(str(filename)).expanduser()
                        if candidate.is_file():
                            return str(candidate.resolve())
                    except Exception:
                        pass
                    try:
                        return original_func(filename, base_cache_dir, *args, **kwargs)
                    except TypeError:
                        return original_func(filename, *args, **kwargs)
                wrapped._seedvr2_gui_wrapper = True
                return wrapped
            setattr(owner, name, make_wrapper(original))
        cli._gui_model_resolver_installed = True

    def _runtime_sig(self, p: Dict[str, Any]) -> str:
        keys = ["dit_model", "dit_model_path", "vae_model", "model_dir", "cuda_device",
                "dit_offload_device", "vae_offload_device", "tensor_offload_device",
                "blocks_to_swap", "swap_io_components", "attention_mode", "compile_dit",
                "compile_vae", "compile_backend", "compile_mode", "compile_fullgraph",
                "compile_dynamic", "compile_dynamo_cache_size_limit", "compile_dynamo_recompile_limit",
                "vae_encode_tiled", "vae_encode_tile_size", "vae_encode_tile_overlap",
                "vae_decode_tiled", "vae_decode_tile_size", "vae_decode_tile_overlap", "tile_debug",
                "cache_dit", "cache_vae", "keep_models_resident"]
        return _hash_json({k: p.get(k) for k in keys})

    @staticmethod
    def _file_identity(value: str, model_dir: str = "") -> Dict[str, Any]:
        if not value:
            return {"name": ""}
        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() or raw.parent != Path(".") else []
        if model_dir:
            candidates.append(Path(model_dir).expanduser() / raw.name)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if resolved.is_file():
                    st = resolved.stat()
                    return {"path": str(resolved), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
            except Exception:
                continue
        return {"name": raw.name, "model_dir": str(Path(model_dir).expanduser()) if model_dir else ""}

    def _prepare_task_context(self, cli, p: Dict[str, Any], log):
        sig = self._runtime_sig(p)
        model_dir = str(p.get("model_dir", "") or "")
        if not model_dir:
            model_dir = str(Path(cli.__file__).resolve().parent / "models" / str(getattr(cli, "SEEDVR2_FOLDER_NAME", "SEEDVR2")))
        # The identity represents model *weights*, not the execution policy. Device,
        # offload, BlockSwap, attention and compile settings can be changed without
        # throwing away the expensive weights: upstream prepare_runner reconfigures
        # the cached runner in place for those settings.
        model_identity = _hash_json({
            "dit_model": self._file_identity(str(p.get("dit_model_path") or p.get("dit_model") or ""), model_dir),
            "vae_model": self._file_identity(str(p.get("vae_model") or getattr(cli, "DEFAULT_VAE", "")), model_dir),
            "model_dir": str(Path(model_dir).expanduser().resolve()),
        })
        model_changed = getattr(self, "_model_identity", None) not in (None, model_identity)
        if sig != self._runtime_signature:
            # Drop only per-run context (device metadata, intermediates and runner
            # reference). The upstream global cache remains available and will
            # refresh runner configuration using the new settings.
            self._runner_cache = {}
            self._runtime_signature = sig
            log("推理配置发生变化：重建任务上下文；模型权重缓存保持可复用。")
        if model_changed:
            # remove_dit/remove_vae also evict runner templates associated with the
            # corresponding node IDs in the upstream GlobalModelCache.
            self._clear_seedvr2_model_cache(cli, log)
            self._runner_cache = {}
            log("检测到 DiT/VAE 权重文件发生变化，已清除旧权重和关联 Runner 模板。")
        self._model_identity = model_identity
        self._model_path_overrides(cli, p)

    def _clear_seedvr2_model_cache(self, cli, log):
        """Remove the two stable CLI cache entries before loading a different file.

        The upstream SeedVR2 model cache validates the cached model by filename;
        two different files can share a basename, so path changes need explicit
        eviction rather than filename-only validation.
        """
        try:
            module = importlib.import_module("src.core.model_configuration")
            getter = getattr(module, "get_global_cache", None)
            if not callable(getter):
                log("当前 SeedVR2 版本没有公开 get_global_cache；已清空 GUI runner 引用，但全局模型缓存可能仍保留旧权重。")
                return
            cache = getter()
            cache.remove_dit({"node_id": "cli_dit"}, self._debug or cli.debug)
            cache.remove_vae({"node_id": "cli_vae"}, self._debug or cli.debug)
            log("DiT/VAE 的旧全局缓存条目已清除；新模型将按新的文件路径加载。")
        except Exception as exc:
            log(f"清除旧全局模型缓存时未能调用当前版本的缓存 API：{type(exc).__name__}: {exc}；请留意模型切换后的 RAM/VRAM 占用。")

    def _build_args(self, cli, params: Dict[str, Any]):
        p = dict(params)
        # Keep both model families cached in the resident GUI process by default.
        keep = bool(p.get("keep_models_resident", True))
        p["cache_dit"] = bool(p.get("cache_dit", False) or keep)
        p["cache_vae"] = bool(p.get("cache_vae", False) or keep)
        if keep and p.get("dit_offload_device", "none") == "none":
            p["dit_offload_device"] = "cpu"
        if keep and p.get("vae_offload_device", "none") == "none":
            p["vae_offload_device"] = "cpu"
        dit_path = str(p.get("dit_model_path", "") or "").strip()
        if dit_path:
            p["dit_model"] = Path(dit_path).name
        vae_path = str(p.get("vae_model", "") or "").strip()
        if vae_path:
            p["vae_model"] = Path(vae_path).name
        else:
            p["vae_model"] = getattr(cli, "DEFAULT_VAE", "")
        # Every input video is checkpointed at explicit Chunk boundaries. chunk_size=0
        # means one single Chunk covering all frames.
        args = type("SeedVR2Args", (), {})()
        for key, value in p.items():
            setattr(args, key, value)
        defaults = {
            "output_format": "auto", "video_backend": "opencv", "use_10bit": False,
            "model_dir": None, "resolution": 1080, "max_resolution": 0, "batch_size": 5,
            "uniform_batch_size": False, "seed": 42, "skip_first_frames": 0, "load_cap": 0,
            "chunk_size": 0, "prepend_frames": 0, "temporal_overlap": 0,
            "color_correction": "lab", "input_noise_scale": 0.0, "latent_noise_scale": 0.0,
            "cuda_device": "", "dit_offload_device": "none", "vae_offload_device": "none",
            "tensor_offload_device": "cpu", "blocks_to_swap": 0, "swap_io_components": False,
            "vae_encode_tiled": False, "vae_encode_tile_size": 1024, "vae_encode_tile_overlap": 128,
            "vae_decode_tiled": False, "vae_decode_tile_size": 1024, "vae_decode_tile_overlap": 128,
            "tile_debug": "false", "attention_mode": "sdpa", "compile_dit": False,
            "compile_vae": False, "compile_backend": "inductor", "compile_mode": "default",
            "compile_fullgraph": False, "compile_dynamic": False,
            "compile_dynamo_cache_size_limit": 64, "compile_dynamo_recompile_limit": 128,
            "cache_dit": True, "cache_vae": True, "debug": False,
        }
        for k, v in defaults.items():
            if not hasattr(args, k):
                setattr(args, k, v)
        if not getattr(args, "model_dir", None):
            args.model_dir = None
        return args

    def _checkpoint_model_identity(self, cli, params: Dict[str, Any]) -> Dict[str, Any]:
        model_dir = str(params.get("model_dir", "") or "")
        if not model_dir and cli is not None:
            model_dir = str(Path(cli.__file__).resolve().parent / "models" / str(getattr(cli, "SEEDVR2_FOLDER_NAME", "SEEDVR2")))
        return {
            "dit": self._file_identity(str(params.get("dit_model_path") or params.get("dit_model") or ""), model_dir),
            "vae": self._file_identity(str(params.get("vae_model") or getattr(cli, "DEFAULT_VAE", "")), model_dir),
            "engine": self._engine_file_identity,
        }

    def _output_base(self, input_path: Path, output_root: Path, params: Dict[str, Any], cli=None) -> Path:
        source_id = hashlib.sha256(str(input_path.resolve()).encode("utf-8")).hexdigest()[:10]
        # Fingerprint includes source stat, weight-file stat and inference engine version.
        identity = {"source": _source_fingerprint(input_path), "params": params,
                    "models": self._checkpoint_model_identity(cli, params) if cli is not None else None}
        run_id = _hash_json(identity)[:14]
        return output_root / f"{_safe_filename(input_path.stem or input_path.name)}_{source_id}" / "runs" / run_id

    def _manifest_for(self, work_dir: Path, source: Path, params: Dict[str, Any], total_frames: int,
                      chunk_size: int, overlap: int, fps: float, width: int, height: int) -> Dict[str, Any]:
        manifest_path = work_dir / "checkpoint.json"
        fingerprint = _hash_json({"source": _source_fingerprint(source), "params": params,
                                  "models": self._checkpoint_model_identity(self._engine, params),
                                  "total_frames": total_frames, "chunk_size": chunk_size, "overlap": overlap})
        if manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
                if old.get("fingerprint") == fingerprint:
                    return old
                raise ValueError("任务参数或输入文件已变化")
            except Exception:
                bad = manifest_path.with_name(f"checkpoint.invalid-{int(time.time())}.json")
                try:
                    os.replace(manifest_path, bad)
                except Exception:
                    pass
        manifest = {"schema": 1, "fingerprint": fingerprint, "source": _source_fingerprint(source),
                    "params_fingerprint": _hash_json(params), "total_frames": total_frames,
                    "chunk_size": chunk_size, "overlap": overlap, "fps": fps,
                    "width": width, "height": height, "completed_chunks": [],
                    "created_at": _now(), "updated_at": _now(), "final_output": "", "complete": False}
        _atomic_json(manifest_path, manifest)
        return manifest

    def _resume_completed(self, work_dir: Path, manifest: Dict[str, Any], log) -> List[Path]:
        chunks = []
        kept = []
        for item in manifest.get("completed_chunks", []):
            path = work_dir / item.get("file", "")
            if path.is_dir():
                pngs = sorted(path.glob("*.png"))
                is_valid = len(pngs) == int(item.get("frames", -1)) and bool(pngs)
            else:
                is_valid = path.is_file() and path.stat().st_size >= 1024 and _is_valid_video(path)
            if is_valid and not path.is_dir():
                expected_size = int(item.get("size", 0) or 0)
                expected_frames = int(item.get("frames", 0) or 0)
                if expected_size and path.stat().st_size != expected_size:
                    is_valid = False
                if is_valid and expected_frames > 0 and not _verify_video_frames(path, expected_frames):
                    is_valid = False
            if is_valid:
                chunks.append(path)
                kept.append(item)
            else:
                log(f"检查点中的 Chunk 文件缺失/无效，将从此 Chunk 重新处理：{path}")
                break
        if len(kept) != len(manifest.get("completed_chunks", [])):
            manifest["completed_chunks"] = kept
            manifest["complete"] = False
            manifest["final_output"] = ""
            _atomic_json(work_dir / "checkpoint.json", manifest)
        return chunks

    def _read_exact_frames(self, cli, cap, count: int):
        if count <= 0:
            return None
        tensor = cli._read_frames_from_cap(cap, count)
        if tensor is None or tensor.shape[0] != count:
            got = 0 if tensor is None else tensor.shape[0]
            raise RuntimeError(f"读取视频帧数量不足：需要 {count} 帧，实际读取 {got} 帧。")
        return tensor

    @staticmethod
    def _discard_source_frames(cap, count: int, purpose: str) -> None:
        """Decode-and-discard frames instead of relying on approximate keyframe seeking."""
        for index in range(max(0, int(count))):
            ok, _frame = cap.read()
            if not ok:
                raise RuntimeError(f"{purpose}失败：需要跳过 {count} 帧，但在第 {index + 1} 帧处读取失败。")

    def _clear_transient_runner_context(self) -> None:
        """Drop per-inference tensors while retaining reusable model/runner objects."""
        ctx = self._runner_cache.get("ctx")
        if not isinstance(ctx, dict):
            return
        keep = {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device",
                "tensor_offload_device", "compute_dtype"}
        for key in list(ctx.keys()):
            if key not in keep:
                ctx.pop(key, None)

    def _process_video_resumable(self, cli, job_id: str, input_path: Path, output_dir: Path,
                                 params: Dict[str, Any], args, debug, log):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{input_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            skip = int(args.skip_first_frames)
            total_frames = max(0, total - skip)
            if args.load_cap > 0:
                total_frames = min(total_frames, int(args.load_cap))
            if total_frames <= 0:
                raise RuntimeError("跳过起始帧/帧数限制后，没有可处理的视频帧。")
            chunk_size = int(args.chunk_size) if int(args.chunk_size) > 0 else total_frames
            overlap = int(args.temporal_overlap) if int(args.temporal_overlap) > 0 else 0
            if overlap >= chunk_size and chunk_size > 1:
                raise ValueError("Temporal Overlap 必须小于实际 Chunk Size。")
            total_chunks = (total_frames + chunk_size - 1) // chunk_size
            work_dir = output_dir / "checkpoint"
            chunks_dir = work_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            manifest = self._manifest_for(work_dir, input_path, params, total_frames, chunk_size, overlap,
                                          fps, width, height)
            chunk_paths = self._resume_completed(work_dir, manifest, log)
            completed = len(chunk_paths)
            completed_frames = sum(int(x.get("frames", 0)) for x in manifest.get("completed_chunks", []))
            log(f"视频信息：{total} 帧，{width}×{height}，{fps:.3f} FPS；分为 {total_chunks} 个 Chunk。")
            log(f"检查点恢复：已完成 {completed}/{total_chunks} 个 Chunk（{completed_frames}/{total_frames} 帧）。")
            with self.cv:
                rec = self.by_id.get(job_id)
                if rec:
                    rec["chunk_completed"] = completed
                    rec["chunk_total"] = total_chunks
                    rec["progress"] = min(92, int(92 * completed / max(1, total_chunks)))
                    rec["checkpoint"] = str(work_dir / "checkpoint.json")
                    self._persist_locked()

            if completed >= total_chunks and manifest.get("complete") and manifest.get("final_output"):
                final = Path(manifest["final_output"])
                is_png_final = (str(params.get("output_format", "auto")).lower() == "png"
                                and final.is_dir()
                                and len(list(final.glob("*.png"))) == total_frames)
                if is_png_final or (final.is_file() and _verify_video_frames(final, total_frames)):
                    log(f"检查点显示任务已完成，复用现有最终输出：{final}")
                    return str(final), completed, total_chunks

            # Resume exact temporal context by decoding all completed source frames up to
            # (next_start - overlap), then preserving the last overlap raw frames.
            resume_new_frames = min(completed_frames, total_frames)
            context_count = min(overlap, resume_new_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            discard = skip + resume_new_frames - context_count
            self._discard_source_frames(cap, discard, "恢复检查点时重新定位视频帧")
            raw_context = self._read_exact_frames(cli, cap, context_count) if context_count else None
            device_list = [x.strip() for x in str(args.cuda_device or "0").split(",") if x.strip()] or ["0"]
            if len(device_list) != 1:
                raise RuntimeError("可断点恢复的 Chunk 推理目前要求单 GPU。多 GPU 保留原 CLI 分发路径，但不支持跨任务常驻缓存/Chunk 检查点。")
            device_id = device_list[0]
            if completed == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._discard_source_frames(cap, skip, "跳过视频开头帧")

            for chunk_idx in range(completed, total_chunks):
                self._control_hook(job_id)
                remaining = total_frames - (chunk_idx * chunk_size)
                this_count = min(chunk_size, remaining)
                new_frames = self._read_exact_frames(cli, cap, this_count)
                if chunk_idx == 0 or raw_context is None or context_count <= 0:
                    frames = new_frames
                    actual_context = 0
                else:
                    frames = cli.torch.cat([raw_context, new_frames], dim=0)
                    actual_context = int(raw_context.shape[0])
                chunk_args = type("ChunkArgs", (), {})()
                chunk_args.__dict__.update(vars(args))
                if chunk_idx > 0:
                    chunk_args.prepend_frames = 0
                log(f"开始 Chunk {chunk_idx + 1}/{total_chunks}：{this_count} 个新帧 + {actual_context} 个上下文帧。")
                with self.cv:
                    rec = self.by_id.get(job_id)
                    if rec:
                        rec["chunk_current"] = chunk_idx + 1
                        rec["chunk_total"] = total_chunks
                        rec["progress"] = min(92, int(92 * chunk_idx / max(1, total_chunks)))
                        self._persist_locked()
                def phase_callback(*_a, **_kw):
                    self._control_hook(job_id)
                result = cli._process_frames_core(
                    frames_tensor=frames.to(cli.torch.float16), args=chunk_args,
                    device_id=device_id, debug=debug, runner_cache=self._runner_cache,
                    progress_callback=phase_callback)
                self._control_hook(job_id)
                # The upstream pipeline prepends reversed source frames only on
                # the first chunk. Remove them before committing the output chunk.
                if chunk_idx == 0 and int(getattr(chunk_args, "prepend_frames", 0)) > 0:
                    prepend_count = int(chunk_args.prepend_frames)
                    if int(result.shape[0]) > prepend_count:
                        result = result[prepend_count:]
                    else:
                        raise RuntimeError(
                            f"首个 Chunk 输出只有 {int(result.shape[0])} 帧，无法移除 {prepend_count} 个前置帧。"
                        )
                if actual_context:
                    result = result[actual_context:]
                if result is None or int(result.shape[0]) <= 0:
                    raise RuntimeError(f"Chunk {chunk_idx + 1} 没有生成有效帧。")
                if int(result.shape[0]) != this_count:
                    raise RuntimeError(f"Chunk {chunk_idx + 1} 输出帧数与输入帧数不一致：输入 {this_count}，输出 {int(result.shape[0])}。为防止错误续接，不提交该 Chunk 检查点。")

                next_raw_context = new_frames[-overlap:].clone() if overlap > 0 else None
                frame_count = int(this_count)
                if str(args.output_format).lower() == "png":
                    chunk_name = f"chunk_{chunk_idx + 1:06d}_frames"
                    final_chunk = chunks_dir / chunk_name
                    temp_chunk = chunks_dir / (chunk_name + ".partial")
                    if temp_chunk.exists():
                        shutil.rmtree(temp_chunk, ignore_errors=True)
                    cli.save_frames_to_image(result, str(temp_chunk), input_path.stem,
                                             start_index=chunk_idx * chunk_size)
                    pngs = sorted(temp_chunk.glob("*.png"))
                    if len(pngs) != frame_count:
                        raise RuntimeError(f"PNG Chunk {chunk_idx + 1} 保存帧数异常：期望 {frame_count}，实际 {len(pngs)}。")
                    try:
                        self._control_hook(job_id)
                    except TaskAborted:
                        shutil.rmtree(temp_chunk, ignore_errors=True)
                        raise
                    if final_chunk.exists():
                        shutil.rmtree(final_chunk, ignore_errors=True)
                    os.replace(temp_chunk, final_chunk)
                    chunk_size_bytes = sum(x.stat().st_size for x in pngs)
                else:
                    chunk_name = f"chunk_{chunk_idx + 1:06d}.mp4"
                    final_chunk = chunks_dir / chunk_name
                    partial_chunk = chunks_dir / (chunk_name + ".partial.mp4")
                    partial_chunk.unlink(missing_ok=True)
                    writer = cli.save_frames_to_video(result, str(partial_chunk), fps,
                                                       video_backend=args.video_backend,
                                                       use_10bit=args.use_10bit)
                    if writer is not None:
                        writer.release()
                    if not partial_chunk.is_file() or partial_chunk.stat().st_size < 1024:
                        raise RuntimeError(f"Chunk {chunk_idx + 1} 写入失败或文件大小异常。")
                    if not _verify_video_frames(partial_chunk, frame_count):
                        actual = _video_frame_count(partial_chunk)
                        raise RuntimeError(f"Chunk {chunk_idx + 1} 视频校验失败：期望 {frame_count} 帧，实际 {actual}。")
                    try:
                        self._control_hook(job_id)
                    except TaskAborted:
                        partial_chunk.unlink(missing_ok=True)
                        raise
                    os.replace(partial_chunk, final_chunk)
                    chunk_size_bytes = final_chunk.stat().st_size
                del result, frames, new_frames
                manifest.setdefault("completed_chunks", []).append({
                    "index": chunk_idx, "file": str(Path("chunks") / chunk_name),
                    "frames": frame_count, "size": chunk_size_bytes, "committed_at": _now()})
                manifest["updated_at"] = _now()
                _atomic_json(work_dir / "checkpoint.json", manifest)
                chunk_paths.append(final_chunk)
                log(f"Chunk {chunk_idx + 1}/{total_chunks} 已写入并关闭：{final_chunk.name}；检查点已提交。")
                with self.cv:
                    rec = self.by_id.get(job_id)
                    if rec:
                        rec["progress"] = min(94, int(94 * (chunk_idx + 1) / max(1, total_chunks)))
                        rec["chunk_completed"] = chunk_idx + 1
                        rec["output_path"] = str(output_dir)
                        rec["checkpoint"] = str(work_dir / "checkpoint.json")
                        self._persist_locked()
                # The raw overlap tail is carried to the next Chunk. On resume it is
                # reconstructed by decoding the tail from the original source video.
                raw_context = next_raw_context
                context_count = int(raw_context.shape[0]) if raw_context is not None else 0
                self._clear_transient_runner_context()
                if self._safe_stop_pending() and chunk_idx < total_chunks - 1:
                    raise SafeStop("安全停止：当前 Chunk 已完成并提交检查点；再次提交相同任务即可续跑。")
                cli.clear_memory(debug=debug, deep=True, force=True, timer_name="gui_chunk_cleanup")

            # All chunks are committed. MP4 chunks are concatenated; PNG chunks are
            # collected into one globally numbered image sequence directory.
            output_dir.mkdir(parents=True, exist_ok=True)
            if str(args.output_format).lower() == "png":
                final_output = output_dir / f"{_safe_filename(input_path.stem)}_SeedVR2_frames"
                assembling_dir = output_dir / (final_output.name + ".assembling")
                if assembling_dir.exists():
                    shutil.rmtree(assembling_dir, ignore_errors=True)
                assembling_dir.mkdir(parents=True, exist_ok=True)
                for part_dir in chunk_paths:
                    for png in sorted(part_dir.glob("*.png")):
                        shutil.copy2(png, assembling_dir / png.name)
                expected_pngs = sum(int(x.get("frames", 0)) for x in manifest.get("completed_chunks", []))
                if len(list(assembling_dir.glob("*.png"))) != expected_pngs:
                    raise RuntimeError("最终 PNG 序列数量与已提交 Chunk 帧数不一致。")
                backup_dir = output_dir / (final_output.name + ".previous")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                if final_output.exists():
                    os.replace(final_output, backup_dir)
                try:
                    os.replace(assembling_dir, final_output)
                except Exception:
                    if backup_dir.exists() and not final_output.exists():
                        os.replace(backup_dir, final_output)
                    raise
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
            else:
                final_output = output_dir / f"{_safe_filename(input_path.stem)}_SeedVR2.mp4"
                expected_frames = sum(int(x.get("frames", 0)) for x in manifest.get("completed_chunks", []))
                _merge_chunks(chunk_paths, final_output, fps, args.video_backend, log,
                              expected_frames=expected_frames)
            manifest["complete"] = True
            manifest["final_output"] = str(final_output)
            manifest["updated_at"] = _now()
            _atomic_json(work_dir / "checkpoint.json", manifest)
            log(f"最终输出已生成：{final_output}")
            return str(final_output), total_chunks, total_chunks
        finally:
            cap.release()

    def _safe_stop_pending(self) -> bool:
        with self.cv:
            return self.safe_stop_requested

    def _run_one_file(self, cli, job_id: str, input_path: Path, task_root: Path,
                      params: Dict[str, Any], args, debug, log):
        kind = cli.get_input_type(str(input_path))
        if kind == "unknown":
            raise ValueError(f"不支持的输入类型：{input_path}")
        if kind == "video":
            if len([x for x in str(args.cuda_device or "0").split(",") if x.strip()]) == 1:
                source_tag = hashlib.sha256(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
                output_dir = task_root / f"{_safe_filename(input_path.stem)}_{source_tag}"
                output_dir.mkdir(parents=True, exist_ok=True)
                return self._process_video_resumable(cli, job_id, input_path, output_dir, params, args, debug, log)
            # Multi-GPU compatibility fallback: retain the upstream processing path.
            log("检测到多 GPU：使用 inference_cli 原生多 GPU 分发回退；此模式不提供单 GPU 的 Chunk 断点恢复或跨任务模型常驻。")
        source_tag = hashlib.sha256(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
        file_out_dir = task_root / f"{_safe_filename(input_path.stem)}_{source_tag}"
        file_out_dir.mkdir(parents=True, exist_ok=True)
        original_format = args.output_format
        if original_format == "auto":
            args.output_format = "mp4" if kind == "video" else "png"
        if kind == "video":
            output = file_out_dir / f"{_safe_filename(input_path.stem)}_SeedVR2.mp4"
        else:
            output = file_out_dir / f"{_safe_filename(input_path.stem)}_SeedVR2.png"
        try:
            frame_count = cli.process_single_file(str(input_path), args,
                [x.strip() for x in str(args.cuda_device or "0").split(",") if x.strip()] or ["0"],
                str(output), format_auto_detected=(original_format == "auto"),
                runner_cache=self._runner_cache if len(str(args.cuda_device or "0").split(",")) == 1 else None,
                progress_callback=(lambda *_a, **_kw: self._control_hook(job_id))
                    if len([x for x in str(args.cuda_device or "0").split(",") if x.strip()]) <= 1 else None)
        finally:
            args.output_format = original_format
            self._clear_transient_runner_context()
        return str(output), 1, 1

    def _run_job(self, job_id: str):
        with self.cv:
            rec = self.by_id[job_id]
            params = dict(rec["params"])
            input_path = Path(rec["input"]).expanduser().resolve()
            output_root = Path(rec["output_root"]).expanduser().resolve()
            rec["status"] = "running"
            rec["started_at"] = rec.get("started_at") or _now()
            self.stop_requested = False
            self.safe_stop_requested = False
            self._log_locked(rec, "任务开始；将直接在当前 Gradio Python 进程内调用 SeedVR2 推理核心。")
        started = time.monotonic()
        def log(message):
            self.log(job_id, message)
        try:
            cli = self._load_engine(rec["cli_path"])
            cli_path = Path(rec["cli_path"]).resolve()
            self._engine_file_identity = {
                "inference_cli": _source_fingerprint(cli_path),
                "seedvr2_runtime": _source_fingerprint(Path(__file__).resolve()),
            }
            args = self._build_args(cli, params)
            if bool(params.get("keep_models_resident", True)):
                forced = []
                if params.get("dit_offload_device", "none") == "none":
                    forced.append("DiT")
                if params.get("vae_offload_device", "none") == "none":
                    forced.append("VAE")
                if forced:
                    log("跨任务常驻缓存已启用：" + "/".join(forced) + " 的 Offload Device=none 将在运行时按 CPU RAM offload 处理，以降低 8GB 显存压力。")
            # Use one Debug instance per job; the engine itself remains resident.
            base_debug = cli.Debug(enabled=bool(args.debug))
            debug = _TaskDebug(base_debug, log)
            self._debug = debug
            cli.debug = debug
            if bool(params.get("dit_model_path") or params.get("vae_model")):
                self._model_path_overrides(cli, params)
            # Download registry models if needed. For explicit local model paths the
            # patched resolver points directly to the selected files.
            dit_is_explicit = bool(str(params.get("dit_model_path", "") or "").strip())
            vae_value = str(params.get("vae_model", "") or "").strip()
            vae_path_obj = Path(vae_value).expanduser() if vae_value else None
            vae_is_explicit = bool(vae_path_obj and (vae_path_obj.is_absolute() or vae_path_obj.parent != Path(".")))
            # Request only the model families that still need registry resolution/download.
            # Passing an empty selector for the custom-path family avoids accidentally
            # downloading the upstream default weight as an unused side effect.
            dit_to_download = "" if dit_is_explicit else args.dit_model
            vae_to_download = "" if vae_is_explicit else args.vae_model
            if dit_to_download or vae_to_download:
                ok = cli.download_weight(dit_model=dit_to_download, vae_model=vae_to_download,
                                         model_dir=args.model_dir, debug=debug)
                if not ok:
                    raise RuntimeError("SeedVR2 模型权重准备失败，请检查模型名、模型目录和网络/缓存。")
            else:
                log("DiT 和 VAE 均使用用户指定的本地模型文件，不触发模型下载。")

            # Compute cache identity only after downloads complete; first-run model
            # files now have their final path/size/mtime and can be reused next job.
            self._prepare_task_context(cli, params, log)
            output_root.mkdir(parents=True, exist_ok=True)
            task_root = self._output_base(input_path, output_root, params, cli=cli)
            task_root.mkdir(parents=True, exist_ok=True)
            with self.cv:
                cur = self.by_id[job_id]
                cur["output_path"] = str(task_root)
                self._persist_locked()

            media = cli.get_media_files(str(input_path)) if input_path.is_dir() else [str(input_path)]
            if not media:
                raise RuntimeError(f"输入目录中没有支持的媒体文件：{input_path}")
            final_outputs = []
            for index, raw in enumerate(media, 1):
                self._control_hook(job_id)
                file_path = Path(raw).resolve()
                log(f"文件 {index}/{len(media)}：{file_path.name}")
                out, completed, total = self._run_one_file(cli, job_id, file_path, task_root, params, args, debug, log)
                final_outputs.append(out)
                with self.cv:
                    rec = self.by_id[job_id]
                    rec["progress"] = int(100 * index / max(1, len(media)))
                    rec["output_path"] = out if len(media) == 1 else str(task_root)
                    self._persist_locked()
                if self._safe_stop_pending() and index < len(media):
                    raise SafeStop("安全停止：当前文件/Chunk 已完成；剩余输入文件未处理。")
            elapsed = time.monotonic() - started
            with self.cv:
                rec = self.by_id[job_id]
                rec["status"] = "succeeded"
                rec["progress"] = 100
                rec["elapsed_seconds"] = round(elapsed, 3)
                rec["finished_at"] = _now()
                rec["output_path"] = final_outputs[0] if len(final_outputs) == 1 else str(task_root)
                rec["return_code"] = 0
                self._log_locked(rec, f"任务完成，耗时 {elapsed:.1f}s。输出：{rec['output_path']}")
        except SafeStop as exc:
            self._clear_transient_runner_context()
            with self.cv:
                rec = self.by_id[job_id]
                rec["status"] = "cancelled"
                rec["error"] = str(exc)
                rec["finished_at"] = _now()
                rec["elapsed_seconds"] = round(time.monotonic() - started, 3)
                self._log_locked(rec, str(exc))
        except TaskAborted as exc:
            self._clear_transient_runner_context()
            with self.cv:
                rec = self.by_id[job_id]
                rec["status"] = "cancelled"
                rec["error"] = str(exc)
                rec["finished_at"] = _now()
                rec["elapsed_seconds"] = round(time.monotonic() - started, 3)
                self._log_locked(rec, str(exc))
        except Exception as exc:
            self._clear_transient_runner_context()
            with self.cv:
                rec = self.by_id[job_id]
                rec["status"] = "failed"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["finished_at"] = _now()
                rec["elapsed_seconds"] = round(time.monotonic() - started, 3)
                self._log_locked(rec, rec["error"])
                self._log_locked(rec, traceback.format_exc())
        finally:
            with self.cv:
                self.current_id = None
                self._persist_locked()
                self.cv.notify_all()

    def _worker_loop(self):
        while True:
            with self.cv:
                while not self.pending or self.pause_requested:
                    self.cv.wait(timeout=0.5)
                job_id = self.pending.popleft()
                if job_id not in self.by_id:
                    continue
                self.current_id = job_id
                self.stop_requested = False
                self.safe_stop_requested = False
                self.pause_requested = False
                self.paused = False
            self._run_job(job_id)
            with self.cv:
                self.stop_requested = False
                self.safe_stop_requested = False
                self.pause_requested = False
                self.paused = False
                self.current_id = None
                self.cv.notify_all()

    def clear_model_cache(self) -> str:
        """Explicitly unload the resident DiT/VAE cache while the queue is idle."""
        with self.cv:
            if self.current_id or self.pending:
                return "当前仍有运行中或等待中的任务；请先让队列空闲，再清理模型缓存。"
            cli = self._engine
            debug = self._debug
        if cli is None:
            self._runner_cache.clear()
            self._runtime_signature = None
            self._model_identity = None
            return "当前会话尚未加载 SeedVR2 模型，无需清理。"
        self._clear_seedvr2_model_cache(cli, lambda msg: print(f"[SeedVR2 GUI] {msg}"))
        self._runner_cache.clear()
        self._runtime_signature = None
        self._model_identity = None
        try:
            cli.clear_memory(debug=debug or cli.debug, deep=True, force=True, timer_name="gui_manual_model_unload")
        except Exception as exc:
            return f"已请求移除模型缓存，但释放临时显存时出现异常：{type(exc).__name__}: {exc}"
        return "已请求卸载 DiT/VAE 全局缓存并清理临时显存；CUDA allocator 可能保留可复用缓存块。"

    def snapshot(self) -> Dict[str, Any]:
        with self.cv:
            records = list(self.records)
            current = self.by_id.get(self.current_id) if self.current_id else None
            return {"records": records, "current": dict(current) if current else None,
                    "pending": len(self.pending), "paused": self.pause_requested or self.paused,
                    "active": self.current_id is not None, "can_stop": self.current_id is not None or bool(self.pending)}

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self.cv:
            rec = self.by_id.get(record_id)
            return dict(rec) if rec else None

    def delete_record(self, record_id: str) -> Tuple[bool, str]:
        with self.cv:
            rec = self.by_id.get(record_id)
            if not rec:
                return False, "找不到所选历史记录。"
            if rec.get("status") in self.ACTIVE:
                return False, "不能删除活动任务记录。"
            self.records = [r for r in self.records if r.get("id") != record_id]
            self.by_id.pop(record_id, None)
            self._persist_locked()
            return True, "已删除所选历史记录。"

    def clear_history(self) -> Tuple[bool, str]:
        with self.cv:
            if self.current_id or self.pending:
                return False, "队列仍有运行中或等待中的任务，请先停止队列。"
            self.records.clear()
            self.by_id.clear()
            self._persist_locked()
            return True, "历史记录已全部清空。"
