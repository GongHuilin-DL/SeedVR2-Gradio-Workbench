#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SeedVR2 Gradio Workbench: in-process inference, resident models and checkpoints.

Place this file, seedvr2_runtime.py and inference_cli.py in the SeedVR2 project
root (beside src/), then run: python seedvr2_gradio.py
"""
from __future__ import annotations

import collections
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import math
import html
import json
import os
import platform
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "seedvr2_gradio_config.json"
HISTORY_PATH = APP_DIR / "seedvr2_gradio_history.json"
DEFAULT_CLI = Path(os.environ.get("SEEDVR2_CLI", APP_DIR / "inference_cli.py"))
MAX_HISTORY = 300
MAX_JOB_LOG_LINES = 500
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# These values mirror parse_arguments() in the supplied inference_cli.py.
DEFAULTS: Dict[str, Any] = {
    "output_format": "auto",
    "video_backend": "opencv",
    "use_10bit": False,
    "model_dir": "",
    "dit_model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "dit_model_path": "",
    "vae_model": "",
    "keep_models_resident": True,
    "resolution": 1080,
    "max_resolution": 0,
    "batch_size": 5,
    "uniform_batch_size": False,
    "seed": 42,
    "skip_first_frames": 0,
    "load_cap": 0,
    "chunk_size": 0,
    "prepend_frames": 0,
    "temporal_overlap": 0,
    "color_correction": "lab",
    "input_noise_scale": 0.0,
    "latent_noise_scale": 0.0,
    "cuda_device": "",
    "dit_offload_device": "none",
    "vae_offload_device": "none",
    "tensor_offload_device": "cpu",
    "blocks_to_swap": 0,
    "swap_io_components": False,
    "vae_encode_tiled": False,
    "vae_encode_tile_size": 1024,
    "vae_encode_tile_overlap": 128,
    "vae_decode_tiled": False,
    "vae_decode_tile_size": 1024,
    "vae_decode_tile_overlap": 128,
    "tile_debug": "false",
    "attention_mode": "sdpa",
    "compile_dit": False,
    "compile_vae": False,
    "compile_backend": "inductor",
    "compile_mode": "default",
    "compile_fullgraph": False,
    "compile_dynamic": False,
    "compile_dynamo_cache_size_limit": 64,
    "compile_dynamo_recompile_limit": 128,
    "cache_dit": False,
    "cache_vae": False,
    "debug": False,
}

# Each entry is: key, label, control type, extra settings.
FIELD_SPECS = [
    ("output_format", "输出格式", "dropdown", {"choices": ["auto", "mp4", "png"], "info": "auto：按输入类型自动选择"}),
    ("video_backend", "视频编码后端", "dropdown", {"choices": ["opencv", "ffmpeg"], "info": "视频输出编码后端。OpenCV 通常兼容性更好；FFmpeg 可配合 10-bit x265。"}),
    ("use_10bit", "10-bit 视频输出", "checkbox", {"info": "仅在 video_backend=ffmpeg 时有效"}),
    ("model_dir", "模型目录（留空使用 CLI 默认）", "textbox", {"info": "CLI --model_dir。留空时使用 CLI 默认的 models/SEEDVR2 路径。"}),
    ("dit_model", "DiT 模型", "model", {"info": "从当前 SeedVR2 项目的 model_registry 自动读取可用模型；量化等级由模型文件决定。若填写 DiT 模型文件路径，则优先使用路径指定文件。"}),
    ("dit_model_path", "DiT 模型文件路径", "textbox", {"info": "可选：直接输入 DiT 权重完整路径，或使用旁边的资源管理器按钮选择。填写后优先级高于上方 DiT 模型名。"}),
    ("vae_model", "VAE 模型文件 / 名称", "textbox", {"info": "填写 VAE 模型文件名（由 model_dir 搜索）或本地模型文件完整路径；支持资源管理器选择。"}),
    ("keep_models_resident", "跨任务常驻 DiT / VAE", "checkbox", {"info": "在同一 Gradio Python 进程中复用 DiT/VAE 模型缓存。为避免占满 8GB 显存，Offload Device=none 时会自动使用 CPU RAM；也可手动指定其他 Offload Device。"}),
    ("resolution", "目标短边分辨率", "dropdown_int", {"choices": [256, 384, 480, 512, 640, 720, 768, 896, 1024, 1080, 1280, 1440, 1536, 1920], "info": "输出视频的短边目标像素尺寸；可从常用值选择，也可输入自定义正整数。"}),
    ("max_resolution", "最大边长限制（0=不限）", "dropdown_int", {"choices": [0, 512, 720, 768, 1024, 1280, 1440, 1536, 1920, 2048, 2560, 2880, 3840], "info": "限制输出任意一边的最大像素；0 表示不限制。"}),
    ("batch_size", "Batch Size（需满足 4n+1）", "dropdown_int", {"choices": [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 49, 65, 81], "info": "帧批大小必须满足 4n+1。手动输入不符合规则时，界面会自动修正为最近的合法值并提示。"}),
    ("uniform_batch_size", "统一末批 Batch Size", "checkbox", {"info": "将最后一个不足批大小的 Batch 填充到统一长度；可改善尾批时序，但会增加计算量。"}),
    ("seed", "随机种子", "number_int", {"minimum": -2147483648, "maximum": 2147483647, "info": "随机种子；固定 Seed 有助于对比模型、量化等级和参数差异。"}),
    ("skip_first_frames", "跳过开头帧数", "dropdown_int", {"choices": [0, 1, 2, 5, 10, 15, 30, 60, 120], "info": "视频开始处跳过的帧数；图片输入不适用。可从常见值选取，也可输入自定义非负整数。"}),
    ("load_cap", "最多处理帧数（0=全部）", "dropdown_int", {"choices": [0, 25, 50, 100, 200, 300, 500, 1000], "info": "限制视频读取的最大帧数；0 表示处理全部可用帧。"}),
    ("chunk_size", "流式分块帧数（0=单个大 Chunk）", "dropdown_int", {"choices": [0, 25, 50, 100, 200, 330, 660, 1000], "info": "断点续处理以 Chunk 为最小提交单位。建议长视频设置 100–660；0 表示整个视频作为一个 Chunk，中断后只能重做这个视频。"}),
    ("prepend_frames", "前置反向帧数", "dropdown_int", {"choices": [0, 1, 2, 4, 8, 16], "info": "在输入开头添加反向帧以减轻起始处伪影；CLI 会在输出中移除这些前置帧。"}),
    ("temporal_overlap", "时序重叠帧数", "dropdown_int", {"choices": [0, 1, 2, 3, 4, 8, 12, 16], "info": "相邻流式 Chunk 或多 GPU 分段之间的重叠帧数；增加重叠可能改善连续性，但会增加计算量。"}),
    ("color_correction", "色彩校正", "dropdown", {"choices": ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], "info": "选择输出与输入之间的颜色匹配方法；lab 是建议的通用起点，none 关闭色彩校正。"}),
    ("input_noise_scale", "输入噪声强度", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "info": "在输入图像阶段注入噪声，范围 0–1；可能改变细节，建议从 0 开始测试。"}),
    ("latent_noise_scale", "Latent 噪声强度", "slider", {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "info": "在潜空间阶段注入噪声，范围 0–1；可能影响细节重建与随机性。"}),
    ("cuda_device", "CUDA 设备 ID（如 0 或 0,1；留空=默认）", "dropdown_device", {"info": "从本机检测到的 CUDA GPU 生成选项；也允许输入自定义设备编号组合。留空使用 CLI 默认设备。"}),
    ("dit_offload_device", "DiT Offload Device", "dropdown_device", {"info": "DiT 空闲权重卸载位置。none 表示不主动卸载；cpu 表示卸载到系统内存；GPU ID 表示另一张 GPU。BlockSwap 需要可用的 Offload Device。"}),
    ("vae_offload_device", "VAE Offload Device", "dropdown_device", {"info": "VAE 空闲时卸载到的位置。低显存设备可考虑 cpu。"}),
    ("tensor_offload_device", "Tensor Offload Device", "dropdown_device", {"info": "中间 Tensor 的存储位置；cpu 通常更节省显存，none 表示尽量保留在推理设备。"}),
    ("blocks_to_swap", "Transformer BlockSwap 数量", "dropdown_int", {"choices": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36], "info": "将部分 Transformer Block 在推理过程中交换到 Offload Device；3B 上限 32，7B 上限 36，且必须设置 DiT Offload Device。"}),
    ("swap_io_components", "交换 DiT I/O 组件", "checkbox", {"info": "将 DiT 输入/输出组件卸载以进一步节省 VRAM；要求 DiT Offload Device 不是 none。"}),
    ("vae_encode_tiled", "启用 VAE Encode Tiling", "checkbox", {"info": "将 VAE 编码切分为多个 Tile 处理，可降低峰值显存，但会增加调度开销。"}),
    ("vae_encode_tile_size", "Encode Tile Size", "dropdown_int", {"choices": [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048], "info": "VAE 编码 Tile 边长（像素）；较小值通常更省显存，但会增加切片开销。仅 Encode Tiling 开启时生效。"}),
    ("vae_encode_tile_overlap", "Encode Tile Overlap", "dropdown_int", {"choices": [0, 32, 64, 96, 128, 160, 192, 256, 320], "info": "编码 Tile 之间的重叠像素，用于降低接缝；必须小于 Encode Tile Size。"}),
    ("vae_decode_tiled", "启用 VAE Decode Tiling", "checkbox", {"info": "将 VAE 解码切分为多个 Tile 处理，可降低峰值显存，但会增加调度开销。"}),
    ("vae_decode_tile_size", "Decode Tile Size", "dropdown_int", {"choices": [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048], "info": "VAE 解码 Tile 边长（像素）；较小值通常更省显存，但会增加切片开销。仅 Decode Tiling 开启时生效。"}),
    ("vae_decode_tile_overlap", "Decode Tile Overlap", "dropdown_int", {"choices": [0, 32, 64, 96, 128, 160, 192, 256, 320], "info": "解码 Tile 之间的重叠像素，用于降低接缝；必须小于 Decode Tile Size。"}),
    ("tile_debug", "Tile 调试可视化", "dropdown", {"choices": ["false", "encode", "decode"], "info": "仅调试时输出 VAE Tile 可视化信息；正常推理建议保持 false。"}),
    ("attention_mode", "Attention 后端", "dropdown", {"choices": ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], "info": "注意力计算后端。SDPA 为较稳妥的 PyTorch 内置选项；其他选项需要安装相应依赖，并与 GPU/驱动匹配。"}),
    ("compile_dit", "torch.compile DiT", "checkbox", {"info": "对 DiT 使用 torch.compile；首次编译可能较慢，建议与 eager 模式分开 benchmark。"}),
    ("compile_vae", "torch.compile VAE", "checkbox", {"info": "对 VAE 使用 torch.compile；首次编译可能增加时间和 RAM/VRAM 使用。"}),
    ("compile_backend", "Compile Backend", "dropdown", {"choices": ["inductor", "cudagraphs"], "info": "torch.compile 后端；inductor 可进行算子优化，cudagraphs 偏向降低启动开销。"}),
    ("compile_mode", "Compile Mode", "dropdown", {"choices": ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], "info": "torch.compile 优化模式；max-autotune 首次编译较慢，建议单独比较稳定推理耗时。"}),
    ("compile_fullgraph", "Compile Fullgraph", "checkbox", {"info": "要求编译器尽量将整段模型捕获为单个完整图；遇到图断点时可能失败。"}),
    ("compile_dynamic", "Compile Dynamic Shapes", "checkbox", {"info": "允许动态形状输入，可能减少不同分辨率/Batch 导致的重复编译；结果依赖模型和后端。"}),
    ("compile_dynamo_cache_size_limit", "Dynamo Cache Size Limit", "dropdown_int", {"choices": [16, 32, 64, 128, 256, 512, 1024], "info": "TorchDynamo 的代码缓存容量上限；提高可减少不同输入形状带来的缓存淘汰，但可能增加内存使用。"}),
    ("compile_dynamo_recompile_limit", "Dynamo Recompile Limit", "dropdown_int", {"choices": [16, 32, 64, 128, 256, 512], "info": "同一代码路径允许重新编译的次数上限；输入形状频繁变化时可适当提高。"}),
    ("cache_dit", "缓存 DiT 模型（批处理/流式）", "checkbox", {"info": "批处理/流式任务间复用 DiT 权重；要求 DiT Offload Device，CLI 会将 none 解释为 CPU offload。"}),
    ("cache_vae", "缓存 VAE 模型（批处理/流式）", "checkbox", {"info": "批处理/流式任务间复用 VAE 权重；要求 VAE Offload Device，CLI 会将 none 解释为 CPU offload。"}),
    ("debug", "启用 Debug 日志", "checkbox", {"info": "输出更详细的 CLI 诊断信息；便于排错，但日志更多、I/O 开销略高。"}),
]
PARAM_KEYS = [x[0] for x in FIELD_SPECS]

VALUE_CHOICES = {
    "output_format": {"auto", "mp4", "png"},
    "video_backend": {"opencv", "ffmpeg"},
    "color_correction": {"lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"},
    "tile_debug": {"false", "encode", "decode"},
    "attention_mode": {"sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"},
    "compile_backend": {"inductor", "cudagraphs"},
    "compile_mode": {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"},
}

# Numeric contracts shared by UI callbacks, submission validation and Runtime hand-off.
# Values are validated before any parameter dictionary is persisted or submitted.
INT_PARAMETER_RANGES = {
    "resolution": (1, 16384),
    "max_resolution": (0, 32768),
    "batch_size": (1, 1001),
    "seed": (-2147483648, 2147483647),
    "skip_first_frames": (0, 100000000),
    "load_cap": (0, 100000000),
    "chunk_size": (0, 1000000),
    "prepend_frames": (0, 10000),
    "temporal_overlap": (0, 10000),
    "blocks_to_swap": (0, 36),
    "vae_encode_tile_size": (64, 8192),
    "vae_decode_tile_size": (64, 8192),
    "vae_encode_tile_overlap": (0, 8191),
    "vae_decode_tile_overlap": (0, 8191),
    "compile_dynamo_cache_size_limit": (1, 100000),
    "compile_dynamo_recompile_limit": (1, 100000),
}
FLOAT_PARAMETER_RANGES = {
    "input_noise_scale": (0.0, 1.0),
    "latent_noise_scale": (0.0, 1.0),
}
INT_PARAMETER_KEYS = set(INT_PARAMETER_RANGES)
FLOAT_PARAMETER_KEYS = set(FLOAT_PARAMETER_RANGES)
NUMERIC_PARAMETER_KEYS = INT_PARAMETER_KEYS | FLOAT_PARAMETER_KEYS

DEVICE_INFO_CACHE: Optional[Dict[str, Any]] = None
# Values in this set are persisted by their coupled callbacks to avoid a race
# between a raw-value autosave event and a normalization/linkage callback.
LINKED_AUTOSAVE_KEYS = {
    "batch_size", "dit_model", "blocks_to_swap", "dit_offload_device", "swap_io_components",
    "vae_offload_device", "cache_dit", "cache_vae", "vae_encode_tiled", "vae_encode_tile_size",
    "vae_encode_tile_overlap", "vae_decode_tiled", "vae_decode_tile_size", "vae_decode_tile_overlap",
    "video_backend", "use_10bit", "chunk_size", "temporal_overlap",
}

_MODEL_CHOICES_CACHE: Dict[str, List[str]] = {}
_MODEL_CHOICES_LOCK = threading.RLock()

FALLBACK_MODELS = [
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_3b-Q5_K_M.gguf",
    "seedvr2_ema_3b-Q6_K.gguf",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b-Q4_K_M.gguf",
    "seedvr2_ema_7b-Q8_0.gguf",
]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def pretty_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_model_choices(cli_path: Path, force_refresh: bool = False) -> List[str]:
    """Read the registry in this process; never spawn a second inference interpreter."""
    try:
        resolved_cli = cli_path.expanduser().resolve()
        cache_key = str(resolved_cli)
    except Exception:
        resolved_cli = cli_path
        cache_key = str(cli_path)
    with _MODEL_CHOICES_LOCK:
        if not force_refresh and cache_key in _MODEL_CHOICES_CACHE:
            return _MODEL_CHOICES_CACHE[cache_key][:]
    choices: List[str] = []
    if resolved_cli.exists():
        try:
            import sys
            if str(resolved_cli.parent) not in sys.path:
                sys.path.insert(0, str(resolved_cli.parent))
            from src.utils.model_registry import get_available_dit_models
            values = get_available_dit_models()
            if isinstance(values, (list, tuple)):
                choices = [str(x) for x in values if str(x).strip()]
        except Exception:
            pass
    if not choices:
        choices = FALLBACK_MODELS[:]
    with _MODEL_CHOICES_LOCK:
        _MODEL_CHOICES_CACHE[cache_key] = choices[:]
    return choices


def safe_config_load() -> Tuple[Dict[str, Any], str]:
    cfg = {
        "cli_path": str(DEFAULT_CLI),
        "input_paths": "",
        "output_root": str(APP_DIR / "outputs"),
        **DEFAULTS,
    }
    if not CONFIG_PATH.exists():
        return cfg, "使用默认配置（尚无持久化配置文件）。"
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("配置根节点必须是 JSON 对象")
        for key in cfg:
            if key in raw:
                value = raw[key]
                if key in DEFAULTS:
                    default = DEFAULTS[key]
                    if isinstance(default, bool) and not isinstance(value, bool):
                        continue
                    if isinstance(default, int) and not isinstance(default, bool):
                        try: value = int(value)
                        except Exception: continue
                    if isinstance(default, float):
                        try: value = float(value)
                        except Exception: continue
                    if isinstance(default, str) and value is not None:
                        value = str(value)
                if key in VALUE_CHOICES and value not in VALUE_CHOICES[key]:
                    continue
                if key in ("cli_path", "input_paths", "output_root"):
                    value = str(value or "")
                cfg[key] = value
        return cfg, f"已从 {CONFIG_PATH.name} 恢复上次配置。"
    except Exception as exc:
        # Preserve the damaged file for diagnosis; never prevent GUI startup.
        try:
            backup = CONFIG_PATH.with_name(CONFIG_PATH.name + f".corrupt-{int(time.time())}")
            shutil.copy2(CONFIG_PATH, backup)
        except Exception:
            pass
        return cfg, f"配置文件读取失败，已回退默认值：{type(exc).__name__}: {exc}"


_CONFIG_LOCK = threading.RLock()

def save_config_values(values: Dict[str, Any]) -> str:
    try:
        with _CONFIG_LOCK:
            current, _ = safe_config_load()
            current.update(values)
            atomic_write_json(CONFIG_PATH, current)
        return f"自动保存成功 · {dt.datetime.now().strftime('%H:%M:%S')}"
    except Exception as exc:
        return f"配置保存失败：{type(exc).__name__}: {exc}"


def load_history_records() -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("历史记录文件根节点不是列表")
        records = [r for r in raw if isinstance(r, dict) and r.get("id")]
        # A process that was active during a previous app session cannot be resumed safely.
        changed = False
        for rec in records:
            if rec.get("status") in {"queued", "running", "paused", "cancelling"}:
                rec["status"] = "interrupted"
                rec.setdefault("logs", []).append({"time": now_iso(), "text": "应用重启：历史任务不会自动恢复，请从历史记录重新载入并提交。"})
                rec["finished_at"] = now_iso()
                changed = True
        if changed:
            atomic_write_json(HISTORY_PATH, records[-MAX_HISTORY:])
        return records[-MAX_HISTORY:]
    except Exception as exc:
        try:
            shutil.copy2(HISTORY_PATH, HISTORY_PATH.with_name(HISTORY_PATH.name + f".corrupt-{int(time.time())}"))
        except Exception:
            pass
        print(f"[History] Failed to load history: {exc}", file=sys.stderr)
        return []


from seedvr2_runtime import InProcessTaskManager, TaskAborted  # same-process inference engine

MANAGER = InProcessTaskManager()


def get_model_choices(cli_path_value: str) -> List[str]:
    cli = Path(cli_path_value).expanduser() if cli_path_value else DEFAULT_CLI
    return load_model_choices(cli.resolve())


# -----------------------------------------------------------------------------
# Native file/folder dialogs and local device discovery
# -----------------------------------------------------------------------------

def _native_dialog(dialog_kind: str, current: str = "", title: str = "选择路径") -> Tuple[Optional[str], str]:
    """Open a native OS dialog. Works best when Gradio runs on the local desktop."""
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        current_path = Path(current).expanduser() if current else None
        initial_dir = None
        if current_path:
            initial_dir = str(current_path if current_path.is_dir() else current_path.parent)
            if not Path(initial_dir).exists():
                initial_dir = None
        if dialog_kind == "file":
            chosen = filedialog.askopenfilename(
                parent=root, title=title, initialdir=initial_dir,
                filetypes=[("SeedVR2 CLI", "inference_cli.py"), ("Python files", "*.py"), ("All files", "*.*")],
            )
        elif dialog_kind == "media_files":
            chosen = filedialog.askopenfilenames(
                parent=root, title=title, initialdir=initial_dir,
                filetypes=[("Media files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
            )
            return ([str(x) for x in chosen] if chosen else None), "已取消选择。" if not chosen else f"已选择 {len(chosen)} 个文件。"
        else:
            chosen = filedialog.askdirectory(parent=root, title=title, initialdir=initial_dir, mustexist=True)
        if not chosen:
            return None, "已取消选择，原路径保持不变。"
        return str(chosen), "路径选择成功。"
    except Exception as exc:
        return None, f"无法打开系统资源管理器对话框：{type(exc).__name__}: {exc}。也可以直接在路径框中粘贴完整路径。"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _browse_cli(current: str, current_model: str, current_blocks: Any) -> Tuple[Any, Any, Any, str]:
    chosen, note = _native_dialog("file", current, "选择 inference_cli.py")
    if not chosen:
        return gr.update(), gr.update(), gr.update(), note
    path = Path(chosen).expanduser().resolve()
    if path.name.lower() != "inference_cli.py":
        return gr.update(), gr.update(), gr.update(), "请选择文件名为 inference_cli.py 的 CLI 文件。"
    models = load_model_choices(path)
    model = current_model if current_model in models else (models[0] if models else current_model)
    if model and model not in models:
        models.append(model)
    block_update, block_note = _on_model_change(model, current_blocks)
    save_config_values({"cli_path": str(path), "dit_model": model})
    return str(path), gr.update(choices=models, value=model), block_update, f"CLI 路径已更新；检测到 {len(models)} 个模型候选项。{block_note}"


def _browse_directory(current: str, kind: str) -> Tuple[Any, str]:
    title = {"output": "选择输出根目录", "model": "选择 SeedVR2 模型目录", "input": "选择输入目录"}.get(kind, "选择目录")
    chosen, note = _native_dialog("directory", current, title)
    if not chosen:
        return gr.update(), note
    config_key = {"output": "output_root", "model": "model_dir", "input": "input_paths"}.get(kind)
    if config_key:
        save_config_values({config_key: chosen})
    return chosen, note


def _browse_model_file(current: str, model_kind: str) -> Tuple[str, str]:
    title = "选择 DiT 主干模型文件" if model_kind == "dit" else "选择 VAE 模型文件"
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        current_path = Path(current).expanduser() if current else None
        initial_dir = str(current_path.parent if current_path and current_path.is_file() else current_path) if current_path else None
        if initial_dir and not Path(initial_dir).exists():
            initial_dir = None
        chosen = filedialog.askopenfilename(
            parent=root, title=title, initialdir=initial_dir,
            filetypes=[("Model weights", "*.safetensors *.gguf *.pt *.pth *.bin *.ckpt"), ("All files", "*.*")],
        )
        if not chosen:
            return current or "", "已取消模型文件选择，原设置保持不变。"
        return str(Path(chosen).resolve()), f"已选择 {model_kind.upper()} 模型：`{chosen}`"
    except Exception as exc:
        return current or "", f"无法打开原生文件选择器：{type(exc).__name__}: {exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _browse_dit_model_file(current: str) -> Tuple[str, str]:
    value, note = _browse_model_file(current, "dit")
    save_config_values({"dit_model_path": value})
    return value, note


def _browse_vae_model_file(current: str) -> Tuple[str, str]:
    value, note = _browse_model_file(current, "vae")
    save_config_values({"vae_model": value})
    return value, note


def _browse_input_files(current_text: str) -> Tuple[str, str]:
    chosen, note = _native_dialog("media_files", "", "选择一个或多个媒体文件")
    if not chosen:
        return current_text or "", note
    existing = split_input_paths(current_text)
    for item in chosen:
        if item not in existing:
            existing.append(item)
    value = "\n".join(existing)
    save_config_values({"input_paths": value})
    return value, note


def _browse_input_folder(current_text: str) -> Tuple[str, str]:
    chosen, note = _native_dialog("directory", "", "选择输入媒体目录")
    if not chosen:
        return current_text or "", note
    existing = split_input_paths(current_text)
    if chosen not in existing:
        existing.append(chosen)
    value = "\n".join(existing)
    save_config_values({"input_paths": value})
    return value, note


def detect_local_devices() -> Dict[str, Any]:
    """Query the active interpreter directly so the GUI uses its current torch build."""
    result: Dict[str, Any] = {"backend": "cpu", "cuda": [], "mps": False, "note": "未检测到 CUDA GPU；仍可手动编辑设备值。"}
    try:
        import torch
        cuda = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend and mps_backend.is_available())
        result.update({"cuda": cuda, "mps": mps, "backend": "cuda" if cuda else ("mps" if mps else "cpu")})
        result["note"] = f"检测到 {len(cuda)} 张 CUDA GPU。" if cuda else ("检测到 Apple MPS 设备。" if mps else "PyTorch 未检测到 CUDA/MPS 设备。")
        if cuda or mps:
            return result
    except Exception as exc:
        result["note"] = f"当前 Python 的 PyTorch 设备查询失败：{type(exc).__name__}: {exc}。尝试使用 nvidia-smi。"
    try:
        cp = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=8, encoding="utf-8", errors="replace")
        if cp.returncode == 0:
            cuda = [x.strip() for x in cp.stdout.splitlines() if x.strip()]
            if cuda:
                result.update({"backend": "cuda", "cuda": cuda, "note": f"通过 nvidia-smi 检测到 {len(cuda)} 张 GPU；PyTorch 设备查询未成功。"})
    except Exception:
        pass
    return result


def build_device_choices(device_info: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    cuda_names = device_info.get("cuda", [])
    cuda_choices: List[Tuple[str, str]] = [("默认设备（留空）", "")]
    for i, name in enumerate(cuda_names):
        cuda_choices.append((f"GPU {i} · {name}", str(i)))
    # Offer common multi-GPU combinations, but keep the list compact.
    if len(cuda_names) > 1:
        for n in range(2, min(len(cuda_names), 4) + 1):
            for combo in itertools.combinations(range(len(cuda_names)), n):
                ids = ",".join(str(x) for x in combo)
                cuda_choices.append((f"GPU {ids} · 多卡", ids))
    offload_choices: List[Tuple[str, str]] = [("none · 不卸载", "none"), ("cpu · 系统内存", "cpu")]
    if device_info.get("backend") == "mps":
        offload_choices.append(("mps · Apple GPU", "mps"))
    else:
        for i, name in enumerate(cuda_names):
            offload_choices.append((f"GPU {i} · {name}", str(i)))
            offload_choices.append((f"cuda:{i} · 显式设备", f"cuda:{i}"))
    return cuda_choices, offload_choices


def _strict_int(value: Any, key: str) -> int:
    """Convert an integer-like UI value without accepting fractions, NaN or infinity."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} 必须是整数，当前值为 {value!r}。")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{key} 必须是整数，当前值为 {value!r}。") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{key} 必须是有限整数，当前值为 {value!r}。")
    return int(number)


def _strict_float(value: Any, key: str) -> float:
    """Convert a numeric UI value to a finite float; reject bool/blank/non-finite input."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} 必须是有效数值，当前值为 {value!r}。")
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} 必须是有效数值，当前值为 {value!r}。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} 必须是有限数值，当前值为 {value!r}。")
    return number


def _strict_bool(value: Any, key: str) -> bool:
    """Normalize Gradio checkbox-like values to an actual bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{key} 必须是布尔值，当前值为 {value!r}。")


def _normalize_parameter_value(key: str, value: Any) -> Any:
    """Return one parameter in the exact Python type expected by the CLI/Runtime."""
    if key not in DEFAULTS:
        raise KeyError(f"未知参数：{key}")

    default = DEFAULTS[key]
    if isinstance(default, bool):
        normalized = _strict_bool(value, key)
    elif isinstance(default, int) and not isinstance(default, bool):
        normalized = _strict_int(value, key)
        bounds = INT_PARAMETER_RANGES.get(key)
        if bounds is not None and not bounds[0] <= normalized <= bounds[1]:
            raise ValueError(f"{key} 必须在 {bounds[0]} 到 {bounds[1]} 之间，当前值为 {normalized}。")
        if key == "batch_size" and normalized % 4 != 1:
            raise ValueError("batch_size 必须满足 4n+1，例如 1、5、9、13、17。")
    elif isinstance(default, float):
        normalized = _strict_float(value, key)
        bounds = FLOAT_PARAMETER_RANGES.get(key)
        if bounds is not None and not bounds[0] <= normalized <= bounds[1]:
            raise ValueError(f"{key} 必须在 {bounds[0]} 到 {bounds[1]} 之间，当前值为 {normalized}。")
    else:
        normalized = str(default if value is None else value).strip()

    if key in VALUE_CHOICES and normalized not in VALUE_CHOICES[key]:
        allowed = ", ".join(sorted(VALUE_CHOICES[key]))
        raise ValueError(f"{key} 只能选择：{allowed}。")

    # Normalize device tokens consistently while keeping them strings for the CLI.
    if key in {"dit_offload_device", "vae_offload_device", "tensor_offload_device"}:
        normalized = normalized.lower()
    if key == "cuda_device":
        normalized = normalized.replace(" ", "")
    return normalized


def normalize_ui_parameters(values: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize every UI parameter; invalid fields fall back internally but are reported.

    Callers must not submit when errors is non-empty. Using defaults for invalid fields
    keeps follow-up validation/preview code from crashing while the user sees the error.
    """
    normalized: Dict[str, Any] = {}
    errors: List[str] = []
    for key, default in DEFAULTS.items():
        raw_value = values.get(key, default)
        try:
            normalized[key] = _normalize_parameter_value(key, raw_value)
        except (TypeError, ValueError, OverflowError, KeyError) as exc:
            normalized[key] = default
            errors.append(f"{key}：{exc}")
    return normalized, errors


def _on_numeric_parameter_change(key: str, value: Any) -> Tuple[Any, str]:
    """Convert hand-entered Dropdown/Number/Slider values and autosave only valid data."""
    try:
        normalized = _normalize_parameter_value(key, value)
    except (TypeError, ValueError, OverflowError) as exc:
        return gr.update(), f"⚠️ {key} 输入无效，未保存：{exc}"

    save_note = save_config_values({key: normalized})
    changed_type = not isinstance(value, type(normalized))
    changed_value = value != normalized
    if changed_type or changed_value:
        note = f"✓ {key} 已转换为 {type(normalized).__name__}：{normalized}。{save_note}"
    else:
        note = f"✓ {key} 参数合法。{save_note}"
    return gr.update(value=normalized), note


def _normalise_batch_size(value: Any, persist: bool = True) -> Tuple[Any, str]:
    """Preserve the UI's existing 4n+1 correction while always returning an int."""
    try:
        numeric = _strict_float(value, "batch_size")
    except (TypeError, ValueError, OverflowError):
        numeric = float(DEFAULTS["batch_size"])
        source_note = f"输入 {value!r} 不是有效数字，已回退到默认值 5；"
    else:
        source_note = ""
    rounded = int(round(numeric))
    bounded = max(1, min(rounded, 1001))
    corrected = 1 + 4 * int(round((bounded - 1) / 4))
    corrected = max(1, min(corrected, 1001))
    if not math.isclose(numeric, corrected, rel_tol=0.0, abs_tol=1e-9):
        msg = f"{source_note}Batch Size 输入 {value!r} 不满足整数且 4n+1 的约束，已自动修正为 {corrected}。"
    else:
        msg = f"{source_note}Batch Size 合法：{corrected}（满足 4n+1）。"
    if persist:
        save_config_values({"batch_size": corrected})
    return gr.update(value=corrected), msg


def _model_block_limit(model_name: str) -> int:
    """Conservative block-swap limit inferred from registry label or selected filename."""
    return 36 if "7b" in str(model_name).lower() else 32


def _on_dit_path_change(model_path: str, current_blocks: Any) -> Tuple[Any, str]:
    selected = str(model_path or "").strip()
    if not selected:
        return _on_model_change(DEFAULTS["dit_model"], current_blocks)
    limit = _model_block_limit(Path(selected).name)
    try:
        value = _strict_int(current_blocks or 0, "blocks_to_swap")
    except (TypeError, ValueError, OverflowError):
        value = 0
    fixed = min(max(0, value), limit)
    save_config_values({"dit_model_path": selected, "blocks_to_swap": fixed})
    choices = [0, 4, 8, 12, 16, 20, 24, 28, 32] + ([36] if limit == 36 else [])
    note = f"根據自選 DiT 檔名推斷 BlockSwap 上限為 {limit}。"
    if fixed != value:
        note += f" 已將目前值從 {value} 修正為 {fixed}。"
    return gr.update(value=fixed, choices=choices), note


def _on_model_change(model_name: str, current_blocks: Any) -> Tuple[Any, str]:
    limit = _model_block_limit(model_name)
    try:
        value = _strict_int(current_blocks or 0, "blocks_to_swap")
    except (TypeError, ValueError, OverflowError):
        value = 0
    fixed = min(max(0, value), limit)
    note = f"已切换模型；当前模型 BlockSwap 建议范围为 0–{limit}。"
    if fixed != value:
        note += f" 当前值已从 {value} 调整为 {fixed}。"
    save_config_values({"dit_model": model_name, "blocks_to_swap": fixed})
    return gr.update(value=fixed, choices=[0, 4, 8, 12, 16, 20, 24, 28, 32] + ([36] if limit == 36 else [])), note


def _on_blocks_change(blocks: Any, dit_offload: str, swap_io: bool, model_name: str) -> Tuple[Any, Any, str]:
    try:
        value = max(0, _strict_int(blocks or 0, "blocks_to_swap"))
    except (TypeError, ValueError, OverflowError):
        return gr.update(), gr.update(), "BlockSwap 必须是整数；未修改当前配置。"
    limit = _model_block_limit(model_name)
    fixed = min(value, limit)
    block_update = gr.update(value=fixed) if (fixed != value or not isinstance(blocks, int) or isinstance(blocks, bool)) else gr.update()
    if fixed != value:
        save_config_values({"blocks_to_swap": fixed})
        note = f"当前模型 BlockSwap 上限为 {limit}，已将 {value} 修正为 {fixed}。"
    else:
        note = "显存参数设置正常。"
    effective_offload = "cpu" if (fixed > 0 or swap_io) and str(dit_offload) == "none" else str(dit_offload)
    save_config_values({"blocks_to_swap": fixed, "dit_offload_device": effective_offload, "swap_io_components": bool(swap_io)})
    if effective_offload != str(dit_offload):
        return block_update, gr.update(value=effective_offload), note + " DiT Offload Device 自动调整为 cpu。"
    return block_update, gr.update(), note


def _on_tiling_toggle(enabled: bool, tile_size: Any, tile_overlap: Any, overlap_key: str) -> Tuple[Any, Any, str]:
    try:
        size = _strict_int(tile_size, "tile_size")
        overlap = _strict_int(tile_overlap, "tile_overlap")
    except (TypeError, ValueError, OverflowError):
        save_config_values({"vae_encode_tiled" if overlap_key.startswith("vae_encode") else "vae_decode_tiled": bool(enabled)})
        return gr.update(interactive=bool(enabled)), gr.update(interactive=bool(enabled)), "Tile Size / Overlap 必须为整数，请修正后提交。"
    tiled_key = "vae_encode_tiled" if overlap_key.startswith("vae_encode") else "vae_decode_tiled"
    if enabled and overlap >= size:
        fixed = max(0, min(128, size // 4))
        save_config_values({tiled_key: bool(enabled), overlap_key: fixed})
        return gr.update(interactive=True), gr.update(interactive=True, value=fixed), f"Tile Overlap 必须小于 Tile Size，已修正为 {fixed}。"
    save_config_values({tiled_key: bool(enabled), overlap_key: overlap})
    return (
        gr.update(interactive=bool(enabled), value=size),
        gr.update(interactive=bool(enabled), value=overlap),
        "VAE Tiling 参数状态已更新。",
    )


def _on_video_backend_change(backend: str, use_10bit: bool) -> Tuple[Any, str]:
    effective_10bit = bool(use_10bit and backend == "ffmpeg")
    save_config_values({"video_backend": backend, "use_10bit": effective_10bit})
    if not effective_10bit and use_10bit:
        return gr.update(value=False), "已切换为 OpenCV，自动关闭 10-bit；10-bit 需要 FFmpeg 后端。"
    return gr.update(), "FFmpeg 可用时可启用 10-bit x265 输出。"


def _on_10bit_change(enabled: bool) -> Tuple[Any, str]:
    if enabled and shutil.which("ffmpeg") is None:
        save_config_values({"use_10bit": False})
        return gr.update(value=False), "系统 PATH 中未找到 ffmpeg，无法启用 10-bit；已恢复关闭状态。"
    if enabled:
        save_config_values({"video_backend": "ffmpeg", "use_10bit": True})
        return gr.update(value="ffmpeg"), "已自动切换为 FFmpeg 后端；10-bit 使用 x265 编码。"
    save_config_values({"use_10bit": False})
    return gr.update(), "10-bit 已关闭。"


def _on_tile_overlap_change(overlap: Any, tile_size: Any, key: str) -> Tuple[Any, str]:
    try:
        ov, size = _strict_int(overlap, "tile_overlap"), _strict_int(tile_size, "tile_size")
    except (TypeError, ValueError, OverflowError):
        return gr.update(), "Tile 参数必须为整数；请修正后提交。"
    if ov < 0:
        save_config_values({key: 0})
        return gr.update(value=0), "Tile Overlap 不能小于 0，已修正为 0。"
    if ov >= size:
        fixed = max(0, min(128, size // 4))
        save_config_values({key: fixed})
        return gr.update(value=fixed), f"Overlap 必须小于 Tile Size，已自动修正为 {fixed}。"
    save_config_values({key: ov})
    update = gr.update(value=ov) if (not isinstance(overlap, int) or isinstance(overlap, bool)) else gr.update()
    return update, "Tile 参数合法。"


def _on_tile_size_change(tile_size: Any, overlap: Any, size_key: str, overlap_key: str) -> Tuple[Any, Any, str]:
    try:
        size, ov = _strict_int(tile_size, "tile_size"), _strict_int(overlap, "tile_overlap")
    except (TypeError, ValueError, OverflowError):
        return gr.update(), gr.update(), "Tile Size / Overlap 必须为整数；请修正后提交。"
    if size < 64:
        size = 64
        save_config_values({size_key: size})
    overlap_update = gr.update(value=ov) if (not isinstance(overlap, int) or isinstance(overlap, bool)) else gr.update()
    note = "Tile Size / Overlap 合法。"
    if ov < 0 or ov >= size:
        fixed = max(0, min(128, size // 4))
        save_config_values({overlap_key: fixed})
        overlap_update = gr.update(value=fixed)
        note = f"Tile Size 变化后，Overlap 已自动调整为 {fixed}（必须满足 0 ≤ Overlap < Tile Size）。"
    size_update = gr.update(value=size) if (size != _strict_int(tile_size, "tile_size") or not isinstance(tile_size, int) or isinstance(tile_size, bool)) else gr.update()
    prefix = "vae_encode" if size_key.startswith("vae_encode") else "vae_decode"
    save_config_values({size_key: size, prefix + "_tile_overlap": max(0, min(ov, size - 1)) if ov >= size or ov < 0 else ov})
    return size_update, overlap_update, note


def _on_cache_toggle(enabled: bool, offload_device: str, config_key: str) -> Tuple[Any, str]:
    target_key = "dit_offload_device" if config_key == "cache_dit" else "vae_offload_device"
    effective = "cpu" if enabled and str(offload_device) == "none" else str(offload_device)
    save_config_values({config_key: bool(enabled), target_key: effective})
    if effective != str(offload_device):
        return gr.update(value=effective), "已启用模型缓存，Offload Device 自动设为 cpu（与 CLI 缓存模式的默认卸载行为一致）。"
    return gr.update(), "缓存设置已更新。"


def _on_compile_toggle(compile_dit: bool, compile_vae: bool) -> Tuple[Any, Any, Any, Any, Any, Any]:
    enabled = bool(compile_dit or compile_vae)
    update = gr.update(interactive=enabled)
    return update, update, update, update, update, update



def _on_chunk_overlap_change(chunk_size: Any, overlap: Any) -> Tuple[Any, Any, str]:
    """Keep Temporal Overlap smaller than Chunk Size whenever streaming is enabled."""
    try:
        chunk = _strict_int(chunk_size, "chunk_size")
        ov = _strict_int(overlap, "temporal_overlap")
    except (TypeError, ValueError, OverflowError):
        return gr.update(), gr.update(), "Chunk Size 与 Temporal Overlap 必须为整数。"
    if chunk < 0 or ov < 0:
        fixed_chunk = max(0, chunk)
        fixed_overlap = max(0, ov)
        save_config_values({"chunk_size": fixed_chunk, "temporal_overlap": fixed_overlap})
        return gr.update(value=fixed_chunk), gr.update(value=fixed_overlap), "Chunk Size / Temporal Overlap 不能为负数，已修正。"
    if chunk > 0 and ov >= chunk:
        fixed = max(0, chunk - 1)
        save_config_values({"chunk_size": chunk, "temporal_overlap": fixed})
        chunk_update = gr.update(value=chunk) if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)) else gr.update()
        return chunk_update, gr.update(value=fixed), f"流式处理要求 Temporal Overlap 小于 Chunk Size；已将 Overlap 自动调整为 {fixed}。"
    save_config_values({"chunk_size": chunk, "temporal_overlap": ov})
    chunk_update = gr.update(value=chunk) if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)) else gr.update()
    overlap_update = gr.update(value=ov) if (not isinstance(overlap, int) or isinstance(overlap, bool)) else gr.update()
    return chunk_update, overlap_update, "Chunk Size 与 Temporal Overlap 设置正常。"

def _on_max_resolution_change(resolution: Any, max_resolution: Any) -> str:
    try:
        r = _strict_int(resolution, "resolution")
        m = _strict_int(max_resolution, "max_resolution")
        if not 1 <= r <= 16384 or not 0 <= m <= 32768:
            return "分辨率参数超出允许范围。"
        if m > 0 and m < r:
            return f"⚠️ 最大边长 {m}px 小于目标短边 {r}px，可能产生不符合预期的缩放限制；建议设为 0 或不小于 {r}。"
    except (TypeError, ValueError, OverflowError):
        return "分辨率参数必须是合法整数。"
    return "分辨率设置正常。"



def validate_settings(p: Dict[str, Any], cli_path: str, output_root: str, inputs: List[str]) -> List[str]:
    # Keep this validator safe when called independently of the normal submit path.
    p, normalization_errors = normalize_ui_parameters(p)
    errors: List[str] = list(normalization_errors)
    cli = Path(str(cli_path or "")).expanduser()
    if not cli.is_file():
        errors.append(f"找不到 CLI：{cli}")

    if not str(output_root or "").strip():
        errors.append("输出根目录不能为空。")
    else:
        out_path = Path(output_root).expanduser()
        if out_path.exists() and not out_path.is_dir():
            errors.append(f"输出根目录不是文件夹：{out_path}")

    dit_path = str(p.get("dit_model_path", "") or "").strip()
    if dit_path:
        model_file = Path(dit_path).expanduser()
        if not model_file.is_file():
            errors.append(f"DiT 模型文件不存在：{model_file}")
    vae_value = str(p.get("vae_model", "") or "").strip()
    if vae_value:
        vae_candidate = Path(vae_value).expanduser()
        if (vae_candidate.is_absolute() or vae_candidate.parent != Path(".")) and not vae_candidate.is_file():
            errors.append(f"VAE 模型文件不存在：{vae_candidate}")

    model_dir = str(p.get("model_dir", "") or "").strip()
    if model_dir:
        model_path = Path(model_dir).expanduser()
        if not model_path.exists() or not model_path.is_dir():
            errors.append(f"模型目录不存在或不是文件夹：{model_path}")

    if not inputs:
        errors.append("请至少添加一个文件或目录输入路径。")
    valid_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
                 ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    image_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            errors.append(f"输入不存在：{path}")
        elif path.is_file() and path.suffix.lower() not in valid_ext:
            errors.append(f"不支持的输入格式：{path.suffix}（{path}）")
        elif path.is_dir() and not any(x.is_file() and x.suffix.lower() in valid_ext for x in path.iterdir()):
            errors.append(f"目录中没有 CLI 支持的媒体文件：{path}")

    parsed: Dict[str, int] = {}
    for key, (minimum, maximum) in INT_PARAMETER_RANGES.items():
        if key == "batch_size":
            continue  # Validated separately below, including the 4n+1 constraint.
        try:
            value = _strict_int(p.get(key), key)
            parsed[key] = value
            if not minimum <= value <= maximum:
                errors.append(f"{key} 必须在 {minimum} 到 {maximum} 之间。")
        except Exception:
            errors.append(f"{key} 必须是合法整数（范围 {minimum}–{maximum}）。")

    try:
        batch_size = _strict_int(p.get("batch_size"), "batch_size")
        if not 1 <= batch_size <= 1001 or batch_size % 4 != 1:
            errors.append("batch_size 必须在 1–1001 之间并满足 4n+1，例如 1、5、9、13、17。")
    except Exception:
        errors.append("batch_size 必须是满足 4n+1 的整数。")

    model_choice = str(p.get("dit_model", ""))
    if cli.is_file():
        supported_models = load_model_choices(cli)
        if supported_models and model_choice not in supported_models and not str(p.get("dit_model_path", "") or "").strip():
            errors.append(f"当前 CLI 的 Model Registry 不包含模型：{model_choice}。请刷新模型列表、选择有效模型，或指定 DiT 模型文件路径。")
    model_name = str(p.get("dit_model_path") or model_choice).lower()
    model_block_limit = 36 if "7b" in model_name else 32
    if parsed.get("blocks_to_swap", 0) > model_block_limit:
        errors.append(f"当前模型 BlockSwap 上限为 {model_block_limit}。")

    for key, (minimum, maximum) in FLOAT_PARAMETER_RANGES.items():
        try:
            value = _strict_float(p.get(key), key)
            if not minimum <= value <= maximum:
                errors.append(f"{key} 必须在 {minimum} 到 {maximum} 之间。")
        except Exception:
            errors.append(f"{key} 必须是有限数值（范围 {minimum}–{maximum}）。")

    if p.get("vae_encode_tiled") and parsed.get("vae_encode_tile_overlap", 0) >= parsed.get("vae_encode_tile_size", 64):
        errors.append("VAE Encode Tile Overlap 必须小于 Encode Tile Size。")
    if p.get("vae_decode_tiled") and parsed.get("vae_decode_tile_overlap", 0) >= parsed.get("vae_decode_tile_size", 64):
        errors.append("VAE Decode Tile Overlap 必须小于 Decode Tile Size。")
    if parsed.get("chunk_size", 0) > 0 and parsed.get("temporal_overlap", 0) >= parsed.get("chunk_size", 1):
        errors.append("启用流式分块时，Temporal Overlap 必须小于 Chunk Size。")
    if (parsed.get("blocks_to_swap", 0) > 0 or p.get("swap_io_components")) and str(p.get("dit_offload_device", "none")) == "none":
        errors.append("启用 BlockSwap / Swap I/O 时，DiT Offload Device 不能为 none。")

    detected = globals().get("DEVICE_INFO_CACHE") or {}
    gpu_count = len(detected.get("cuda", []))
    has_mps = bool(detected.get("mps")) or detected.get("backend") == "mps"
    cuda_ids = str(p.get("cuda_device", "") or "").strip()
    if cuda_ids:
        if not re.fullmatch(r"\d+(,\d+)*", cuda_ids):
            errors.append("cuda_device 格式应为设备数字 ID，例如 0 或 0,1；留空使用默认设备。")
        else:
            ids = cuda_ids.split(",")
            if len(set(ids)) != len(ids):
                errors.append("CUDA Device 不允许重复填写相同 GPU ID。")
            if gpu_count and any(int(x) >= gpu_count for x in ids):
                errors.append(f"本机检测到 {gpu_count} 张 CUDA GPU，编号范围为 0–{gpu_count - 1}。")
            elif not gpu_count and detected.get("backend") == "cpu":
                errors.append("当前 Python 环境未检测到 CUDA GPU，不能指定 CUDA Device。")

    for key in ("dit_offload_device", "vae_offload_device", "tensor_offload_device"):
        value = str(p.get(key, "none") or "none").strip().lower()
        if value not in {"none", "cpu", "mps"} and not re.fullmatch(r"(?:cuda:)?\d+", value):
            errors.append(f"{key} 必须为 none、cpu、mps、GPU 编号或 cuda:N。")
        elif value == "mps" and not has_mps:
            errors.append(f"{key} 选择了 mps，但本机未检测到可用的 Apple MPS。")
        elif value not in {"none", "cpu", "mps"}:
            device_id = int(value.split(":")[-1])
            if gpu_count and device_id >= gpu_count:
                errors.append(f"{key} 指定了不存在的 GPU 编号 {device_id}。")
            elif not gpu_count and detected.get("backend") == "cpu":
                errors.append(f"{key} 指定了 GPU {device_id}，但当前 Python 环境没有检测到 CUDA GPU。")

    for key, allowed in VALUE_CHOICES.items():
        if p.get(key) not in allowed:
            errors.append(f"{key} 只能选择：{', '.join(sorted(allowed))}。")

    if p.get("output_format") == "mp4":
        for source in inputs:
            candidate = Path(source).expanduser()
            if candidate.is_file() and candidate.suffix.lower() in image_ext:
                errors.append(f"output_format=mp4 不能用于图片输入：{candidate}。请选择 auto 或 png。")
            elif candidate.is_dir() and any(x.is_file() and x.suffix.lower() in image_ext for x in candidate.iterdir()):
                errors.append(f"目录包含图片，不能对整个目录强制 output_format=mp4：{candidate}。请选择 auto 或 png。")

    if p.get("use_10bit") and p.get("video_backend") != "ffmpeg":
        errors.append("10-bit 输出要求 video_backend=ffmpeg。")
    if p.get("video_backend") == "ffmpeg" and shutil.which("ffmpeg") is None:
        errors.append("video_backend=ffmpeg，但系统 PATH 中未找到 ffmpeg。")
    if p.get("use_10bit") and shutil.which("ffmpeg") is None:
        errors.append("找不到 ffmpeg，无法进行 10-bit 输出。")
    if platform.system() == "Darwin" and (p.get("blocks_to_swap") or p.get("swap_io_components")):
        errors.append("当前 CLI 在 macOS 上不支持 BlockSwap / Swap I/O。")
    if platform.system() == "Darwin" and cuda_ids:
        errors.append("当前 CLI 在 macOS 上不提供 --cuda_device 参数。")
    return errors

def split_input_paths(text: str) -> List[str]:
    paths: List[str] = []
    seen = set()
    for raw in (text or "").splitlines():
        value = raw.strip().strip('"').strip("'")
        if not value:
            continue
        key = os.path.normcase(os.path.abspath(os.path.expanduser(value)))
        if key not in seen:
            seen.add(key)
            paths.append(value)
    return paths


def config_from_ui(values: List[Any]) -> Dict[str, Any]:
    return {key: value for key, value in zip(PARAM_KEYS, values)}


def _settings_changed(key: str, value: Any) -> str:
    try:
        normalized = _normalize_parameter_value(key, value) if key in DEFAULTS else value
    except (TypeError, ValueError, OverflowError) as exc:
        return f"参数未保存：{key} 输入无效：{exc}"
    return save_config_values({key: normalized})


def _merge_uploads(files: Any, current_text: str) -> str:
    values = []
    if files:
        if not isinstance(files, (list, tuple)):
            files = [files]
        for item in files:
            path = getattr(item, "name", None) or getattr(item, "path", None) or str(item)
            if path:
                values.append(str(path))
    existing = split_input_paths(current_text)
    combined = existing + [p for p in values if p not in existing]
    result = "\n".join(combined)
    save_config_values({"input_paths": result})
    return result


def _history_choices() -> List[Tuple[str, str]]:
    snap = MANAGER.snapshot()
    records = sorted(snap["records"], key=lambda r: r.get("created_at", ""), reverse=True)
    choices = []
    for r in records:
        created = str(r.get("created_at", ""))[:19].replace("T", " ")
        label = f"{created} | {r.get('status', '?').upper()} | {r.get('name', r.get('input', ''))} | {r.get('id', '')[:8]}"
        choices.append((label, r["id"]))
    return choices


def _load_config_outputs(components: List[Any]) -> Tuple[Any, ...]:
    cfg, note = safe_config_load()
    saved_cli = cfg.get("cli_path", str(DEFAULT_CLI))
    available_models = load_model_choices(Path(saved_cli).expanduser().resolve())
    if cfg.get("dit_model") not in available_models and not cfg.get("dit_model_path"):
        cfg["dit_model"] = available_models[0] if available_models else DEFAULTS["dit_model"]
        note += "（已将当前不可用的 DiT 模型名替换为可用候选项；指定模型文件路径时不覆盖路径设置。）"
    values = [saved_cli, cfg.get("input_paths", ""), cfg.get("output_root", str(APP_DIR / "outputs"))]
    values.extend(cfg.get(k, DEFAULTS[k]) for k in PARAM_KEYS)
    return tuple(values + [note])


def _submit(input_paths_text: str, cli_path: str, output_root: str, *ui_params: Any) -> Tuple[str, Any, Any]:
    raw_params = config_from_ui(list(ui_params))

    # Keep the established Batch Size auto-correction, then normalize all fields.
    batch_update, batch_note = _normalise_batch_size(raw_params.get("batch_size", DEFAULTS["batch_size"]), persist=False)
    corrected_batch = _strict_int(batch_update.get("value", DEFAULTS["batch_size"]), "batch_size")
    raw_params["batch_size"] = corrected_batch
    p, normalization_errors = normalize_ui_parameters(raw_params)

    paths = split_input_paths(input_paths_text)
    errors = normalization_errors
    if not errors:
        errors = validate_settings(p, cli_path, output_root, paths)
    if errors:
        return (
            "❌ 提交失败，未调用 SeedVR2 Runtime：\n"
            + "\n".join(f"• {x}" for x in errors)
            + f"\n\n{batch_note}",
            gr.update(choices=_history_choices()),
            gr.update(value=corrected_batch),
        )

    # p now contains native Python int/float/bool/str values. This exact dictionary
    # is persisted and passed to InProcessTaskManager; do not pass raw Gradio values.
    saved = {"cli_path": cli_path, "input_paths": input_paths_text, "output_root": output_root, **p}
    save_note = save_config_values(saved)
    count, message = MANAGER.submit(paths, p, cli_path, output_root)
    if count:
        return f"✅ {message}\n{save_note}\n{batch_note}", gr.update(choices=_history_choices()), gr.update(value=corrected_batch)
    return f"ℹ️ {message}\n{batch_note}", gr.update(choices=_history_choices()), gr.update(value=corrected_batch)

def _pause() -> str:
    return MANAGER.pause()


def _resume() -> str:
    return MANAGER.resume()


def _stop() -> str:
    return MANAGER.stop_all()


def _safe_stop() -> str:
    return MANAGER.safe_stop()


def _clear_model_cache() -> str:
    return MANAGER.clear_model_cache()


def _refresh() -> Tuple[str, str, str, List[List[Any]], Any, Any, Any, Any]:
    snap = MANAGER.snapshot()
    current = snap["current"]
    queued = snap["pending"]
    if current:
        status = current.get("status", "running").upper()
        title = html.escape(current.get("name", "active task"))
        pct = max(0, min(100, int(current.get("progress", 0))))
        chunk_text = ""
        if current.get("chunk_total"):
            chunk_text = f" · Chunk {current.get('chunk_completed', 0)}/{current.get('chunk_total')}"
        label = f"{status} · {title} · {pct}%{chunk_text} · 等待队列 {queued}"
        bar = f'<div class="progress-wrap"><div class="progress-head"><span>{status} · {title}{chunk_text}</span><b>{pct}%</b></div><div class="progress-bg"><div class="progress-fill" style="width:{pct}%"></div></div><small>进度按已提交 Chunk / 文件更新；当前模型 Batch 推理过程以实时日志为准。等待队列：{queued}</small></div>'
        logs = current.get("logs", [])[-180:]
    else:
        status = "PAUSED" if snap["paused"] else ("QUEUE IDLE" if queued == 0 else "QUEUED")
        label = f"{status} · 等待任务 {queued}"
        bar = f'<div class="progress-wrap"><div class="progress-head"><span>{status}</span><b>{queued} pending</b></div><div class="progress-bg"><div class="progress-fill" style="width:0%"></div></div><small>同一个 Python 进程内串行调用 SeedVR2 核心；启用常驻模型缓存时，相同模型配置可跨任务复用。</small></div>'
        latest = sorted(snap["records"], key=lambda r: r.get("created_at", ""))[-1:] 
        logs = latest[0].get("logs", [])[-180:] if latest else []
    log_text = "\n".join(f"[{x.get('time','')[-8:]}] {x.get('text','')}" for x in logs)
    rows = []
    for r in sorted(snap["records"], key=lambda x: x.get("created_at", ""), reverse=True)[:80]:
        rows.append([r.get("id", "")[:8], r.get("status", ""), r.get("name", ""), f"{r.get('progress', 0)}%", pretty_time(r.get("elapsed_seconds", 0)), r.get("output_path", "")])
    active = bool(current or queued)
    return (label, bar, log_text, rows,
            gr.update(interactive=active), gr.update(interactive=active),
            gr.update(interactive=bool(current and not snap["paused"])),
            gr.update(interactive=bool(snap["paused"])))


def _preview(input_paths_text: str, cli_path: str, output_root: str, *ui_params: Any) -> str:
    paths = split_input_paths(input_paths_text)
    if not paths:
        return "请先填写输入路径。"
    p, parameter_errors = normalize_ui_parameters(config_from_ui(list(ui_params)))
    if parameter_errors:
        return "### 参数预检查未通过\n" + "\n".join(f"- {x}" for x in parameter_errors)
    keep = p.get("keep_models_resident", True)
    chunk = p["chunk_size"]
    devices = [x.strip() for x in str(p.get("cuda_device", "") or "0").split(",") if x.strip()]
    single_gpu = len(devices) <= 1
    plan = (
        "### 同进程执行方案\n"
        f"- 推理核心：`{Path(cli_path).resolve()}`（由当前 Gradio 进程直接 import，不启动 CLI 子进程）\n"
        f"- 输入任务：{len(paths)} 项；按队列顺序串行执行\n"
        f"- Chunk Size：`{chunk}`（0 表示整个视频作为一个 Chunk）\n"
        f"- Chunk 检查点：每个 Chunk 独立写入并关闭，原子更新 `checkpoint.json`\n"
        f"- 完成处理：所有 Chunk 完成后合并为最终视频\n"
        f"- 模型驻留：{'启用' if keep else '关闭'}；模型/设备/编译等配置变化时重建运行上下文\n"
        f"- GPU 路径：{'单 GPU；支持 Chunk 断点、常驻缓存和协作控制' if single_gpu else '多 GPU；回退到上游 multiprocessing 路径'}\n"
        f"- DiT：`{p.get('dit_model_path') or p.get('dit_model')}`\n"
        f"- VAE：`{p.get('vae_model') or '使用 SeedVR2 默认 VAE'}`\n"
        "- 恢复方式：重新提交相同输入与参数，程序从最后一个有效检查点继续。"
    )
    if not single_gpu:
        plan += "\n\n**多 GPU 限制**：此兼容路径不提供 Chunk 级断点、跨任务模型常驻，也不能在 worker 推理期间响应暂停/安全停止/立即停止；这些控制要等上游多进程任务自行结束后才能恢复。长视频如需可恢复和协作控制，请选择单 GPU。"
    return plan


def _restore_history(record_id: str) -> Tuple[Any, ...]:
    rec = MANAGER.get_record(record_id) if record_id else None
    if not rec:
        return tuple([gr.update()] * (3 + len(PARAM_KEYS)) + ["未选择有效历史记录。"])
    params = rec.get("params", {})
    restored_params, restore_errors = normalize_ui_parameters(params)
    values = [rec.get("cli_path", str(DEFAULT_CLI)), rec.get("input", ""), rec.get("output_root", str(APP_DIR / "outputs"))]
    values.extend(restored_params.get(k, DEFAULTS[k]) for k in PARAM_KEYS)
    restored_model = params.get("dit_model", DEFAULTS["dit_model"])
    choices = load_model_choices(Path(values[0]).expanduser().resolve())
    model_component_index = 3 + PARAM_KEYS.index("dit_model")
    if restored_model not in choices and choices:
        choices.append(restored_model)
    values[model_component_index] = gr.update(choices=choices, value=restored_model)
    save_values = {k: v for k, v in zip(PARAM_KEYS, values[3:])}
    save_values["dit_model"] = restored_model
    save_config_values({"cli_path": values[0], "input_paths": values[1], "output_root": values[2], **save_values})
    restore_note = f"已将历史任务 {rec.get('id')} 的输入与参数恢复到工作区。"
    if restore_errors:
        restore_note += " 以下历史参数无效，已回退默认值：" + "；".join(restore_errors)
    return tuple(values + [restore_note])


def _delete_history(record_id: str, confirmed: bool) -> Tuple[str, Any, Any]:
    if not confirmed:
        return "请先勾选删除确认。", gr.update(choices=_history_choices(), value=record_id), False
    ok, message = MANAGER.delete_record(record_id or "")
    return message, gr.update(choices=_history_choices(), value=None), False


def _clear_history(confirmed: bool) -> Tuple[str, Any, Any]:
    if not confirmed:
        return "请先勾选清空确认。", gr.update(choices=_history_choices()), False
    ok, message = MANAGER.clear_history()
    return message, gr.update(choices=_history_choices(), value=None), False


def _reload_model_choices(cli_path: str, current_model: str, current_blocks: Any) -> Tuple[Any, Any, str]:
    choices = load_model_choices(Path(cli_path).expanduser().resolve(), force_refresh=True)
    value = current_model if current_model in choices else (choices[0] if choices else None)
    if value:
        block_update, block_note = _on_model_change(value, current_blocks)
        save_config_values({"dit_model": value})
    else:
        block_update, block_note = gr.update(), ""
    return gr.update(choices=choices, value=value), block_update, f"模型列表已刷新：发现 {len(choices)} 个模型候选项。{block_note}"


def build_app() -> gr.Blocks:
    css = r"""
    :root { --seed-accent: #7c5cff; }
    .gradio-container { max-width: 1540px !important; margin: auto !important; }
    .seed-hero { padding: 28px 30px; border-radius: 22px; background: radial-gradient(circle at 90% 10%, rgba(33,211,238,.28), transparent 28%), linear-gradient(120deg,#17162e 0%,#29205a 48%,#13566b 100%); color: #fff; margin-bottom: 14px; box-shadow: 0 14px 36px rgba(17,24,39,.18); }
    .seed-hero h1 { margin: 0 0 7px 0; font-size: 30px; letter-spacing: -.7px; }
    .seed-hero p { margin: 0; color: rgba(255,255,255,.78); max-width: 980px; line-height: 1.6; }
    .seed-chip { display:inline-block; margin: 13px 7px 0 0; padding: 5px 10px; border: 1px solid rgba(255,255,255,.24); border-radius: 99px; font-size: 12px; color: #fff; background: rgba(255,255,255,.09); }
    .progress-wrap { padding: 14px 16px; border: 1px solid var(--border-color-primary); border-radius: 14px; background: var(--background-fill-secondary); }
    .progress-head { display:flex; justify-content:space-between; gap:12px; margin-bottom:9px; font-weight:600; }
    .progress-bg { height: 10px; background: var(--border-color-primary); border-radius: 20px; overflow:hidden; }
    .progress-fill { height:100%; background: linear-gradient(90deg,#7c5cff,#22c7d6); border-radius:20px; transition:width .35s ease; }
    .progress-wrap small { display:block; margin-top:9px; opacity:.72; line-height:1.45; }
    .seed-section { border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 14px; }
    .seed-muted { opacity:.72; font-size:13px; }
    .seed-actions button { min-height: 44px; }
    .seed-path-button button { min-width: 112px; }
    footer { display:none !important; }
    """
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate").set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_primary_text_color="white",
    )
    cfg, _ = safe_config_load()
    initial_cli = str(cfg.get("cli_path", DEFAULT_CLI))
    model_choices = load_model_choices(Path(initial_cli).expanduser())
    model_default = cfg.get("dit_model", DEFAULTS["dit_model"])
    if model_default not in model_choices:
        model_choices = model_choices + ([model_default] if model_default else [])
    device_info = detect_local_devices()
    global DEVICE_INFO_CACHE
    DEVICE_INFO_CACHE = device_info
    cuda_choices, offload_choices = build_device_choices(device_info)

    def opts(key: str, fallback: List[Any]) -> List[Any]:
        spec = next((x[3] for x in FIELD_SPECS if x[0] == key), {})
        return spec.get("choices", fallback)

    def num_dropdown(label: str, key: str, choices: List[int], info: str, scale: int = 1, interactive: bool = True):
        value = cfg.get(key, DEFAULTS[key])
        if value not in choices:
            choices = choices + [value]
        return gr.Dropdown(label=label, choices=choices, value=value, allow_custom_value=True, filterable=True, info=info, scale=scale, interactive=interactive)

    gradio_major = int(str(getattr(gr, "__version__", "4")).split(".")[0])
    blocks_kwargs = {"title": "SeedVR2 Workbench"}
    if gradio_major < 6:
        blocks_kwargs.update({"theme": theme, "css": css})
    else:
        blocks_kwargs["fill_width"] = True
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML("""<div class="seed-hero"><h1>SeedVR2 · Video Restoration Workbench</h1><p>面向本地推理的统一控制台：同进程 SeedVR2 推理、DiT/VAE 常驻缓存、Chunk 独立落盘与断点恢复、批量队列、模型文件选择和完整参数控制。</p><span class="seed-chip">In-Process Engine</span><span class="seed-chip">Resident DiT / VAE</span><span class="seed-chip">Chunk Checkpoint</span><span class="seed-chip">Resume & Safe Stop</span><span class="seed-chip">CPU Offload / BlockSwap</span></div>""")

        # Paths and input sources
        with gr.Group():
            gr.Markdown("### 路径与输入任务")
            with gr.Row():
                cli_path = gr.Textbox(label="SeedVR2 推理核心路径（同进程导入）", value=initial_cli, scale=5, interactive=False, info="此文件由当前 Gradio Python 进程直接导入；请将 GUI、运行时和 inference_cli.py 放在 SeedVR2 项目根目录，不再启动独立 CLI 子进程。")
                browse_cli_btn = gr.Button("核心路径固定", scale=1, variant="secondary", interactive=False, elem_classes=["seed-path-button"])
                output_root = gr.Textbox(label="输出根目录", value=cfg.get("output_root", str(APP_DIR / "outputs")), scale=4, info="每个输入任务会在该目录下创建独立输出子目录。")
                browse_output_btn = gr.Button("浏览输出…", scale=1, variant="secondary", elem_classes=["seed-path-button"])
            with gr.Row():
                input_paths = gr.Textbox(label="输入文件 / 目录路径（每行一项，可批量提交）", value=cfg.get("input_paths", ""), lines=4, scale=6, placeholder="D:\\videos\\clip01.mp4\nD:\\images\\test.png\nD:\\dataset\\folder", info="可以填写多个视频、图片或目录；目录会按 CLI 支持的媒体扩展名扫描。")
                with gr.Column(scale=2):
                    with gr.Row():
                        browse_input_files_btn = gr.Button("选择媒体文件…", variant="secondary")
                        browse_input_folder_btn = gr.Button("选择输入目录…", variant="secondary")
                    upload_files = gr.File(label="也可拖放/上传文件（可多选）", file_count="multiple", type="filepath")
                    gr.Markdown("<div class='seed-muted'>原生文件选择器适用于本机启动的 Gradio；远程部署时请直接填写服务端可访问的绝对路径。</div>")
            path_status = gr.Markdown("路径尚未更改。")
        upload_files.change(_merge_uploads, inputs=[upload_files, input_paths], outputs=input_paths, queue=False)

        comps: Dict[str, Any] = {}
        with gr.Tabs():
            with gr.Tab("模型与输出"):
                with gr.Row():
                    comps["dit_model"] = gr.Dropdown(label="DiT 模型名称 / 量化版本", choices=model_choices, value=model_default, allow_custom_value=True, filterable=True, info="从 model_registry 选择模型文件名；如使用独立权重路径，在下方指定路径，路径优先级更高。量化等级由权重文件决定。", scale=3)
                    reload_models_btn = gr.Button("刷新模型列表", scale=1, variant="secondary")
                with gr.Row():
                    comps["dit_model_path"] = gr.Textbox(label="DiT 主干模型文件路径（可选）", value=cfg.get("dit_model_path", ""), scale=5, placeholder="可填写完整 .safetensors / .gguf 路径", info="填写后优先使用该文件；留空则使用上面的 DiT 模型名和模型目录。")
                    browse_dit_model_btn = gr.Button("选择 DiT 文件…", scale=1, variant="secondary", elem_classes=["seed-path-button"])
                with gr.Row():
                    comps["vae_model"] = gr.Textbox(label="VAE 模型文件名 / 完整路径", value=cfg.get("vae_model", DEFAULTS["vae_model"]), scale=5, placeholder="VAE 模型名或完整模型文件路径", info="填写文件名时从模型目录解析；填写完整路径时直接使用该文件。")
                    browse_vae_model_btn = gr.Button("选择 VAE 文件…", scale=1, variant="secondary", elem_classes=["seed-path-button"])
                with gr.Row():
                    comps["model_dir"] = gr.Textbox(label="模型目录（留空使用默认目录）", value=cfg.get("model_dir", DEFAULTS["model_dir"]), scale=4, info="CLI --model_dir。DiT/VAE 文件名模式从该目录查找；如分别位于不同目录，可分别指定完整模型文件路径。")
                    browse_model_btn = gr.Button("浏览模型目录…", scale=1, variant="secondary", elem_classes=["seed-path-button"])
                with gr.Row():
                    comps["output_format"] = gr.Dropdown(label="输出格式", choices=["auto", "mp4", "png"], value=cfg.get("output_format", DEFAULTS["output_format"]), info="auto 按输入媒体类型自动选择；视频可保存 MP4，图片输出 PNG。")
                    comps["video_backend"] = gr.Dropdown(label="视频编码后端", choices=["opencv", "ffmpeg"], value=cfg.get("video_backend", DEFAULTS["video_backend"]), info="OpenCV 兼容性好；FFmpeg 可用于 10-bit x265。")
                    comps["use_10bit"] = gr.Checkbox(label="10-bit（FFmpeg x265）", value=cfg.get("use_10bit", False), info="仅 FFmpeg 后端支持；启用时会自动切换到 FFmpeg。")
            with gr.Tab("处理参数与画质"):
                with gr.Row():
                    comps["resolution"] = num_dropdown("目标短边分辨率", "resolution", [256, 384, 480, 512, 640, 720, 768, 896, 1024, 1080, 1280, 1440, 1536, 1920], "输出短边目标像素尺寸；支持输入自定义正整数。")
                    comps["max_resolution"] = num_dropdown("最大边长（0=不限）", "max_resolution", [0, 512, 720, 768, 1024, 1280, 1440, 1536, 1920, 2048, 2560, 2880, 3840], "限制输出任意边的最大像素；0 表示不限制。")
                    comps["batch_size"] = num_dropdown("Batch Size（4n+1）", "batch_size", [1, 5, 9, 13, 17, 21, 25, 29, 33, 41, 49, 65, 81], "必须满足 4n+1；手动输入非法值时会自动修正并提示。")
                    comps["uniform_batch_size"] = gr.Checkbox(label="统一末批大小", value=cfg.get("uniform_batch_size", False), info="将最后一个不足批大小的 Batch 填充到统一长度，以改善尾批时序一致性，但会增加计算量。")
                with gr.Row():
                    comps["seed"] = gr.Number(label="Seed", value=cfg.get("seed", 42), precision=0, info="随机种子；相同模型、输入与参数下有助于复现实验。")
                    comps["skip_first_frames"] = num_dropdown("跳过开头帧数", "skip_first_frames", [0, 1, 2, 5, 10, 15, 30, 60, 120], "视频起始处跳过的帧数；图片输入不适用。支持自定义非负整数。")
                    comps["load_cap"] = num_dropdown("最多处理帧数（0=全部）", "load_cap", [0, 25, 50, 100, 200, 300, 500, 1000], "限制视频读取的最大帧数；0 表示处理全部可用帧。")
                with gr.Row():
                    comps["chunk_size"] = num_dropdown("Chunk Size（0=单个大 Chunk）", "chunk_size", [0, 25, 50, 100, 200, 330, 660, 1000], "断点续处理以 Chunk 为最小提交单位。建议长视频设置 100–660；0 表示整个视频作为一个 Chunk，中断后只能重做这个视频。")
                    comps["prepend_frames"] = num_dropdown("Prepend Frames", "prepend_frames", [0, 1, 2, 4, 8, 16], "在开头加入反向帧缓解起始伪影；CLI 会在输出中移除这些前置帧。")
                    comps["temporal_overlap"] = num_dropdown("Temporal Overlap", "temporal_overlap", [0, 1, 2, 3, 4, 8, 12, 16], "相邻 Chunk 或多 GPU 分段的重叠帧数；与 Chunk Size 联动校验。")
                with gr.Row():
                    comps["color_correction"] = gr.Dropdown(label="色彩校正", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], value=cfg.get("color_correction", "lab"), info="lab 为感知色彩匹配；none 关闭色彩校正。")
                    comps["input_noise_scale"] = gr.Slider(label="Input Noise Scale", minimum=0, maximum=1, step=.01, value=cfg.get("input_noise_scale", 0.0), info="输入图像噪声注入强度，范围 0–1；较大值可能改变细节。")
                    comps["latent_noise_scale"] = gr.Slider(label="Latent Noise Scale", minimum=0, maximum=1, step=.01, value=cfg.get("latent_noise_scale", 0.0), info="潜空间噪声注入强度，范围 0–1；建议从 0 开始比较。")
                resolution_hint = gr.Markdown("分辨率设置正常。")
                parameter_validation_status = gr.Markdown("数值参数校验：手动输入后会自动转换类型；不合法的值不会保存或提交。")
            with gr.Tab("设备与显存优化"):
                gr.Markdown(f"<div class='seed-muted'>本机设备检测：{html.escape(device_info.get('note', ''))} · 可手动输入自定义设备值；CUDA 设备 ID 由 CLI 传递。</div>")
                with gr.Row():
                    comps["cuda_device"] = gr.Dropdown(label="CUDA Device（留空=CLI 默认）", choices=cuda_choices, value=cfg.get("cuda_device", ""), allow_custom_value=True, info="通过当前 Conda Python 查询 CUDA GPU。支持单卡 0 或多卡 0,1；macOS CLI 不提供此参数。", scale=2)
                    comps["dit_offload_device"] = gr.Dropdown(label="DiT Offload Device", choices=offload_choices, value=cfg.get("dit_offload_device", "none"), allow_custom_value=True, info="DiT 权重空闲时存放位置。BlockSwap / Swap I/O 要求非 none。", scale=2)
                    comps["vae_offload_device"] = gr.Dropdown(label="VAE Offload Device", choices=offload_choices, value=cfg.get("vae_offload_device", "none"), allow_custom_value=True, info="VAE 空闲时卸载位置；cpu 可节省 VRAM。", scale=2)
                    comps["tensor_offload_device"] = gr.Dropdown(label="Tensor Offload Device", choices=offload_choices, value=cfg.get("tensor_offload_device", "cpu"), allow_custom_value=True, info="中间 Tensor 暂存位置；cpu 通常更省显存。", scale=2)
                with gr.Row():
                    refresh_devices_btn = gr.Button("重新检测本机设备", variant="secondary")
                    comps["blocks_to_swap"] = num_dropdown("Transformer BlockSwap 数量", "blocks_to_swap", [0, 4, 8, 12, 16, 20, 24, 28, 32, 36], "交换到 Offload Device 的 Transformer Block 数量。3B 上限 32，7B 上限 36；需要 DiT Offload。")
                    comps["swap_io_components"] = gr.Checkbox(label="Swap I/O Components", value=cfg.get("swap_io_components", False), info="额外卸载 DiT 输入/输出组件以减少 VRAM；需要 DiT Offload Device。")
                gr.Markdown("#### VAE Tiling")
                with gr.Row():
                    comps["vae_encode_tiled"] = gr.Checkbox(label="Encode Tiling", value=cfg.get("vae_encode_tiled", False), info="对 VAE 编码进行切片，降低峰值显存。启用后下方 Tile 参数生效。")
                    comps["vae_encode_tile_size"] = num_dropdown("Encode Tile Size", "vae_encode_tile_size", [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048], "编码 Tile 边长像素；值越小通常越省显存。", interactive=bool(cfg.get("vae_encode_tiled", False)))
                    comps["vae_encode_tile_overlap"] = num_dropdown("Encode Tile Overlap", "vae_encode_tile_overlap", [0, 32, 64, 96, 128, 160, 192, 256, 320], "编码 Tile 重叠像素；必须小于 Tile Size，可降低切片接缝。", interactive=bool(cfg.get("vae_encode_tiled", False)))
                with gr.Row():
                    comps["vae_decode_tiled"] = gr.Checkbox(label="Decode Tiling", value=cfg.get("vae_decode_tiled", False), info="对 VAE 解码进行切片，降低峰值显存。")
                    comps["vae_decode_tile_size"] = num_dropdown("Decode Tile Size", "vae_decode_tile_size", [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048], "解码 Tile 边长像素；较小值更省显存但会增加开销。", interactive=bool(cfg.get("vae_decode_tiled", False)))
                    comps["vae_decode_tile_overlap"] = num_dropdown("Decode Tile Overlap", "vae_decode_tile_overlap", [0, 32, 64, 96, 128, 160, 192, 256, 320], "解码 Tile 重叠像素；必须小于 Tile Size。", interactive=bool(cfg.get("vae_decode_tiled", False)))
                comps["tile_debug"] = gr.Dropdown(label="Tile Debug", choices=["false", "encode", "decode"], value=cfg.get("tile_debug", "false"), info="可视化 VAE Tile；调试用，通常保持 false。")
            with gr.Tab("性能与缓存"):
                with gr.Row():
                    comps["keep_models_resident"] = gr.Checkbox(label="跨任务常驻 DiT / VAE（建议开启）", value=cfg.get("keep_models_resident", True), info="在当前 Gradio Python 进程中复用 DiT 与 VAE 权重。默认 Offload 到 CPU RAM，降低 8GB 显存压力；关闭后不主动保持模型缓存。")
                    clear_model_cache_btn = gr.Button("清空模型缓存并释放显存", variant="secondary")
                with gr.Row():
                    comps["attention_mode"] = gr.Dropdown(label="Attention Backend", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], value=cfg.get("attention_mode", "sdpa"), info="注意力实现后端。SDPA 是 PyTorch 内置默认选项；其他后端需安装匹配依赖并满足硬件条件。")
                    comps["compile_backend"] = gr.Dropdown(label="Compile Backend", choices=["inductor", "cudagraphs"], value=cfg.get("compile_backend", "inductor"), info="torch.compile 编译后端；inductor 通常是默认选择。", interactive=bool(cfg.get("compile_dit") or cfg.get("compile_vae")))
                    comps["compile_mode"] = gr.Dropdown(label="Compile Mode", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], value=cfg.get("compile_mode", "default"), info="编译优化策略；更激进的模式可能增加首次编译时间和内存使用。", interactive=bool(cfg.get("compile_dit") or cfg.get("compile_vae")))
                with gr.Row():
                    comps["compile_dit"] = gr.Checkbox(label="Compile DiT", value=cfg.get("compile_dit", False), info="启用 DiT torch.compile；首次运行通常需要编译，需对比稳定运行时间。")
                    comps["compile_vae"] = gr.Checkbox(label="Compile VAE", value=cfg.get("compile_vae", False), info="启用 VAE torch.compile；需权衡编译开销与后续推理速度。")
                    comps["compile_fullgraph"] = gr.Checkbox(label="Compile Fullgraph", value=cfg.get("compile_fullgraph", False), info="要求尽量编译为完整图；若存在图断点可能导致编译失败。", interactive=bool(cfg.get("compile_dit") or cfg.get("compile_vae")))
                    comps["compile_dynamic"] = gr.Checkbox(label="Compile Dynamic Shapes", value=cfg.get("compile_dynamic", False), info="允许动态输入形状，可能减少重复编译，但也可能影响优化效果。", interactive=bool(cfg.get("compile_dit") or cfg.get("compile_vae")))
                with gr.Row():
                    comps["compile_dynamo_cache_size_limit"] = num_dropdown("Dynamo Cache Size Limit", "compile_dynamo_cache_size_limit", [16, 32, 64, 128, 256, 512, 1024], "TorchDynamo 缓存容量限制；较高值可能增加 RAM 使用。", interactive=bool(cfg.get("compile_dit") or cfg.get("compile_vae")))
                    comps["compile_dynamo_recompile_limit"] = num_dropdown("Dynamo Recompile Limit", "compile_dynamo_recompile_limit", [16, 32, 64, 128, 256, 512], "允许重新编译的次数上限；输入形状种类较多时可适当提高。", interactive=bool(cfg.get("compile_dit") or cfg.get("compile_vae")))
                with gr.Row():
                    comps["cache_dit"] = gr.Checkbox(label="Cache DiT", value=cfg.get("cache_dit", False), info="多文件批处理时复用 DiT；需要 Offload Device。多 GPU 模式需配合 Chunk Size。")
                    comps["cache_vae"] = gr.Checkbox(label="Cache VAE", value=cfg.get("cache_vae", False), info="多文件批处理时复用 VAE；需要 Offload Device。多 GPU 模式需配合 Chunk Size。")
                    comps["debug"] = gr.Checkbox(label="Debug Logs", value=cfg.get("debug", False), info="开启详细诊断日志；会增加日志量。")

        param_components = [comps[k] for k in PARAM_KEYS]
        # Save every field after an actual user change. Explicit callback functions below
        # handle coupled values and write corrected values back to the config.
        for key, component in comps.items():
            if key in LINKED_AUTOSAVE_KEYS:
                continue  # Their coupled callbacks below validate and persist canonical values.
            if key in NUMERIC_PARAMETER_KEYS:
                component.change(
                    fn=lambda value, k=key: _on_numeric_parameter_change(k, value),
                    inputs=component,
                    outputs=[component, parameter_validation_status],
                    queue=False,
                )
            else:
                component.change(fn=lambda value, k=key: _settings_changed(k, value), inputs=component, outputs=None, queue=False)
        for component, key in [(cli_path, "cli_path"), (input_paths, "input_paths"), (output_root, "output_root")]:
            component.change(fn=lambda value, k=key: _settings_changed(k, value), inputs=component, outputs=None, queue=False)

        # Native browsing buttons
        browse_output_btn.click(lambda current: _browse_directory(current, "output"), inputs=output_root, outputs=[output_root, path_status], queue=False)
        browse_model_btn.click(lambda current: _browse_directory(current, "model"), inputs=comps["model_dir"], outputs=[comps["model_dir"], path_status], queue=False)
        browse_dit_model_btn.click(_browse_dit_model_file, inputs=comps["dit_model_path"], outputs=[comps["dit_model_path"], path_status], queue=False)
        browse_vae_model_btn.click(_browse_vae_model_file, inputs=comps["vae_model"], outputs=[comps["vae_model"], path_status], queue=False)
        browse_input_files_btn.click(_browse_input_files, inputs=[input_paths], outputs=[input_paths, path_status], queue=False)
        browse_input_folder_btn.click(_browse_input_folder, inputs=[input_paths], outputs=[input_paths, path_status], queue=False)

        # Reactive validation and parameter linkage
        comps["batch_size"].change(_normalise_batch_size, inputs=comps["batch_size"], outputs=[comps["batch_size"], resolution_hint], queue=False)
        comps["resolution"].change(_on_max_resolution_change, inputs=[comps["resolution"], comps["max_resolution"]], outputs=resolution_hint, queue=False)
        comps["max_resolution"].change(_on_max_resolution_change, inputs=[comps["resolution"], comps["max_resolution"]], outputs=resolution_hint, queue=False)
        comps["chunk_size"].change(_on_chunk_overlap_change, inputs=[comps["chunk_size"], comps["temporal_overlap"]], outputs=[comps["chunk_size"], comps["temporal_overlap"], resolution_hint], queue=False)
        comps["temporal_overlap"].change(_on_chunk_overlap_change, inputs=[comps["chunk_size"], comps["temporal_overlap"]], outputs=[comps["chunk_size"], comps["temporal_overlap"], resolution_hint], queue=False)
        comps["dit_model"].change(_on_model_change, inputs=[comps["dit_model"], comps["blocks_to_swap"]], outputs=[comps["blocks_to_swap"], resolution_hint], queue=False)
        comps["dit_model_path"].change(_on_dit_path_change, inputs=[comps["dit_model_path"], comps["blocks_to_swap"]], outputs=[comps["blocks_to_swap"], resolution_hint], queue=False)
        block_link_inputs = [comps["blocks_to_swap"], comps["dit_offload_device"], comps["swap_io_components"], comps["dit_model"]]
        block_link_outputs = [comps["blocks_to_swap"], comps["dit_offload_device"], resolution_hint]
        comps["blocks_to_swap"].change(_on_blocks_change, inputs=block_link_inputs, outputs=block_link_outputs, queue=False)
        comps["swap_io_components"].change(_on_blocks_change, inputs=block_link_inputs, outputs=block_link_outputs, queue=False)
        comps["dit_offload_device"].change(_on_blocks_change, inputs=block_link_inputs, outputs=block_link_outputs, queue=False)
        comps["dit_offload_device"].change(lambda device, enabled: _on_cache_toggle(enabled, device, "cache_dit"), inputs=[comps["dit_offload_device"], comps["cache_dit"]], outputs=[comps["dit_offload_device"], resolution_hint], queue=False)
        comps["vae_offload_device"].change(lambda device, enabled: _on_cache_toggle(enabled, device, "cache_vae"), inputs=[comps["vae_offload_device"], comps["cache_vae"]], outputs=[comps["vae_offload_device"], resolution_hint], queue=False)
        comps["vae_encode_tiled"].change(lambda en, size, ov: _on_tiling_toggle(en, size, ov, "vae_encode_tile_overlap"), inputs=[comps["vae_encode_tiled"], comps["vae_encode_tile_size"], comps["vae_encode_tile_overlap"]], outputs=[comps["vae_encode_tile_size"], comps["vae_encode_tile_overlap"], resolution_hint], queue=False)
        comps["vae_decode_tiled"].change(lambda en, size, ov: _on_tiling_toggle(en, size, ov, "vae_decode_tile_overlap"), inputs=[comps["vae_decode_tiled"], comps["vae_decode_tile_size"], comps["vae_decode_tile_overlap"]], outputs=[comps["vae_decode_tile_size"], comps["vae_decode_tile_overlap"], resolution_hint], queue=False)
        comps["vae_encode_tile_overlap"].change(lambda ov, sz: _on_tile_overlap_change(ov, sz, "vae_encode_tile_overlap"), inputs=[comps["vae_encode_tile_overlap"], comps["vae_encode_tile_size"]], outputs=[comps["vae_encode_tile_overlap"], resolution_hint], queue=False)
        comps["vae_decode_tile_overlap"].change(lambda ov, sz: _on_tile_overlap_change(ov, sz, "vae_decode_tile_overlap"), inputs=[comps["vae_decode_tile_overlap"], comps["vae_decode_tile_size"]], outputs=[comps["vae_decode_tile_overlap"], resolution_hint], queue=False)
        comps["vae_encode_tile_size"].change(lambda sz, ov: _on_tile_size_change(sz, ov, "vae_encode_tile_size", "vae_encode_tile_overlap"), inputs=[comps["vae_encode_tile_size"], comps["vae_encode_tile_overlap"]], outputs=[comps["vae_encode_tile_size"], comps["vae_encode_tile_overlap"], resolution_hint], queue=False)
        comps["vae_decode_tile_size"].change(lambda sz, ov: _on_tile_size_change(sz, ov, "vae_decode_tile_size", "vae_decode_tile_overlap"), inputs=[comps["vae_decode_tile_size"], comps["vae_decode_tile_overlap"]], outputs=[comps["vae_decode_tile_size"], comps["vae_decode_tile_overlap"], resolution_hint], queue=False)
        comps["video_backend"].change(_on_video_backend_change, inputs=[comps["video_backend"], comps["use_10bit"]], outputs=[comps["use_10bit"], resolution_hint], queue=False)
        comps["use_10bit"].change(_on_10bit_change, inputs=comps["use_10bit"], outputs=[comps["video_backend"], resolution_hint], queue=False)
        comps["cache_dit"].change(lambda enabled, device: _on_cache_toggle(enabled, device, "cache_dit"), inputs=[comps["cache_dit"], comps["dit_offload_device"]], outputs=[comps["dit_offload_device"], resolution_hint], queue=False)
        comps["cache_vae"].change(lambda enabled, device: _on_cache_toggle(enabled, device, "cache_vae"), inputs=[comps["cache_vae"], comps["vae_offload_device"]], outputs=[comps["vae_offload_device"], resolution_hint], queue=False)
        compile_common = [comps["compile_backend"], comps["compile_mode"], comps["compile_fullgraph"], comps["compile_dynamic"], comps["compile_dynamo_cache_size_limit"], comps["compile_dynamo_recompile_limit"]]
        comps["compile_dit"].change(_on_compile_toggle, inputs=[comps["compile_dit"], comps["compile_vae"]], outputs=compile_common, queue=False)
        comps["compile_vae"].change(_on_compile_toggle, inputs=[comps["compile_dit"], comps["compile_vae"]], outputs=compile_common, queue=False)

        def refresh_devices() -> Tuple[Any, Any, Any, Any, str]:
            info = detect_local_devices()
            global DEVICE_INFO_CACHE
            DEVICE_INFO_CACHE = info
            cuda_opts, offload_opts = build_device_choices(info)
            return (gr.update(choices=cuda_opts), gr.update(choices=offload_opts), gr.update(choices=offload_opts), gr.update(choices=offload_opts), info.get("note", "设备检测完成。"))
        refresh_devices_btn.click(refresh_devices, outputs=[comps["cuda_device"], comps["dit_offload_device"], comps["vae_offload_device"], comps["tensor_offload_device"], path_status], queue=False)
        clear_model_cache_btn.click(_clear_model_cache, outputs=path_status, queue=False)

        with gr.Row(elem_classes=["seed-actions"]):
            submit_btn = gr.Button("提交作业 / 开始队列", variant="primary", scale=3)
            pause_btn = gr.Button("暂停", variant="secondary")
            resume_btn = gr.Button("恢复", variant="secondary", interactive=False)
            safe_stop_btn = gr.Button("安全停止（完成当前 Chunk）", variant="stop", interactive=False)
            stop_btn = gr.Button("尽快终止（当前 Batch 边界）", variant="stop", interactive=False)
            preview_btn = gr.Button("查看执行方案", variant="secondary")
        submit_status = gr.Markdown("就绪。请设置输入路径与模型参数。")
        preview_output = gr.Markdown()
        gr.Markdown("<div class='seed-muted'>参数提示：将鼠标悬停在参数名称旁的说明图标上，可查看参数用途与限制。数值下拉框允许输入自定义值；程序仍会在提交前检查合法性。</div>")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 任务运行状态")
                state_line = gr.Markdown("QUEUE IDLE")
                progress_html = gr.HTML("<div class='progress-wrap'>等待提交任务。</div>")
                log_view = gr.Textbox(label="实时推理日志", lines=18, max_lines=28, interactive=False, autoscroll=True)
            with gr.Column(scale=2):
                gr.Markdown("### 作业队列 / 最近记录")
                jobs_table = gr.Dataframe(headers=["ID", "Status", "Input", "Progress", "Elapsed", "Output"], value=[], interactive=False, wrap=True)
                autosave_status = gr.Markdown("配置自动保存已启用。")
                startup_status = gr.Markdown("")

        with gr.Accordion("历史记录管理与快速复现", open=False):
            history_select = gr.Dropdown(label="选择历史记录", choices=_history_choices(), value=None, interactive=True)
            with gr.Row():
                restore_btn = gr.Button("将选中记录恢复到工作区", variant="primary")
                delete_confirm = gr.Checkbox(label="我确认删除所选记录", value=False)
                delete_btn = gr.Button("删除所选记录", variant="stop")
            with gr.Row():
                clear_confirm = gr.Checkbox(label="我确认清空全部历史记录（不可撤销）", value=False)
                clear_btn = gr.Button("清空全部历史", variant="stop")
            history_status = gr.Markdown("删除和清空操作均需要勾选确认。活动任务记录不能删除。")

        with gr.Row():
            save_btn = gr.Button("立即保存全部 UI 配置", variant="secondary")
            config_path_display = gr.Markdown(f"配置文件：`{CONFIG_PATH}` · 历史文件：`{HISTORY_PATH}`")

        submit_btn.click(_submit, inputs=[input_paths, cli_path, output_root] + param_components,
                         outputs=[submit_status, history_select, comps["batch_size"]], concurrency_limit=1)
        pause_btn.click(_pause, outputs=submit_status, queue=False)
        resume_btn.click(_resume, outputs=submit_status, queue=False)
        stop_btn.click(_stop, outputs=submit_status, queue=False)
        safe_stop_btn.click(_safe_stop, outputs=submit_status, queue=False)
        preview_btn.click(_preview, inputs=[input_paths, cli_path, output_root] + param_components, outputs=preview_output, queue=False)
        reload_models_btn.click(_reload_model_choices, inputs=[cli_path, comps["dit_model"], comps["blocks_to_swap"]], outputs=[comps["dit_model"], comps["blocks_to_swap"], submit_status], queue=False)
        restore_btn.click(_restore_history, inputs=history_select,
                          outputs=[cli_path, input_paths, output_root] + param_components + [history_status], queue=False)
        delete_btn.click(_delete_history, inputs=[history_select, delete_confirm], outputs=[history_status, history_select, delete_confirm], queue=False)
        clear_btn.click(_clear_history, inputs=clear_confirm, outputs=[history_status, history_select, clear_confirm], queue=False)

        def save_all(input_value: str, cli_value: str, output_value: str, *vals: Any) -> Tuple[str, Any]:
            raw = config_from_ui(list(vals))
            batch_update, batch_note = _normalise_batch_size(raw.get("batch_size", DEFAULTS["batch_size"]), persist=False)
            raw["batch_size"] = _strict_int(batch_update.get("value", DEFAULTS["batch_size"]), "batch_size")
            normalized, errors = normalize_ui_parameters(raw)
            if errors:
                return "❌ 配置未保存，存在无效参数：\n" + "\n".join(f"- {x}" for x in errors), gr.update()
            data = {"cli_path": cli_value, "input_paths": input_value, "output_root": output_value, **normalized}
            return save_config_values(data) + "\n" + batch_note, batch_update
        save_btn.click(save_all, inputs=[input_paths, cli_path, output_root] + param_components, outputs=[autosave_status, comps["batch_size"]], queue=False)

        load_outputs = [cli_path, input_paths, output_root] + param_components + [startup_status]
        demo.load(lambda: _load_config_outputs(load_outputs), outputs=load_outputs, queue=False)
        timer = gr.Timer(1.0)
        timer.tick(_refresh, outputs=[state_line, progress_html, log_view, jobs_table, stop_btn, safe_stop_btn, pause_btn, resume_btn], queue=False)
        gr.Markdown("<div class='seed-muted'>安全说明：推理在同一个 Python 进程内执行。安全停止会完成当前 Chunk、关闭临时视频并提交 checkpoint；尽快终止会在下一个模型 Batch 回调边界中断，当前 Chunk 不提交。暂停同样在 Batch 回调边界生效。相同输入和参数再次提交时从最后一个有效 Chunk 恢复。单 GPU 路径支持模型跨任务常驻缓存。多 GPU 保留原生 multiprocessing 回退，但不具备同等级别的常驻缓存和 Chunk 断点恢复，且 worker 推理期间不能响应暂停/停止请求；长视频建议使用单 GPU。</div>")

    demo._seedvr2_theme = theme
    demo._seedvr2_css = css
    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=8, max_size=64)
    launch_kwargs = {
        "server_name": os.environ.get("SEEDVR2_GRADIO_HOST", "127.0.0.1"),
        "server_port": int(os.environ.get("SEEDVR2_GRADIO_PORT", "7860")),
        "inbrowser": True,
        "show_error": True,
    }
    if int(str(getattr(gr, "__version__", "4")).split(".")[0]) >= 6:
        launch_kwargs["theme"] = getattr(app, "_seedvr2_theme", None)
        launch_kwargs["css"] = getattr(app, "_seedvr2_css", "")
    app.launch(**launch_kwargs)
