# SeedVR2 Gradio Workbench

A local Gradio workbench for SeedVR2. The inference engine and task manager run inside the same Python process, enabling resident DiT/VAE caches, resumable chunk processing, cooperative pause/stop controls, and an editable parameter interface.

> **Important:** This repository provides the SeedVR2 GUI + runtime integration layer, along with a modified `inference_cli.py`. It does not re-package the upstream SeedVR2 `src/` source code, model weights, or large dependencies. Before use, you need to prepare a compatible SeedVR2 project directory.

---

## 1. Main features / 主要功能

### In-process inference and resident models

- Gradio task manager imports `inference_cli.py` and calls its inference functions directly. It does not create a new CLI subprocess for every file.
- A serial worker executes queued tasks to avoid multiple independent model instances competing for VRAM.
- DiT and VAE can be retained through the upstream global model cache across tasks in the same application process.
- Model identity is tracked using resolved file path, file size and modification time. Changing the actual DiT/VAE weights evicts the old cache entries; changing execution settings rebuilds transient context while retaining reusable weights when possible.
- The default **Keep Models Resident** policy enables both caches. When DiT/VAE offload is set to `none`, the runtime uses CPU RAM offload to reduce VRAM pressure. This is a policy choice, not a guarantee that weights remain physically in VRAM.
- Use **Clear Model Cache and Release VRAM** when you need to explicitly unload cached weights. Closing the Gradio Python process also releases its model objects.

### Chunk-level checkpoint and resume

For single-GPU video processing:

1. The input is read as a sequence of new-frame ranges. `chunk_size=0` means one Chunk for the entire video; a positive value enables multiple Chunks.
2. Each completed Chunk is written to a separate temporary file (`.partial.mp4`) or temporary PNG directory.
3. The writer is closed, the file is checked, and the temporary artifact is atomically renamed to its committed Chunk name.
4. Only after the Chunk artifact is committed does the runtime atomically update `checkpoint/checkpoint.json`.
5. After all Chunks are committed, MP4 Chunks are concatenated into the final video. FFmpeg stream-copy is preferred; OpenCV re-encoding is the fallback for the OpenCV backend. The merged frame count is checked before the final file is promoted.
6. A subsequent submission with the same source, model weights and parameters reuses valid committed Chunks and processes only the remaining ones.

The checkpoint fingerprint includes source path/stat information, parameters, model identities and engine file identity. If the source file, model weights or processing parameters change, the runtime creates a different run directory rather than mixing incompatible Chunks.

The resume path decodes the previous source-frame overlap from the original input before processing the next Chunk. This restores temporal context instead of resuming at a hard boundary with no context.

### Task controls

- **Pause**: pauses cooperatively at the next model Batch callback. The process and model memory remain allocated.
- **Resume**: continues a paused task.
- **Safe Stop**: finishes the current Chunk, closes the writer, commits its checkpoint, then stops before starting the next Chunk. Waiting jobs are cancelled.
- **Immediate Stop**: requests cancellation at the next available model Batch callback. The current incomplete Chunk is not checkpointed; previously committed Chunks remain available for resume. Python/CUDA work is not force-killed from another thread because doing so is unsafe for an in-process runtime.
- **Submit again**: submit the same input and settings to resume from the last valid checkpoint.

### Parameters and model selection

- DiT model: choose a registered model/quantization or specify an explicit local weight file.
- VAE model: type a registry filename or explicit local path.
- Both DiT and VAE support a native Windows file picker and direct text entry.
- The GUI retains the previous parameter sections for input/output, resolution, batch, noise, temporal overlap, devices, offload, BlockSwap, VAE tiling, attention, `torch.compile`, caching and debug logging.
- Parameter fields include contextual help, and submission validation checks common cross-field constraints (4n+1 Batch Size, temporal overlap vs Chunk Size, VAE tile overlap vs tile size, 10-bit vs FFmpeg, BlockSwap vs offload, model-specific BlockSwap limits, device IDs and paths).

## 2. Project structure

```text
SeedVR2-Gradio-Workbench/
├── inference_cli.py              # Modified upstream CLI / in-process inference API
├── seedvr2_runtime.py            # Task queue, resident cache, Chunk/checkpoint engine
├── seedvr2_gradio.py             # Gradio user interface
├── tests/
│   ├── conftest.py
│   ├── test_chunk_resume.py
│   └── test_progress_api_guard.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── NOTICE
└── .gitignore
```

This is an integration overlay rather than a self-contained copy of the complete SeedVR2 upstream source tree. Do not add model weights, local input videos, generated outputs or your local config/history files to Git.

## 3. Requirements

- Windows 10/11 or Linux (native file dialogs are intended for local desktop sessions).
- Python 3.10+.
- A working SeedVR2 environment with PyTorch, the matching CUDA build (for NVIDIA GPUs), OpenCV, NumPy and all upstream SeedVR2 dependencies.
- The SeedVR2 source tree must be compatible with the paired `inference_cli.py`. The Workbench checks for the required `progress_callback` hooks before loading models. If the installed `src/core/generation_phases.py` does not expose these callbacks, the task will fail early with an explicit compatibility error.
- FFmpeg is recommended for reliable chunk concatenation and required for the FFmpeg / 10-bit output backend. `ffprobe` is recommended for more reliable video frame-count validation.

Do **not** reinstall PyTorch or CUDA just to install this GUI. Use the Conda environment that already runs your SeedVR2 CLI.

## 4. Installation (Windows / Conda)

### 4.1 Prepare the upstream SeedVR2 tree

Clone or use an existing compatible SeedVR2 project. The integration target is the project root containing `src/`, `models/` and the upstream Python modules.

Upstream project: <https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler>

### 4.2 Copy the Workbench files

Copy these files to the SeedVR2 project root, alongside the `src/` directory:

- `inference_cli.py` (back up the original first; this repository includes the modified paired version)
- `seedvr2_runtime.py`
- `seedvr2_gradio.py`

Keep the `tests/`, `README.md`, `requirements.txt`, `LICENSE`, `NOTICE` and `.gitignore` in your Workbench Git repository. The GUI/runtime files can also live in the upstream project root for local execution.

### 4.3 Install GUI dependency

```powershell
conda activate seedvr2
python -m pip install -r requirements.txt
```

The requirements file intentionally installs only Gradio. PyTorch, CUDA, OpenCV, NumPy and other SeedVR2 dependencies should remain at the versions already configured for your inference environment.

### 4.4 Start

Run from the SeedVR2 project root:

```powershell
python seedvr2_gradio.py
```

Then open:

```text
http://127.0.0.1:7860
```

Optional host/port configuration:

```powershell
$env:SEEDVR2_GRADIO_HOST="127.0.0.1"
$env:SEEDVR2_GRADIO_PORT="7860"
python seedvr2_gradio.py
```

Only bind to `0.0.0.0` when LAN access is intentional and the host is appropriately protected. The GUI can launch local model inference and access files available to the Python process.

## 5. Recommended settings for 8 GB laptop GPUs

For an 8 GB RTX 4070 Laptop GPU, start conservatively:

- Use a 3B FP8 or other memory-appropriate DiT weight.
- Enable **Keep Models Resident**.
- Leave DiT/VAE offload at `none` initially; the resident policy will use CPU RAM offload for these model families. You may also choose `cpu` explicitly.
- Keep `tensor_offload_device=cpu`.
- Start with `batch_size=5` or `9` and a moderate output resolution.
- Enable VAE tiling if the VAE encode/decode phase has a VRAM peak.
- For long videos, set `chunk_size` to a positive value such as `100`, `200` or `330`; keep `temporal_overlap` smaller than the Chunk Size.
- Avoid enabling `torch.compile` until a stable non-compiled run works; first-time compilation can consume additional time and memory.

These are starting points rather than guaranteed settings. Actual VRAM use depends on input resolution, model variant, quantization, attention backend, offload and tile settings.

## 6. Resume workflow

1. Submit a video with `chunk_size > 0`.
2. Use **Safe Stop** to stop at a committed Chunk boundary, or **Immediate Stop** to abandon the currently running Chunk at its next Batch callback.
3. Check the task output directory if you need to inspect committed temporary files. Do not treat uncommitted `.partial` files as valid results.
4. Submit the same source again with the same model paths, output root and parameters. The runtime finds the matching run fingerprint and validates the committed Chunk files before resuming.
5. After the final Chunk, the application creates the final MP4 and marks the manifest complete.

A cancelled task is not a failed checkpoint. The history record stores task inputs and parameters; use **Restore to Workspace** and submit again. The output/checkpoint directory is not removed when history entries are deleted.

### What is and is not resumable

- Single-GPU video processing: Chunk-level checkpoint and resume.
- PNG sequence output: Chunk directories are committed and collected into a final numbered sequence directory.
- Single-image processing: runs through the upstream image pipeline in-process, but does not need video Chunk checkpoints.
- Multi-GPU processing: compatibility fallback to the upstream native multiprocessing path; it does not currently provide the same resident cache or Chunk-resume guarantees. Pause/stop controls cannot interrupt an active multi-GPU worker run; use single GPU for long resumable videos.
- An interrupted Chunk is recomputed from its beginning. Only successfully committed earlier Chunks are reused.
- The current pipeline writes the enhanced video frames; it does not preserve or remux the source video's audio track.

## 7. Output layout

```text
outputs/
└── source_name_<source-id>/
    └── runs/
        └── <run-fingerprint>/
            └── source_name_<source-id>/
                ├── checkpoint/
                │   ├── checkpoint.json
                │   └── chunks/
                │       ├── chunk_000001.mp4
                │       ├── chunk_000002.mp4
                │       └── ...
                └── source_name_SeedVR2.mp4
```

The exact run path is shown in the task history. Checkpoint files and output artifacts are not automatically deleted by the GUI.

## 8. Development and tests

Run CPU-only regression tests from the Workbench directory:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The checkpoint test creates a synthetic 25-frame video, requests a safe stop after the first 10-frame Chunk, then verifies that the remaining Chunks resume and merge into a 25-frame output. These tests validate the checkpoint/merge orchestration; they do not replace a real GPU inference test with the user's installed SeedVR2 models.

## 9. Known limitations

- The in-process engine binds to the first imported `inference_cli.py` for the lifetime of the GUI. Restart the GUI to switch to a different SeedVR2 source directory.
- Cooperative pause/stop takes effect at the next upstream phase Batch callback. A single currently executing GPU kernel or batch is not pre-empted mid-operation.
- A sudden power loss during a Chunk can leave a `.partial` artifact. It is not added to the checkpoint; the previous committed Chunk remains reusable.
- Model weights stay resident only while the Gradio Python process is alive and the selected cache policy keeps them. Model switching intentionally evicts the previous DiT/VAE cache entries.
- Multi-GPU behavior retains the upstream CLI fallback and is not equivalent to the single-GPU checkpoint path.
- Native Explorer dialogs require the Gradio Python process to run on the local desktop. When running remotely, type server-side absolute paths or use Gradio upload. For durable history/resume, prefer stable absolute input paths over temporary Gradio upload-cache paths.

## 10. License and attribution

The project is distributed under Apache License 2.0. The included `inference_cli.py` is derived from the upstream `numz/ComfyUI-SeedVR2_VideoUpscaler` implementation and contains Workbench modifications; see `NOTICE` and `LICENSE`. SeedVR2 model weights and third-party packages remain subject to their own licenses and terms.
