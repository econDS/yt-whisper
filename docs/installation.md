# Installation

Choose one setup method. Run repository installation commands from the cloned
repository. Use Python 3.12 for the configurations documented here.

## Prerequisites

Install FFmpeg, including ffprobe, and make both executables available on PATH.
Install Node.js 22+ only if you will transcribe YouTube URLs.
Check with `ffmpeg -version`, `ffprobe -version` and, for URLs, `node --version`.

The application supports CPU and CUDA devices. The Windows helper installs an
NVIDIA CUDA build. For other setups, choose PyTorch for your platform using its
[official installation selector](https://pytorch.org/get-started/locally/).
The app does not currently expose Apple's MPS device; use CPU on macOS.

## Windows: NVIDIA and Conda

Requires Conda and an NVIDIA driver compatible with the CUDA 12.6 PyTorch wheel.

```powershell
$env:YTW_DATA_DIR = 'E:\yt-whisper'  # Choose a folder on an existing drive.
.\scripts\setup-windows.ps1 -DataRoot $env:YTW_DATA_DIR
```

If Conda is not at `$env:USERPROFILE\miniconda3\Scripts\conda.exe`, supply its
actual path. For example, adjust this path to your installation:

```powershell
$condaExe = Join-Path $env:USERPROFILE 'miniforge3\Scripts\conda.exe'
.\scripts\setup-windows.ps1 -DataRoot $env:YTW_DATA_DIR -CondaExecutable $condaExe
```

The helper creates `<data-root>/envs/py312` if it does not exist. It sets TEMP/TMP,
pip cache and Conda package cache under the data folder before installation,
installs PyTorch 2.13.0 with CUDA 12.6, then installs `.[ui,thai,dev]` with
`requirements-windows-py312.lock.txt` as constraints. Finally it runs pip check
and the app's doctor command.

This configuration has been exercised on Windows/NVIDIA. The lock file describes
that environment; do not apply it to CPU-only, macOS or other CUDA installations.

## Manual installation

Conda is optional for these routes. No API key is needed.

### Windows CPU

Install Python 3.12 first. In PowerShell:

```powershell
$env:YTW_DATA_DIR = 'E:\yt-whisper'  # Change to a folder on an existing drive.
$env:TEMP = Join-Path $env:YTW_DATA_DIR 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $env:YTW_DATA_DIR 'cache\pip'
New-Item -ItemType Directory -Force $env:TEMP, $env:PIP_CACHE_DIR | Out-Null

py -3.12 -m venv "$env:YTW_DATA_DIR\envs\py312"
$py = Join-Path $env:YTW_DATA_DIR 'envs\py312\python.exe'
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -e ".[ui]"
& $py -m yt_whisper.cli --doctor
& $py -m yt_whisper.ui
```

If Python is available as `python` rather than `py`, verify
`python --version` reports 3.12 and use `python -m venv` for that line.
The install above includes the UI; add the Thai extra if needed.

### Linux or macOS

The following creates an isolated environment in the repository. Set the data path
before installing so pip/temp data also has an explicit location.

```bash
export YTW_DATA_DIR="$HOME/.local/share/yt-whisper"
export TMPDIR="$YTW_DATA_DIR/tmp"
export PIP_CACHE_DIR="$YTW_DATA_DIR/cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

python3.12 -m venv .venv
source .venv/bin/activate
```

On Linux with CPU-only execution:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On macOS:

```bash
python -m pip install torch
```

For a Linux CUDA installation, run the suitable PyTorch command from the
[official selector](https://pytorch.org/get-started/locally/) instead.
Then install the application into the same environment:

```bash
python -m pip install -e ".[ui]"
python -m yt_whisper.cli --doctor
python -m yt_whisper.ui
```

### Choose optional dependencies

With the chosen environment's Python, from the repository:

| Features needed | Install |
| --- | --- |
| CLI and JSON tool | `python -m pip install -e .` |
| Browser UI | `python -m pip install -e ".[ui]"` |
| UI and all three Thonburian Thai models | `python -m pip install -e ".[ui,thai]"` |
| Tests, builds and benchmarking | `python -m pip install -e ".[ui,thai,dev]"` |

On Windows with an explicit `$py` path, replace `python` with `& $py`.
Thai dependencies are loaded only when Thonburian is selected.
For CLI/tool Thai support alone, use `python -m pip install -e ".[thai]"`.
This extra supports Medium, Large-v3 and Distilled Large-v3 with the existing
Transformers/safetensors constraints; it does not install model weights.
See [model choices and per-model offline directories](thonburian.md).

## Launch again

A new terminal may not have the installed environment on PATH.

### Windows with the documented data-folder environment

Set the same data folder you chose during installation:

```powershell
$env:YTW_DATA_DIR = 'E:\yt-whisper'  # Use your actual installation folder.
$py = Join-Path $env:YTW_DATA_DIR 'envs\py312\python.exe'
& $py -m yt_whisper.cli --doctor
& $py -m yt_whisper.ui
```

Alternatively, from the repository in the same terminal:

```powershell
.\ytwhisper_ui.bat
```

Double-clicking the batch file without a configured `YTW_DATA_DIR` uses the default
E:/yt-whisper folder. Setting the variable in PowerShell applies to that session
and child processes, not to an independently launched Explorer window.

### Linux/macOS environment from this guide

```bash
source .venv/bin/activate
# Set YTW_DATA_DIR again here if you used a non-default data folder.
yt_whisper_ui
```

### Other environments

Activate your environment or use its Python executable directly:

```shell
python -m yt_whisper.cli --doctor
python -m yt_whisper.ui
python -m yt_whisper.tool --request request.json --response response.json
```

## Data and preferences

Runtime models, uploads and caches follow `YTW_DATA_DIR`. Model download and
temporary paths are set before ML/UI libraries are imported. No global Windows
settings are changed.

Preferences are stored in `<data-root>/config/user_config.json`. Missing preferences
use defaults. A legacy repository `user_config.json` can still be read for
compatibility, but personal settings are ignored by Git. A generic example is
[examples/user-config.json](../examples/user-config.json).

Changing `YTW_DATA_DIR` changes where the app looks for cached models and
preferences. It does not move an existing environment or model cache automatically.
