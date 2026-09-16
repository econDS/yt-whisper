# yt-whisper

Local transcription of audio/video files and individual YouTube videos, using OpenAI Whisper or Thonburian Whisper. Windows + NVIDIA setup is included.

## Windows: start the updated app

Double-click **ytwhisper_ui.bat** in this repository. The UI listens on `http://127.0.0.1:7860` (or the next free port) and is not shared publicly.

The launcher uses `<data-root>/envs/py312/python.exe`; the default Windows data root is `E:\yt-whisper`.

To install the tested environment again, run PowerShell from the repository:

```powershell
.\scripts\setup-windows.ps1
```

Prerequisites: Conda, an NVIDIA driver compatible with the CUDA 12.6 PyTorch wheel, FFmpeg in PATH, and Node.js >=22 in PATH for YouTube. The setup script accepts `-DataRoot` and `-CondaExecutable` to override its defaults. It sets installation temp, pip cache and Conda package cache before installing anything.

The project requires Python >=3.11; the Windows setup script creates a Python 3.12 environment. Package dependencies are declared in `pyproject.toml`; `requirements-windows-py312.lock.txt` records the tested Windows versions.

## Storage

On Windows, the default data root is **E:\yt-whisper**. Set `YTW_DATA_DIR` before launching to choose another location. On other operating systems the default is `~/.local/share/yt-whisper`.

| Data | Location under the data root |
| --- | --- |
| New environment | `envs/py312` |
| OpenAI model weights | `models/whisper` |
| Local Thonburian model | `models/thonburian` |
| Hugging Face / pip / Conda / other caches | `cache/` |
| Downloads and Gradio uploads | `tmp/` |
| Transcripts | `outputs/<title>-<unique-id>/` |
| User preferences | `config/user_config.json` |

The application configures caches before importing ML/UI libraries. This is process-local; it does not change global Windows settings. Whisper always receives an explicit model directory. Each URL job owns a temporary directory that is removed on success or failure. Gradio upload cache is swept hourly for files older than 24 hours while the app is running.

Local input files are decoded directly; no MP3 is written next to the source. Output directories are unique, so duplicate video titles and repeated runs do not overwrite earlier transcripts.


## Model selection

- `base`: a small model for smoke tests.
- `small`, `medium`, `large-v3`: multilingual models, with increasing memory requirements.
- `turbo` / `large-v3-turbo`: available through the updated Whisper package; downloaded into E: only when selected if not already cached.
- `Thai_Thonburian`: uses Transformers and the local `models/thonburian` directory when its config exists. Set `YTW_THAI_MODEL` to another local model directory if needed. Without a local model, the app uses `biodatlab/whisper-th-medium-combined` and downloads to the configured Hugging Face cache.

The local Thonburian folder must include `model.safetensors`, config, generation config, tokenizer and feature extractor files. The app does not need a duplicate `pytorch_model.bin`. If a Thai segment lacks its ending timestamp, the clip duration is used so subtitle text is retained; review timing for publication.

Only one model is retained by each running app process. Switching models releases its previous model; requests within the UI are serialized. Separate CLI/UI processes still consume memory independently.

**Translate means translate speech into English.** Turbo is rejected for this task because it is not trained for translation. `.en` models force English. Auto language handles missing/empty values. CUDA falls back to CPU if unavailable.

## CLI

From PowerShell, without activating Conda:

```powershell
$py = 'E:\yt-whisper\envs\py312\python.exe'
& $py -m yt_whisper.cli --doctor
& $py -m yt_whisper.cli 'D:\audio\example.mp4' --model base --language th --format all
& $py -m yt_whisper.cli 'https://www.youtube.com/watch?v=VIDEO_ID' --model small --language Auto
& $py -m yt_whisper.cli 'D:\audio\example.wav' --model medium --task translate
```

Supported formats: TXT, JSON, SRT, VTT, TSV and JSONL; `--format all` writes all six. VTT remains the CLI default. JSON preserves segment details and records model, task, device and source. The UI returns text and download links for the selected formats.

Basic CLI options: `--device auto|cuda|cpu`, `--output_dir`, `--break-lines`, `--initial-prompt`, `--word-timestamps`, `--verbose True|False`. Prompt and word-timestamp options are for OpenAI Whisper, not the Thonburian backend.

Playlists are rejected before media downloading; a video URL with a playlist parameter processes only the selected video. YouTube availability is also affected by the site's restrictions and network conditions. The app does not automatically read browser cookies or bypass account checks.

## Other installations

Install a suitable PyTorch wheel first using the [official installer](https://pytorch.org/get-started/locally/), then:

```shell
python -m pip install -e ".[ui,thai]"
yt_whisper --doctor
yt_whisper_ui
```

The base install `pip install -e .` provides the CLI. `ui`, `thai` and `dev` are optional extras. Transformers is imported only when Thonburian is selected.

## Validation

```powershell
$env:TEMP = 'E:\yt-whisper\tmp'
$env:TMP = $env:TEMP
$env:YTW_DATA_DIR = 'E:\yt-whisper'
& 'E:\yt-whisper\envs\py312\python.exe' -m pip check
& 'E:\yt-whisper\envs\py312\python.exe' -m pytest -q
```

Regression tests cover input preservation, download cleanup, playlist rejection, duplicate output titles, settings, Auto language, CPU fallback, model reuse, Turbo translation and UI creation without importing the Thai backend. They do not download model weights.

## License

MIT. Originally forked from [m1guelpf/yt-whisper](https://github.com/m1guelpf/yt-whisper).

## Advanced transcription and evaluation

CLI and UI now expose prompt context, repeating the prompt, word timestamps,
previous-text conditioning, suspected-hallucination silence filtering, clip ranges,
beam size and temperature/fallback sequences. The UI hides these controls for
Thonburian, which uses separate backend settings. Explicit unsupported CLI options
with Thonburian are rejected.

- Leave temperature empty to retain Whisper's default fallback sequence.
- Silence filtering requires word timestamps and a positive threshold in seconds.
- Repeating the initial prompt requires a non-empty prompt.
- Clip ranges use seconds: `10,30,45,60`, or `10` to process from 10 seconds to the end.
- Beam search is used at temperature 0; a sequence containing only nonzero temperatures is rejected if a beam size is given.
- JSON output records effective decoding options for reproducibility.

Example:

```powershell
& $py -m yt_whisper.cli 'D:\audio\example.wav' --model turbo --language th --initial-prompt 'ชื่อคน ศัพท์เฉพาะ' --carry-initial-prompt --word-timestamps --no-condition-on-previous-text --hallucination-silence-threshold 2 --clip-timestamps '10,40' --beam-size 5 --temperature '0,0.2,0.4' --format all
```

`--format all` includes TXT, JSON, SRT, VTT, TSV and JSONL. The UI lets you select formats.
TSV timestamps are integer milliseconds. JSONL writes one complete segment per line,
including word timestamps when enabled, using this app's writer; it does not require
Whisper main.

To reproduce a benchmark (dev extra required):

```powershell
& $py scripts/benchmark.py --audio 'E:\audio\example.wav' --model turbo --output 'E:\yt-whisper\benchmarks\rerun.json'
```

Pass `--reference PATH` for character error rate (CER). Only whitespace is removed
from both reference and hypothesis; punctuation/spelling otherwise remain unchanged.
Lower CER is better. Without a verified reference, the script reports performance
only. It records model-load time separately from inference, sampled peak process
RAM and peak CUDA allocation. Single runs on a desktop are not controlled hardware
benchmarks.

`.github/workflows/tests.yml` runs tests and wheel building on Windows and Linux
with Python 3.12 and CPU PyTorch. The workflow runs after the code is pushed; adding
the file locally does not mean a GitHub run has passed.

### Reproducible comparisons

For reproducible comparisons, specify `--beam-size 5 --temperature 0`. The
benchmark uses seed 0 by default (`--seed` overrides it), records audio/reference
hashes, and can load a separate pinned Whisper checkout with `--whisper-source`.
CUDA results can still vary across hardware/software versions. Evaluate several
representative clips with verified references before choosing model settings.

## Use from another workflow

The versioned [JSON tool interface](docs/tool-interface.md) accepts local audio and
multiple named time ranges in one request, reuses one model, and returns original-file
timestamps with machine-readable status/errors. It runs as a subprocess without
starting a server. Example request: [examples/tool-request.json](examples/tool-request.json).

```powershell
& 'E:\yt-whisper\envs\py312\python.exe' -m yt_whisper.tool --request 'E:\jobs\request.json' --response 'E:\jobs\response.json'
```

stdout is UTF-8 JSON; diagnostics go to stderr. Keep completed results even when a
later range fails. The caller owns transcript comparison and editorial decisions.
For Japanese performance tests, the benchmark also accepts `--language ja`.
