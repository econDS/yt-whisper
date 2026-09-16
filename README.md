# yt-whisper

Transcribe local audio/video files or a single YouTube video with Whisper.
Use a browser UI, a command-line interface, or a JSON subprocess tool from another
application. Transcription runs locally; model weights download on first use.

Outputs: **TXT, JSON, SRT, VTT, TSV and JSONL**. Supports multilingual Whisper
models and optional Thonburian Whisper for Thai.

## Choose how to use it

| Goal | Start here |
| --- | --- |
| Upload a file or paste a URL in a browser | [Browser UI](#browser-ui) |
| Transcribe files from a terminal | [Command line](#command-line) |
| Call transcription from your own workflow | [JSON tool](#json-tool) |

**Install first using one of the paths below.** A GPU and an API key are not
required. GPU acceleration is optional; the bundled Windows setup script
specifically targets NVIDIA CUDA.

## Requirements

- **Python 3.12 recommended**; the package requires Python 3.11 or newer.
- **FFmpeg and ffprobe** on PATH for audio/video processing.
- **Node.js 22+ for YouTube URLs only.** Local files do not need Node.
  This app enables Node in yt-dlp's Python API; see [yt-dlp's runtime documentation](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
- Disk space for the Python environment, selected model weights and working files.
  Large models can require several GB for weights alone.

## Install

Get the source first:

```shell
git clone https://github.com/econDS/yt-whisper.git
cd yt-whisper
```

### Windows with an NVIDIA GPU and Conda

The helper installs Python 3.12, CUDA 12.6 PyTorch and the app's UI/Thai/development
dependencies using the tested Windows constraints.

Run in PowerShell from the cloned repository. **Choose a data folder on a drive
that exists and has enough space; E: is only the default.**

```powershell
$env:YTW_DATA_DIR = 'E:\yt-whisper'  # Change this if needed.
.\scripts\setup-windows.ps1 -DataRoot $env:YTW_DATA_DIR
```

The script expects Conda at `~/miniconda3/Scripts/conda.exe`. If yours is elsewhere,
pass `-CondaExecutable` with its actual path; see the
[installation guide](docs/installation.md#windows-nvidia-and-conda).
It adds the environment to PATH for the current PowerShell session.

Check the installation, then open the UI:

```powershell
yt_whisper --doctor
yt_whisper_ui
```

### CPU, Linux, macOS, or your own environment

Use the [manual installation guide](docs/installation.md#manual-installation).
It includes Windows CPU and Linux/macOS commands, storage configuration, and
optional dependencies. The Windows CUDA lock file is specific to that environment.

## Try it

### Browser UI

Run `yt_whisper_ui` in the installed environment. Open the localhost address
printed in the terminal, normally **http://127.0.0.1:7860**.

1. Choose **File** and upload a short audio/video file, or choose **URL** and paste a video URL.
2. Start with `base` to check the installation; choose the spoken language or Auto.
3. Click **Transcribe**. Read the text and download the selected output formats.

For later Windows sessions, `ytwhisper_ui.bat` opens
`<data-root>/envs/py312/python.exe`. If you chose a custom data folder, set
`YTW_DATA_DIR` to the same folder before launching it. The
[installation guide](docs/installation.md#launch-again) has exact commands.
The UI listens locally and does not create a public sharing link.

### Command line

These commands use the installed environment. Replace `audio.wav` with your file:

```shell
yt_whisper "audio.wav" --model base --language Auto --format all
yt_whisper "audio.wav" --model turbo --language ja --format all
yt_whisper "https://www.youtube.com/watch?v=VIDEO_ID" --model turbo --language Auto
```

The first command creates all six output formats and prints their paths.
The CLI otherwise defaults to VTT. Use `--device cpu` to choose CPU execution,
or `--output_dir` to choose where transcripts are saved.

See the [usage guide](docs/usage.md) for model choices, prompts, timestamps,
clip ranges, temperature settings and benchmarks.

### JSON tool

Call the tool from a script without starting the UI or an HTTP server:

```shell
yt_whisper_tool --request request.json --response response.json
```

Start with [examples/tool-request.json](examples/tool-request.json), then change
`audio_path` and the time ranges to match your own file. Relative audio paths
resolve against the request file's directory.

The versioned response contains text, segments, word timestamps when enabled,
model/settings information and status/errors. Multiple named ranges share one model
within a request. stdout is UTF-8 JSON; progress goes to stderr.

Read the [JSON interface contract and Python caller example](docs/tool-interface.md)
before integrating. In particular, retain completed results if a later range fails.

## Where files go

Set `YTW_DATA_DIR` before running the app to choose the data folder.

| Data | Location under that folder |
| --- | --- |
| OpenAI Whisper weights | `models/whisper/` |
| Local Thonburian weights, if supplied | `models/thonburian/` |
| Model/package caches | `cache/` |
| Downloaded audio and upload/range temporary files | `tmp/` |
| Transcripts | `outputs/<title>-<unique-id>/` |
| UI preferences | `config/user_config.json` |

Windows defaults to `E:\yt-whisper`; Linux/macOS default to
`~/.local/share/yt-whisper`. **E: is not required.** Configure another location
before starting if that drive is unavailable. The installation guide also explains
how to route installation caches and temporary files to the chosen drive.

Model weights and transcripts are not included in the repository. Local inputs are
not overwritten; repeated jobs get separate output folders. Once the selected
model is cached, local-file transcription can run offline.

## Important limits

- `translate` means **translate speech into English**, not into an arbitrary target language. Turbo does not support that task.
- Transcripts and word timings need review; a successful run is not a guarantee of accuracy.
- The JSON tool returns `speaker: null`; this app does not identify speakers.
- Thonburian can produce coarse segment timing; inspect it before using the output as subtitles.
- The URL path accepts a single video, not a playlist. Site restrictions and availability still apply.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `yt_whisper` or `yt_whisper_ui` not found | Use the installed environment. See [launch again](docs/installation.md#launch-again) for explicit Python paths. |
| Missing environment when opening the batch file | Complete installation and use the same `YTW_DATA_DIR` as during setup. |
| Storage drive unavailable | Set `YTW_DATA_DIR` to an existing drive before running. |
| FFmpeg/ffprobe or Node missing | Install the required program and ensure its executable is on PATH. Node is needed only for URLs. |
| GPU unavailable or out of memory | Run `yt_whisper --doctor`; try a smaller model or `--device cpu`. |

## Development

Run tests, build a wheel and reproduce benchmarks using the
[development instructions](docs/usage.md#development).
Dependencies live in `pyproject.toml`; CI is configured for Windows/Linux with
Python 3.12 and CPU PyTorch. CUDA and actual model recognition require separate
integration checks.

## License

MIT. Originally forked from [m1guelpf/yt-whisper](https://github.com/m1guelpf/yt-whisper).
