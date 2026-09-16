# Usage and model settings

Commands below assume the installed environment is on PATH. With an explicit
Windows Python path, replace `yt_whisper` with `& $py -m yt_whisper.cli`.
See [installation and launch instructions](installation.md).

## Models

| Model | Use |
| --- | --- |
| base | Small model for checking that the installation works. |
| small, medium, large-v3 | Multilingual models with increasing memory requirements. Evaluate on your own audio. |
| turbo / large-v3-turbo | Faster transcription option to compare with large-v3; translation is not supported. |
| Thai_Thonburian | Optional Thai backend through Transformers; install the thai extra. |

The CLI defaults to base. Models download to the configured data folder when first
selected if they are not already cached. Each running process retains one model;
switching models releases the previous one. The UI serializes transcription requests.
Separate CLI/UI/tool processes still consume memory independently.

**Translate means translate speech into English.** `.en` models force English.
Use `--language Auto` to detect the language. CUDA falls back to CPU when unavailable.

For Thonburian, the app uses a local `models/thonburian` directory when its config
exists. Set `YTW_THAI_MODEL` to another local directory if needed. Otherwise it
downloads `biodatlab/whisper-th-medium-combined` through the configured Hugging Face
cache. A local model needs safetensors weights, configuration, tokenizer and feature
extractor files; a duplicate pytorch_model.bin is not required.
Missing ending timestamps may be filled with the clip boundary; inspect subtitle timing.

## Files, URLs and outputs

```shell
yt_whisper "audio.wav" --model base --language Auto --format all
yt_whisper "video.mp4" --model turbo --language ja --format json
yt_whisper "audio.wav" --model medium --language ja --task translate
yt_whisper "audio.wav" --device cpu --output_dir transcripts
```

Formats: TXT, JSON, SRT, VTT, TSV and JSONL. `--format all` writes all six;
the CLI default is VTT. JSON preserves segment details and records model, task,
device, source and decoding options. TSV timestamps are integer milliseconds.
JSONL contains one complete segment per line, including word details when enabled.
It uses this app's writer and does not require Whisper main.

Local inputs are decoded without writing an MP3 beside the source. Output folders
are unique. For URL jobs, the downloader owns a temporary directory and removes it
after success or failure. Gradio uploads are swept hourly for files older than
24 hours while the UI is running.

Playlists are rejected; a single video URL containing a playlist parameter processes
only the selected video. The downloader does not automatically read browser cookies.
Site restrictions and network availability can still affect downloads.

## Advanced transcription

CLI and UI expose these options for the OpenAI Whisper backend. The UI hides them
for Thonburian; unsupported explicit CLI settings are rejected for that backend.

| Option | Purpose |
| --- | --- |
| --initial-prompt | Supply names, spellings or terminology as context. |
| --carry-initial-prompt | Repeat a non-empty initial prompt for each decoding window. |
| --word-timestamps | Include estimated word start/end times. |
| --no-condition-on-previous-text | Disable previous-text context; useful to try if phrases repeat. |
| --hallucination-silence-threshold | Positive silence threshold in seconds; requires word timestamps. |
| --clip-timestamps | Ranges in seconds, such as 10,30,45,60; a final start runs to the end. |
| --beam-size | Positive integer; beam search applies at temperature 0. |
| --temperature | A value from 0 to 1, or a comma-separated fallback sequence. |

Omit temperature to preserve Whisper's default fallback sequence. A beam size with
a temperature sequence containing no zero is rejected.

```shell
yt_whisper "audio.wav" --model turbo --language ja --word-timestamps --clip-timestamps "10,40" --beam-size 5 --temperature 0 --format all
```

Other options include `--verbose True|False` and `--break-lines`.
Run `yt_whisper --help` for the full CLI argument list.

For independent named ranges, structured errors and machine-readable stdout,
use the [JSON tool](tool-interface.md). Its range coordinates refer to the supplied
source file. It does not identify speakers or make editorial corrections.

## Benchmarks

Install the dev extra, run from the repository, and use your environment's Python:

```shell
python scripts/benchmark.py --audio "audio.wav" --model turbo --language ja --device cuda --beam-size 5 --temperature 0 --output "benchmarks/run.json"
```

The example writes its report below the current working directory; choose an
absolute `--output` path if it should be on another drive. Use `--device cpu`
for CPU measurements. Benchmark defaults are language th and device cuda, so
specify these explicitly for other audio or hardware.

Pass `--reference reference.txt` for character error rate (CER). The script removes
only whitespace from reference and hypothesis; spelling and punctuation remain.
Lower CER is better. Without a verified reference it reports performance only.

Metrics separate model loading from inference and include sampled peak process
RAM and peak CUDA allocation. Single desktop runs are not controlled hardware
benchmarks. Seed 0 is the default (`--seed` overrides it); results may still vary
across hardware/software versions. Use several representative labeled clips to
choose settings.

`--whisper-source` selects a separate pinned Whisper checkout for comparison;
`--standard-initialization` compares upstream parameter initialization with the
app's memory-saving initialization. Neither option changes the installed package.

## Development

From the repository, with the selected environment's Python:

```shell
python -m pip install -e ".[ui,thai,dev]"
python -m pip check
python -m pytest -q
python -m build --wheel
```

If installation/build temporary files should live on a chosen drive, set the
environment variables shown in the [installation guide](installation.md) first.
Pytest configures its data folder from the system temp directory unless
`YTW_DATA_DIR` is already set. Tests do not download model weights.

The suite covers file preservation, cleanup, duplicate titles, config defaults,
language handling, CPU fallback, model reuse, UI construction, option validation,
export formats and the JSON tool's timing/error contract.

The GitHub Actions workflow tests and builds on Windows/Linux with Python 3.12
and CPU PyTorch. It runs when changes reach GitHub; a local test pass does not
establish that a remote CI run or GPU recognition test has passed.
