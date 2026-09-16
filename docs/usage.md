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
| thonburian-medium | Existing Thonburian Medium Thai baseline; legacy Thai_Thonburian remains an alias. |
| thonburian-large-v3 | Full Thonburian Large-v3 model for Thai ASR. |
| thonburian-distill-large-v3 | Smaller Thai-specialized Large-v3 Turbo derivative. |

The CLI defaults to base. Models download to the configured data folder when first
selected if they are not already cached. Each running process retains one model;
switching models releases the previous one. The UI serializes transcription requests.
Separate CLI/UI/tool processes still consume memory independently.

**Translate means translate speech into English.** `.en` models force English.
Use `--language Auto` to detect the language with official models. For Thonburian,
Auto resolves to Thai; explicit Thai/th also works, and other languages or translate
are rejected before loading. CUDA falls back to CPU when unavailable.

Thonburian uses the optional `.[thai]` dependencies and each checkpoint's own
processor. Models have separate local directories and share the configured Hugging
Face cache. The legacy `models/thonburian` directory and `YTW_THAI_MODEL` override
continue to select Medium only. See [Thonburian models and offline setup](thonburian.md).
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

### Browser modes

Simple mode uses the selected model/language with the default transcription
settings and automatic device selection. Advanced settings are retained on screen
when switching modes, but Simple does not apply them. Simple exports TXT, JSON,
SRT and VTT to the configured outputs directory.

Advanced mode exposes task/device, all six output formats, OpenAI decoding and
word-based subtitle controls, plus segment wrapping (`--break-lines`). The output
subfolder must stay inside the configured outputs directory. CLI `--output_dir`
still allows another destination. Check setup runs the same diagnostics as
`--doctor`, without loading model weights. Multi-file batches and terminal
verbosity remain CLI features.

Use the Copy icon in the transcript toolbar to copy text. Editing the textbox
changes copied text only; exported files keep the original transcription. During
processing Gradio shows stage/progress information; the final status shows total
elapsed time including preparation, model loading/inference and export, excluding
queue wait and browser upload. This is not the benchmark's inference-only metric.

Existing Gradio API endpoints `/transcribe_file` and `/transcribe_url` retain their
two-output `(text, files)` contract. UI events use separate endpoints ending in
`_ui`, returning `(text, files, status)` with mode/export controls. Integrations
should continue using the existing endpoints or the versioned JSON subprocess tool.

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
| --preset | default (existing behavior) or official-cli decoding settings. |
| --best-of | Positive candidate count at nonzero temperatures; ignored at temperature 0. |
| --compression-ratio-threshold | Repetition fallback threshold, default 2.4; none disables it. |
| --logprob-threshold | Low-confidence fallback threshold, default -1.0; none disables it. |
| --no-speech-threshold | Silence probability threshold from 0 to 1, default 0.6; none disables it. |
| --beam-size | Positive integer; beam search applies at temperature 0. |
| --temperature | A value from 0 to 1, or a comma-separated fallback sequence. |

Omit temperature to preserve Whisper's default fallback sequence. A beam size with
a temperature sequence containing no zero is rejected.

```shell
yt_whisper "audio.wav" --model turbo --language ja --word-timestamps --clip-timestamps "10,40" --beam-size 5 --temperature 0 --format all
```

### Decoding preset and thresholds

`--preset official-cli` sets beam size 5, best-of 5, and temperature fallback
0,0.2,0.4,0.6,0.8,1, matching the official Whisper v20250625 CLI decoding defaults.
Explicit values override the preset. It does not change the selected model,
output formats, device or verbosity. Existing commands keep their decoding defaults.

`--temperature 0` means a single temperature and disables fallback in this app,
including with the preset. Best-of is then ignored. Unlike the official CLI,
this app does not implicitly expand a supplied temperature; supply the full sequence.
More candidates may improve some clips but require more time and memory.

Thresholds control fallback and silence decisions; they do not certify accuracy.
Whisper skips a segment when its no-speech probability exceeds the threshold,
unless its average log probability exceeds the logprob threshold. Use representative
labeled audio to compare settings before applying them to a full collection.

In the UI, empty fields inherit preset/default values. Type `none` in a threshold
field to disable that check. In the JSON tool, use JSON `null` instead.

```shell
yt_whisper "audio.wav" --model turbo --language ja --preset official-cli --format all
yt_whisper "audio.wav" --preset official-cli --temperature 0 --no-speech-threshold none
```

### Word-based subtitles

With `--word-timestamps`, SRT/VTT can use the official Whisper subtitle writer:

- `--max-line-width 42 --max-line-count 2`: wrap into up to two lines per cue.
- `--max-words-per-line 8`: an alternative to character-based wrapping.
- `--highlight-words`: underline each word during its estimated timing.

```shell
yt_whisper "audio.wav" --model turbo --word-timestamps --max-line-width 42 --max-line-count 2 --highlight-words --format all
```

These options affect SRT/VTT only; JSON, JSONL, TSV and TXT preserve transcription
content and timings. Without layout options the existing segment-based subtitle
writer remains in use. Do not combine these options with `--break-lines`.
A line count requires a line width; choose either word count or character width.
A single long word is not split, and "word" units depend on Whisper's language
alignment, so a width limit is not a guarantee for every language.

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

The same script accepts all three Thonburian identifiers (or their exact Hugging
Face repository IDs) alongside `large-v3` and `turbo`. See the
[five-model comparison example](thonburian.md#compare-on-the-same-audio).
Reports include the canonical model ID, requested ID/device, actual device,
Thonburian repository/source/revision when available, audio duration, real-time
factor and paths to all transcript exports. OpenAI decoding flags are rejected
for Thonburian; `--standard-initialization` applies only to official models.

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
