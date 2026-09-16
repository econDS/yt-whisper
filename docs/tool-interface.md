# JSON tool interface (schema version 1)

Use yt-whisper as a local subprocess from another project. No UI or HTTP server is
required. Episode naming, transcript comparison, editorial decisions and translation
remain with the caller.

## Invoke

```powershell
& 'E:\yt-whisper\envs\py312\python.exe' -m yt_whisper.tool --request 'E:\jobs\request.json' --response 'E:\jobs\response.json'
```

After installation the equivalent entry point is `yt_whisper_tool`. The module
command works with the existing editable installation. Both work from any directory.

- `--request`: UTF-8 JSON file (UTF-8 BOM accepted).
- `--response`: optional response file. Written atomically; replaces an existing
  response at that explicit path. Cannot overwrite the request or input audio.
  Relative response paths use the caller's working directory. On early JSON/input
  path errors only stdout is guaranteed; an old response file may remain. Always
  check the process exit code and parse stdout for the current invocation.
- **stdout contains one UTF-8 JSON object**, also when validation or inference fails.
  Progress and library diagnostics go to stderr. `--help` is ordinary help text.
- Configure `YTW_DATA_DIR` if necessary. The Windows default is E:/yt-whisper.
  Model downloads and temporary WAV files use that data root.
- FFmpeg **and ffprobe** must be on PATH. No API key or network call is required
  when the selected model is already cached.

## Request

```json
{
  "schema_version": 1,
  "request_id": "review-001",
  "audio_path": "episode.audio.flac",
  "model": "turbo",
  "language": "ja",
  "task": "transcribe",
  "device": "auto",
  "seed": 0,
  "options": {
    "beam_size": 5,
    "temperature": 0,
    "word_timestamps": true
  },
  "ranges": [
    {"id": "check-a", "start": 450, "end": 480},
    {"id": "check-b", "start": 930, "end": 960}
  ]
}
```

| Field | Contract |
| --- | --- |
| schema_version | Required integer 1. Unknown versions/fields are rejected. |
| request_id | Optional non-empty string or null; echoed unchanged. |
| audio_path | Required existing local audio/video file. Relative paths resolve against the request file's directory. URLs are not accepted in this interface. |
| model | Required name from Whisper's available models, or Thai_Thonburian. |
| language | Optional language code/name, Auto or null. Default is automatic detection. Japanese audio uses ja. |
| task | transcribe (default) or translate into English. Turbo translation is rejected. |
| device | auto (default), cpu or cuda. Existing engine CPU fallback applies; each result records the actual device. |
| seed | Integer 0–4294967295, default 0. Reset separately for each range; not a guarantee of identical results across software/hardware versions. |
| options | Existing decoding options, excluding clip_timestamps and verbose. Ranges control slicing. Boolean options must be JSON booleans. Thonburian accepts only its supported defaults. |
| ranges | Optional non-empty list of unique id/start/end objects. Omit or use null for one full-file range with id full. Units are seconds; require 0 <= start < end <= source duration. |

Optional decoding controls (additive to schema version 1):

```json
"options": {
  "preset": "official-cli",
  "beam_size": 3,
  "no_speech_threshold": null,
  "word_timestamps": true
}
```

The preset sets beam size 5, best-of 5 and temperature fallback
0,0.2,0.4,0.6,0.8,1. Explicit fields override it. Omit `preset` or use `default`
to retain existing behavior. A numeric `temperature: 0` disables fallback.
`best_of` is ignored at temperature 0. `beam_size: null` / `best_of: null`
use the upstream unset search setting, even when a preset was selected.

Threshold fields are `compression_ratio_threshold` (positive, default 2.4),
`logprob_threshold` (finite, default -1.0), and `no_speech_threshold`
(0 through 1, default 0.6). Omit to retain defaults; JSON `null` disables the
individual threshold. Responses record resolved values in `configuration.options`,
including disabled thresholds. The preset name is resolved away before inference.
Subtitle layout options belong to CLI/UI exports and are not accepted here.

All ranges are validated before model loading. Ranges may overlap and may be in
any order; results preserve request order. They are **independent transcriptions**:
no previous text crosses range boundaries, and overlapping text is not deduplicated.
Include any desired context in the requested range and select the core text on the
caller side. The same engine/model is reused within one request process.

Each range is decoded from the first audio stream (`0:a:0`) into a temporary
16 kHz mono PCM WAV on the data drive. Temporary files are removed after each
range, including exceptions. No files are written beside the input.

## Response

Top-level fields:

- `schema_version`, `request_id`.
- `status`: complete, partial or failed.
- `timestamp_basis`: source_audio_seconds. Zero is the beginning of the decoded
  first audio stream of the provided file. These are not times relative to the
  extracted temporary clip, nor wall-clock/container PTS values. If the caller
  supplied a pre-cut file, it must add that file's offset into the original episode.
- `review_status`: unreviewed. Completion does not establish accuracy.
- `source`: resolved input path, SHA-256, duration_seconds, audio_stream; null if
  input/preflight has not completed.
- `configuration`: requested model/language/task/device, normalized options and
  seed; null on preflight failure.
- `versions`: package versions, when execution reached this stage. Keep the
  caller's tool checkout revision too when using an editable installation.
- `results`: one entry per requested range after successful preflight.
- `error`: null, or an object with stable `code` and human-readable `message`.

A completed range has `id`, requested `start`/`end`, `status`, `text`,
`segments`, actual `model`/`device`, detected/effective `language`,
`decoded_duration_seconds`, `warnings`, and `elapsed_seconds`.
Elapsed time includes decoding and inference; the first range also includes model
loading. Other processes, including the UI, retain their own model instances.

Each segment has `id` (local to range), `start`, `end`, `text`,
`speaker: null`, `timing_status`, `words` and `diagnostics`.
Each word has `text`, `start`, `end`, `probability` and `timing_status`.
No speaker identities are inferred. Diagnostics/probabilities are model outputs,
not calibrated accuracy scores. Non-finite or unavailable numerical scores are null.

Timing status is valid, missing or out_of_range. Invalid/missing endpoints remain
null when unavailable; finite out-of-range values are preserved and flagged rather
than silently clamped. Word/segment timing remains an estimate requiring review.
Thonburian can return coarse timestamps filled from the clip boundary; a warning
is always included for that backend. Empty text succeeds with an empty_transcript
warning and is not evidence that the range was silent.

## Failures and exit codes

| Exit | Error code | Meaning |
| ---: | --- | --- |
| 0 | null | All requested ranges completed. |
| 2 | invalid_request | Bad JSON/schema/options/ranges or conflicting output path. |
| 3 | input_error | Missing input, FFmpeg/ffprobe problem or decoding failed. |
| 4 | inference_error | Model loading or transcription failed. |
| 5 | response_write_error | Could not persist the optional response; stdout still carries results. |
| 130 | interrupted | User interrupted execution. |
| 1 | internal_error | Unexpected failure; inspect the response and stderr. |

Processing stops at the first failed range. Earlier results remain complete, the
failed range includes its error, and subsequent ranges are not_run. Top-level status
is partial if any results were completed, otherwise failed. There is no automatic
retry or disk resume. The caller can submit only failed/not_run range IDs in a new
request. A process killed externally or a machine shutdown may prevent any final
response; do not equate an absent response with success.

## Python caller from another project

```python
import json
import subprocess

run = subprocess.run(
    [
        r"E:\yt-whisper\envs\py312\python.exe",
        "-m", "yt_whisper.tool",
        "--request", r"E:\jobs\request.json",
        "--response", r"E:\jobs\response.json",
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
)
response = json.loads(run.stdout)  # Parse even when returncode is nonzero.
for item in response["results"]:
    if item["status"] == "complete":
        consume_transcript(item)  # Caller-owned comparison/editorial logic.
if run.returncode:
    report_failure(response["error"], run.stderr)
```

Use the existing episode job's audio path rather than downloading another copy.
Keep tool responses under the caller's own run directories. This interface does not
read episode state, call Gemini, modify hybrid drafts or select editorial wording.
