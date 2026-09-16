# Thonburian Whisper for Thai

All three models use Transformers' `WhisperForConditionalGeneration`; official
OpenAI Whisper models continue to use the existing OpenAI implementation.
Install Thai support from the repository in your chosen environment:

```shell
python -m pip install -e ".[thai]"
# Include the browser UI if needed:
python -m pip install -e ".[ui,thai]"
```

Installing the extra does not download weights. The selected checkpoint downloads
on first use unless its local directory or configured Hugging Face cache already
contains it. Models and temporary files follow `YTW_DATA_DIR`.

## Choices and compatibility

| UI label / internal ID | Hugging Face checkpoint | Role |
| --- | --- | --- |
| Thonburian Medium / `thonburian-medium` | [biodatlab/whisper-th-medium-combined](https://huggingface.co/biodatlab/whisper-th-medium-combined) | Existing Medium baseline, retained for comparison and compatibility. |
| Thonburian Large-v3 / `thonburian-large-v3` | [biodatlab/whisper-th-large-v3-combined](https://huggingface.co/biodatlab/whisper-th-large-v3-combined) | Full newer Large-v3 Thai model. |
| Thonburian Distilled Large-v3 / `thonburian-distill-large-v3` | [biodatlab/distill-whisper-th-large-v3](https://huggingface.co/biodatlab/distill-whisper-th-large-v3) | Smaller Thai-specialized derivative of Large-v3 Turbo. |

`Thai_Thonburian` remains an alias for Medium in CLI, config, UI API and JSON tool
requests. Config loading resolves it without changing the file; a later normal UI
save writes the canonical ID. CLI/tool also accept the exact Hugging Face IDs above.
Result `model` values use canonical IDs, including when an alias was requested.
Callers comparing the old string should accept `thonburian-medium` in responses.

```shell
yt_whisper "audio.wav" --model thonburian-medium --language th --format all
yt_whisper "audio.wav" --model thonburian-large-v3 --language Auto --format all
yt_whisper "audio.wav" --model thonburian-distill-large-v3 --language Thai --format all
```

These choices are for Thai ASR. Auto/omitted language selects `th`; explicit
Thai/th works. Other languages and `translate` fail before model loading. The
checkpoint cards document Thai transcription, not validated speech translation;
the presence of multilingual/translation tokens does not establish suitability.
Use an official multilingual Whisper model for translation to English.

The UI displays friendly names and hides OpenAI-specific decoding/layout controls
for all Thonburian variants. Explicit unsupported CLI/tool options are rejected.
Each process caches one model under its canonical ID, device and actual source.
Aliases reuse that cache; switching variants or source releases the previous model.

## Local and offline weights

Supply a complete checkpoint directory with safetensors weights, `config.json`,
`generation_config.json`, tokenizer and processor/feature-extractor files from the
same model repository. A duplicate `pytorch_model.bin` is not required.

| Model | Directory under data root | Optional override |
| --- | --- | --- |
| Medium | `models/thonburian/` | `YTW_THONBURIAN_MEDIUM_MODEL` |
| Large-v3 | `models/thonburian-large-v3/` | `YTW_THONBURIAN_LARGE_V3_MODEL` |
| Distilled Large-v3 | `models/thonburian-distill-large-v3/` | `YTW_THONBURIAN_DISTILL_LARGE_V3_MODEL` |

An override takes precedence over the default directory. The old `YTW_THAI_MODEL`
override still applies to Medium only, below `YTW_THONBURIAN_MEDIUM_MODEL` in
precedence. The old Medium directory is never automatically used for another
variant. As before, an override may identify a local directory or a compatible
Hub repository; it is the caller's responsibility to supply the intended weights.
The actual source is recorded separately from the registry's checkpoint ID.

Default directories are used when `config.json` exists; otherwise loading uses
the model's Hub ID and `HF_HUB_CACHE` configured under the data root. Local
directories load with `local_files_only=True`. Set `HF_HUB_OFFLINE=1` to enforce
offline operation with already populated Hub caches too. Missing/incomplete local
files raise a loading error. CUDA uses FP16, CPU uses FP32; unavailable CUDA falls
back to CPU.

## Model-specific details and timing limits

Configs/cards inspected on 2026-09-16:

| Model | Encoder / decoder layers | Mel bins | Inspected Hub revision |
| --- | --- | --- | --- |
| Medium | 24 / 24 | 80 | `eebf84255cc7f242a504f64ec09ec33d32903fe1` |
| Large-v3 | 32 / 32 | 128 | `ba7197f618400e41b4826c18b2f48e0bc45ed3ca` |
| Distilled Large-v3 | 32 / 4 | 128 | `62df42cecab9f484226ad5f9afdb557552021bbb` |

All three configs declare `WhisperForConditionalGeneration` and all publish
safetensors. Each uses its own processor, so the Medium 80-bin features are not
reused for the 128-bin models. The Distilled config identifies Large-v3 Turbo as
its base. Some older cards contain stale example code, copied Medium labels or
framework version differences; loading follows the actual checkpoint config.

Generation passes `language="th"` and `task="transcribe"` explicitly through
Transformers' current API, rather than editing deprecated `forced_decoder_ids`.
The existing 30-second chunk pipeline, batch size 1 and segment timestamps remain.
No dependency constraint change was needed for this integration.
Transformers warns that pipeline chunking for sequence-to-sequence models is
experimental and recommends native Whisper generation for long-form audio.
This change retains the existing chunking behavior; evaluate long recordings and
chunk boundaries separately before relying on their timing or accuracy.

The `thai_segments()` boundary fallback remains for all variants: a missing final
end is filled with the audio boundary, empty chunks are ignored, and absent usable
chunks with non-empty text produce a full-clip segment. These estimates retain
text for SRT/VTT but need timing review. Word timestamps/highlighting are not
exposed for Thonburian. TXT/JSON/SRT/VTT/TSV/JSONL still use the shared exporters,
unique output directories and source metadata.

Thonburian JSON adds `model_hf_repo`, `model_source`, `model_backend` and
`model_revision`. Local directories may have no Hub revision, in which case it is
null. Keep your own checkpoint/version record for reproducible local benchmarks.

## Compare on the same audio

Install `.[thai,dev]`, then run each model in its own process using the same audio
and reference. This PowerShell example writes on E:; change the output folder for
your machine. It loads/downloads each model when executed, unlike unit tests.

```powershell
$benchmarkDir = 'E:/yt-whisper/benchmarks/comparison'
$models = 'thonburian-medium', 'thonburian-large-v3', 'thonburian-distill-large-v3', 'large-v3', 'turbo'
foreach ($modelName in $models) {
    python scripts/benchmark.py --audio 'audio.wav' --reference 'reference.txt' --language th --device cuda --model $modelName --output "$benchmarkDir/$modelName.json"
}
```

Omit `--reference` for performance-only measurements. CER removes whitespace from
both Thai reference and hypothesis; it does not remove punctuation or normalize
spelling. Lower CER is better. Benchmark reports record the resolved model/source,
actual device, load/transcription time, audio duration, real-time factor, sampled
peak RAM, peak CUDA allocation and transcript paths. Below 1 real-time factor means
transcription took less time than the audio duration. CUDA allocation is PyTorch's
peak allocation, not total GPU usage across other processes.

Use representative audio and repeated runs on your hardware before choosing a
model. Smaller architecture does not guarantee better speed or accuracy on every
clip. Do not pass OpenAI-only decoding options or `--standard-initialization` to
Thonburian runs. Unit tests mock model loading and never download weights; they
verify integration contracts rather than recognition quality on these checkpoints.
