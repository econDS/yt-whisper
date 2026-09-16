"""Versioned JSON subprocess interface for local transcription clients."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import wave

from .storage import configure_storage

SCHEMA_VERSION = 1
EXIT_CODES = {"invalid_request": 2, "input_error": 3, "inference_error": 4,
              "response_write_error": 5, "internal_error": 1, "interrupted": 130}


class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def invalid(message):
    raise ToolError("invalid_request", message)


def fields(obj, allowed, label):
    if not isinstance(obj, dict):
        invalid(f"{label} must be an object.")
    unknown = obj.keys() - allowed
    if unknown:
        invalid(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        invalid(f"{label} must be a finite number.")
    return float(value)


def validate_request(raw, base_dir):
    from .options import DEFAULTS, decoding_options
    fields(raw, {"schema_version", "request_id", "audio_path", "model", "language",
                 "task", "device", "options", "ranges", "seed"}, "request")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != SCHEMA_VERSION:
        invalid("schema_version must be 1.")
    request_id = raw.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
        invalid("request_id must be a non-empty string or null.")
    source = raw.get("audio_path")
    if not isinstance(source, str) or not source.strip() or "://" in source:
        invalid("audio_path must be a local file path.")
    audio = Path(source).expanduser()
    audio = (base_dir / audio if not audio.is_absolute() else audio).resolve()
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        invalid("model is required.")
    language = raw.get("language")
    if language is not None and not isinstance(language, str):
        invalid("language must be a language code/name, Auto or null.")
    task, device = raw.get("task", "transcribe"), raw.get("device", "auto")
    if task not in ("transcribe", "translate"):
        invalid("task must be transcribe or translate.")
    if device not in ("auto", "cpu", "cuda"):
        invalid("device must be auto, cpu or cuda.")
    seed = raw.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        invalid("seed must be an integer from 0 to 4294967295.")
    options = raw.get("options", {})
    fields(options, set(DEFAULTS) - {"clip_timestamps"}, "options")
    for key in ("word_timestamps", "carry_initial_prompt", "condition_on_previous_text"):
        if key in options and type(options[key]) is not bool:
            invalid(f"options.{key} must be a boolean.")
    if "initial_prompt" in options and options["initial_prompt"] is not None and not isinstance(options["initial_prompt"], str):
        invalid("options.initial_prompt must be a string or null.")
    for key in ("beam_size", "hallucination_silence_threshold"):
        if options.get(key) is not None:
            number(options[key], f"options.{key}")
    temperature = options.get("temperature")
    if temperature not in (None, ""):
        if isinstance(temperature, list):
            for value in temperature:
                number(value, "options.temperature item")
        elif not isinstance(temperature, str):
            number(temperature, "options.temperature")
    try:
        validated_options = decoding_options(model, **options)
    except (TypeError, ValueError) as exc:
        invalid(str(exc))
    ranges = raw.get("ranges")
    if ranges is not None:
        if not isinstance(ranges, list) or not ranges:
            invalid("ranges must be a non-empty array; omit it for the whole file.")
        seen = set()
        clean = []
        for item in ranges:
            fields(item, {"id", "start", "end"}, "range")
            rid = item.get("id")
            if not isinstance(rid, str) or not rid.strip() or rid in seen:
                invalid("Every range needs a unique, non-empty string id.")
            start, end = number(item.get("start"), "range.start"), number(item.get("end"), "range.end")
            if not 0 <= start < end:
                invalid("Ranges require 0 <= start < end.")
            seen.add(rid)
            clean.append({"id": rid, "start": start, "end": end})
        ranges = clean
    # Import model libraries only after storage has been configured by the entry point.
    import whisper
    from .engine import THAI_MODEL, normalize_language
    if model not in whisper.available_models() + [THAI_MODEL]:
        invalid(f"Unknown model: {model}")
    try:
        language = normalize_language(language)
    except ValueError as exc:
        invalid(str(exc))
    if task == "translate" and model in ("turbo", "large-v3-turbo"):
        invalid("Turbo does not support translation.")
    return {"request_id": request_id, "audio_path": audio, "model": model,
            "language": language, "task": task, "device": device, "options": validated_options,
            "ranges": ranges, "seed": seed}


def probe_audio(audio):
    if not audio.is_file():
        raise ToolError("input_error", f"Audio file does not exist: {audio}")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ToolError("input_error", "FFmpeg and ffprobe must be available on PATH.")
    try:
        run = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                              "format=duration:stream=codec_type,duration", "-of", "json", str(audio)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        data = json.loads(run.stdout)
        stream = next(s for s in data["streams"] if s.get("codec_type") == "audio")
        duration = float(stream.get("duration", data.get("format", {}).get("duration")))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid audio duration")
        return duration
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError, StopIteration) as exc:
        raise ToolError("input_error", f"Cannot read audio duration: {audio}") from exc


def extract_range(audio, destination, start, end):
    # Output seeking decodes before seeking; all ranges use the first audio stream.
    try:
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-i", str(audio), "-ss", str(start), "-t", str(end - start),
                        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(destination)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        with wave.open(str(destination), "rb") as clip:
            duration = clip.getnframes() / clip.getframerate()
        if duration <= 0 or abs(duration - (end - start)) > .1:
            raise ValueError("Decoded range duration does not match the requested range.")
        return duration
    except (OSError, ValueError, wave.Error, subprocess.CalledProcessError) as exc:
        raise ToolError("input_error", f"Cannot decode range {start}–{end}: {exc}") from exc


def finite_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def timestamp(value, offset):
    value = finite_value(value)
    return round(value + offset, 6) if value is not None else None


def timing(start, end, lower, upper):
    if start is None or end is None:
        return "missing"
    return "valid" if lower - .001 <= start <= end <= upper + .001 else "out_of_range"


def normalize_segments(raw, start, end):
    segments = []
    for index, segment in enumerate(raw):
        begin, finish = timestamp(segment.get("start"), start), timestamp(segment.get("end"), start)
        words = []
        for word in segment.get("words", []):
            wb, we = timestamp(word.get("start"), start), timestamp(word.get("end"), start)
            words.append({"text": word.get("word", ""), "start": wb, "end": we,
                          "probability": finite_value(word.get("probability")), "timing_status": timing(wb, we, start, end)})
        segments.append({"id": index, "start": begin, "end": finish, "text": segment["text"],
                         "speaker": None, "timing_status": timing(begin, finish, start, end), "words": words,
                         "diagnostics": {key: finite_value(segment[key]) for key in (
                             "avg_logprob", "no_speech_prob", "compression_ratio") if key in segment}})
    return segments


def base_response():
    return {"schema_version": SCHEMA_VERSION, "request_id": None, "status": "failed",
            "timestamp_basis": "source_audio_seconds", "review_status": "unreviewed",
            "source": None, "configuration": None, "results": [], "error": None}


def mark_failure(response, exc):
    response["status"] = "partial" if any(r["status"] == "complete" for r in response["results"]) else "failed"
    response["error"] = {"code": exc.code, "message": str(exc)}
    return EXIT_CODES[exc.code]


def execute(request, storage, response):
    from .engine import Transcriber
    audio = request["audio_path"]
    duration = probe_audio(audio)
    ranges = request["ranges"] or [{"id": "full", "start": 0.0, "end": duration}]
    for item in ranges:
        if item["end"] > duration:
            invalid(f"Range {item['id']} ends after the source duration ({duration:.6f} seconds).")
    with audio.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    response["source"] = {"audio_path": str(audio), "sha256": digest, "duration_seconds": duration,
                          "audio_stream": "0:a:0"}
    response["configuration"] = {key: request[key] for key in ("model", "language", "task", "device", "options", "seed")}
    response["versions"] = {}
    for package in ("yt-whisper", "openai-whisper", "torch", "transformers"):
        try:
            response["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            response["versions"][package] = None
    response["results"] = [dict(item, status="not_run") for item in ranges]
    engine = Transcriber(storage)
    import torch
    import numpy as np
    for index, item in enumerate(ranges):
        target = response["results"][index]
        began = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="tool-", dir=storage.temp) as temporary:
                clip = Path(temporary) / "audio.wav"
                decoded_duration = extract_range(audio, clip, item["start"], item["end"])
                # Ranges are independent; seed and decoding context reset for each one.
                torch.manual_seed(request["seed"])
                np.random.seed(request["seed"])
                try:
                    result = engine.transcribe(
                        clip, request["model"], request["language"], request["task"], request["device"],
                        **dict(request["options"], verbose=False))
                except Exception as exc:
                    raise ToolError("inference_error", str(exc)) from exc
            segments = normalize_segments(result["segments"], item["start"], item["end"])
            warnings = []
            if not result["text"].strip():
                warnings.append("empty_transcript")
            if any(s["timing_status"] != "valid" or any(w["timing_status"] != "valid" for w in s["words"]) for s in segments):
                warnings.append("timing_needs_review")
            if request["model"] == "Thai_Thonburian":
                warnings.append("backend_may_use_coarse_clip_boundary_timestamps")
            target.update(status="complete", text=result["text"].strip(), segments=segments,
                          model=result["model"], language=result.get("language"), device=result["device"],
                          decoded_duration_seconds=decoded_duration, warnings=warnings,
                          elapsed_seconds=round(time.perf_counter() - began, 6))
        except (Exception, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                exc = ToolError("interrupted", "Transcription interrupted.")
            elif not isinstance(exc, ToolError):
                exc = ToolError("internal_error", str(exc))
            target.update(status="failed", error={"code": exc.code, "message": str(exc)})
            return mark_failure(response, exc)
    response["status"] = "complete"
    return 0


def atomic_response(path, response):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".response-", suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(response, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        invalid(message)


def strict_json(text):
    def constant(value):
        invalid(f"Non-finite JSON number is not allowed: {value}")
    def pairs(values):
        obj = {}
        for key, value in values:
            if key in obj:
                invalid(f"Duplicate JSON field: {key}")
            obj[key] = value
        return obj
    try:
        return json.loads(text, parse_constant=constant, object_pairs_hook=pairs)
    except json.JSONDecodeError as exc:
        invalid(f"Invalid JSON: {exc}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    response, output, code = base_response(), None, 0
    try:
        parser = Parser(description=__doc__)
        parser.add_argument("--request", required=True, help="UTF-8 JSON file; relative audio paths use its directory")
        parser.add_argument("--response", help="Optional atomic response JSON file; relative to working directory")
        args = parser.parse_args(argv)
        source = Path(args.request).expanduser().resolve()
        candidate_output = Path(args.response).expanduser().resolve() if args.response else None
        if candidate_output == source:
            invalid("--response must not overwrite the request file.")
        try:
            raw = strict_json(source.read_text(encoding="utf-8-sig"))
        except UnicodeError as exc:
            raise ToolError("invalid_request", "Request must be UTF-8 JSON.") from exc
        except OSError as exc:
            raise ToolError("input_error", str(exc)) from exc
        # Reject output aliases of the input before making any output file writable.
        if isinstance(raw, dict) and isinstance(raw.get("audio_path"), str):
            audio = (source.parent / Path(raw["audio_path"]).expanduser()).resolve()
            if candidate_output == audio or (candidate_output and candidate_output.exists() and audio.exists()
                                             and candidate_output.samefile(audio)):
                invalid("--response must not overwrite the audio file.")
        if candidate_output and candidate_output.exists() and source.samefile(candidate_output):
            invalid("--response must not overwrite the request file.")
        output = candidate_output
        if isinstance(raw, dict) and isinstance(raw.get("request_id"), str):
            response["request_id"] = raw["request_id"]
        # Reserve stdout for exactly one response. Library chatter goes to stderr.
        with redirect_stdout(sys.stderr):
            storage = configure_storage()
            request = validate_request(raw, source.parent)
            code = execute(request, storage, response)
    except ToolError as exc:
        code = mark_failure(response, exc)
    except KeyboardInterrupt:
        code = mark_failure(response, ToolError("interrupted", "Transcription interrupted."))
    except Exception as exc:
        code = mark_failure(response, ToolError("internal_error", str(exc)))
    if output:
        try:
            atomic_response(output, response)
        except (OSError, ValueError, TypeError) as exc:
            code = mark_failure(response, ToolError("response_write_error", str(exc)))
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
