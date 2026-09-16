"""Shared inference with one cached model and serialized GPU access."""
import gc
import os
from pathlib import Path
import threading
import warnings

from .models import LEGACY_THAI_MODEL, canonical_model_id, custom_model_spec, model_ids

# Preserve imports of the original public constant.
THAI_MODEL = LEGACY_THAI_MODEL


def normalize_language(value):
    if value is None or str(value).strip().lower() in ("", "auto"):
        return None
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
    value = str(value).strip().lower()
    if value in LANGUAGES:
        return value
    if value in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[value]
    raise ValueError(f"Unknown language: {value}")



def validate_selection(model_name, language=None, task="transcribe"):
    """Resolve aliases and reject unsupported combinations before model loading."""
    model_name = canonical_model_id(model_name)
    if model_name not in model_ids():
        raise ValueError(f"Unknown model: {model_name}")
    language = normalize_language(language)
    if task not in ("transcribe", "translate"):
        raise ValueError("Task must be transcribe or translate (to English).")
    spec = custom_model_spec(model_name)
    if spec:
        if task not in spec.tasks:
            raise ValueError(f"{spec.display_name} supports Thai transcription only; translate is not supported.")
        if spec.thai_only:
            if language not in (None, "th"):
                raise ValueError(f"{spec.display_name} is Thai-specialized; select Auto or Thai (th).")
            language = "th"
    if model_name in ("turbo", "large-v3-turbo") and task == "translate":
        raise ValueError("Turbo cannot translate. Choose a multilingual model such as medium or large-v3.")
    if model_name.endswith(".en"):
        language = "en"
    return model_name, language


def resolve_device(device):
    import torch
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda.")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; using CPU.")
        return "cpu"
    return device


def thai_segments(chunks, duration, text):
    """Keep text even when Whisper omits the final timestamp at a clip boundary."""
    segments = []
    for chunk in chunks or []:
        if not chunk.get("text", "").strip():
            continue
        start, end = chunk.get("timestamp") or (None, None)
        start = segments[-1]["end"] if start is None and segments else (start or 0.0)
        start = min(duration, max(0.0, float(start)))
        end = duration if end is None else min(duration, max(start, float(end)))
        segments.append({"start": start, "end": end, "text": chunk["text"]})
    if not segments and text.strip():
        segments = [{"start": 0.0, "end": duration, "text": text}]
    return segments

class Transcriber:
    def __init__(self, storage):
        self.storage = storage
        self._lock = threading.RLock()
        self._key = None
        self._model = None

    def _load(self, model_name, device):
        import torch
        import whisper
        model_name = canonical_model_id(model_name)
        spec = custom_model_spec(model_name)
        thai_source = self.storage.thonburian_source(spec) if spec else None
        key = (model_name, device, thai_source)
        if self._key == key:
            return self._model
        self._model = None
        self._key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if spec:
            try:
                from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline
            except ImportError as exc:
                raise RuntimeError('Install Thai support with: pip install ".[thai]"') from exc
            local = Path(thai_source).is_dir()
            kwargs = {"cache_dir": os.environ["HF_HUB_CACHE"], "local_files_only": local}
            processor = AutoProcessor.from_pretrained(thai_source, **kwargs)
            model = WhisperForConditionalGeneration.from_pretrained(
                thai_source, dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True, **kwargs,
            )
            self._model = pipeline(
                "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor, device=0 if device == "cuda" else -1,
                chunk_length_s=30, batch_size=1,
            )
        else:
            # Initialize parameters on the target device to avoid a second large CPU copy.
            with torch.device(device):
                self._model = whisper.load_model(model_name, device=device, download_root=str(self.storage.models))
        self._key = key
        return self._model

    def transcribe(self, path, model_name="base", language=None, task="transcribe", device="auto",
                   **options):
        import whisper
        from .options import decoding_options
        model_name, language = validate_selection(model_name, language, task)
        spec = custom_model_spec(model_name)
        options = decoding_options(model_name, **options)
        device = resolve_device(device)
        with self._lock:
            model = self._load(model_name, device)
            if spec:
                generate = {"task": task, "language": language}
                waveform = whisper.load_audio(str(path))
                raw = model(waveform, generate_kwargs=generate, return_timestamps=True)
                segments = thai_segments(raw.get("chunks", []), len(waveform) / 16000.0, raw["text"])
                result = {"text": raw["text"].strip(), "segments": segments, "language": language}
            else:
                result = model.transcribe(str(path), language=language, task=task,
                                          fp16=device == "cuda", **options)
            result["model"] = "large-v3" if model_name == "large" else model_name
            if spec:
                result["model_hf_repo"] = spec.hf_repo
                result["model_source"] = self._key[2]
                result["model_backend"] = spec.backend
                result["model_revision"] = getattr(model.model.config, "_commit_hash", None)
            result["decoding_options"] = {k: v for k, v in options.items() if k != "verbose"}
            result["task"] = task
            result["device"] = device
            return result
