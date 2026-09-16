"""Validate decoding options identically for CLI, UI and direct callers."""
import math
from .models import supports_openai_options

DEFAULTS = {
    "initial_prompt": None, "carry_initial_prompt": False,
    "word_timestamps": False, "condition_on_previous_text": True,
    "hallucination_silence_threshold": None, "clip_timestamps": "0",
    "beam_size": None, "best_of": None, "temperature": None,
    "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


THRESHOLDS = {"compression_ratio_threshold", "logprob_threshold", "no_speech_threshold"}
PRESETS = {
    "default": {},
    "official-cli": {"beam_size": 5, "best_of": 5,
                     "temperature": (0, .2, .4, .6, .8, 1)},
}


def optional_float(value):
    """CLI/UI spelling 'none' disables a threshold."""
    return None if isinstance(value, str) and value.strip().lower() == "none" else float(value)


def numbers(value, label):
    try:
        values = [float(x.strip()) for x in value.split(",")] if isinstance(value, str) else (
            [float(value)] if isinstance(value, (float, int)) else [float(x) for x in value])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must contain numbers separated by commas.") from exc
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{label} must contain finite, non-negative numbers.")
    return values


def decoding_options(model_name, preset="default", **values):
    unknown = set(values) - set(DEFAULTS) - {"verbose"}
    if unknown:
        raise ValueError(f"Unknown decoding options: {', '.join(sorted(unknown))}")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("Preset must be default or official-cli.")
    if not supports_openai_options(model_name) and preset != "default":
        raise ValueError("Decoding presets are only supported by OpenAI Whisper.")
    options = dict(DEFAULTS, **PRESETS[preset])
    options.update(values)
    for key in ("word_timestamps", "carry_initial_prompt", "condition_on_previous_text"):
        if type(options[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    if options["initial_prompt"] is not None and not isinstance(options["initial_prompt"], str):
        raise ValueError("Initial prompt must be a string or null.")
    options["initial_prompt"] = (options["initial_prompt"] or "").strip() or None
    if options["temperature"] in (None, ""):
        options["temperature"] = None  # Preserve upstream temperature fallback.
    else:
        options["temperature"] = tuple(numbers(options["temperature"], "Temperature"))
        if any(x > 1 for x in options["temperature"]):
            raise ValueError("Temperature must be between 0 and 1.")
    clip = numbers(options["clip_timestamps"] or "0", "Clip timestamps")
    if any(b <= a for a, b in zip(clip, clip[1:])):
        raise ValueError("Clip timestamps must increase: start,end,start,end,...")
    options["clip_timestamps"] = "0" if clip == [0.0] else clip
    for key in ("beam_size", "best_of"):
        value = options[key]
        if value is not None:
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 1 or int(float(value)) != float(value):
                raise ValueError(f"{key} must be a positive integer.")
            options[key] = int(value)
    for key in THRESHOLDS:
        value = options[key]
        if value is not None:
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{key} must be finite or null.")
            value = float(value)
            if key == "compression_ratio_threshold" and value <= 0:
                raise ValueError("Compression ratio threshold must be positive or null.")
            if key == "no_speech_threshold" and not 0 <= value <= 1:
                raise ValueError("No speech threshold must be between 0 and 1 or null.")
            options[key] = value
    beam = options["beam_size"]
    if beam is not None and options["temperature"] and 0 not in options["temperature"]:
        raise ValueError("Beam search requires temperature 0 in the temperature sequence.")
    threshold = options["hallucination_silence_threshold"]
    if threshold is not None:
        if not math.isfinite(float(threshold)) or float(threshold) <= 0:
            raise ValueError("Silence threshold must be a positive number of seconds.")
        if not options["word_timestamps"]:
            raise ValueError("Silence/hallucination filtering requires word timestamps.")
        options["hallucination_silence_threshold"] = float(threshold)
    if options["carry_initial_prompt"] and not options["initial_prompt"]:
        raise ValueError("Repeat prompt requires an initial prompt.")
    if not supports_openai_options(model_name):
        changed = [key for key, default in DEFAULTS.items() if options[key] != default]
        if changed:
            raise ValueError("These options are only supported by OpenAI Whisper: " + ", ".join(changed))
        return {"verbose": options.get("verbose", False)}
    return {key: value for key, value in options.items() if value is not None or key in THRESHOLDS}
