"""Validate decoding options identically for CLI, UI and direct callers."""
import math

DEFAULTS = {
    "initial_prompt": None, "carry_initial_prompt": False,
    "word_timestamps": False, "condition_on_previous_text": True,
    "hallucination_silence_threshold": None, "clip_timestamps": "0",
    "beam_size": None, "temperature": None,
}


def numbers(value, label):
    try:
        values = [float(x.strip()) for x in value.split(",")] if isinstance(value, str) else (
            [float(value)] if isinstance(value, (float, int)) else [float(x) for x in value])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must contain numbers separated by commas.") from exc
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{label} must contain finite, non-negative numbers.")
    return values


def decoding_options(model_name, **values):
    unknown = set(values) - set(DEFAULTS) - {"verbose"}
    if unknown:
        raise ValueError(f"Unknown decoding options: {', '.join(sorted(unknown))}")
    options = dict(DEFAULTS, **values)
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
    beam = options["beam_size"]
    if beam is not None:
        if not math.isfinite(float(beam)) or float(beam) < 1 or int(beam) != float(beam):
            raise ValueError("Beam size must be a positive integer.")
        options["beam_size"] = int(beam)
        if options["temperature"] and 0 not in options["temperature"]:
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
    if model_name == "Thai_Thonburian":
        changed = [key for key, default in DEFAULTS.items() if options[key] != default]
        if changed:
            raise ValueError("These options are only supported by OpenAI Whisper: " + ", ".join(changed))
        return {"verbose": options.get("verbose", False)}
    return {key: value for key, value in options.items() if value is not None}
