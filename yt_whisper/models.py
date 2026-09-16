"""Lightweight registry of Thai-specialized Transformers Whisper checkpoints.

Official models still come from whisper.available_models(). No Transformers
import or model download occurs while listing or resolving these models.
"""
from dataclasses import dataclass

LEGACY_THAI_MODEL = "Thai_Thonburian"


@dataclass(frozen=True)
class WhisperModelSpec:
    id: str
    display_name: str
    hf_repo: str
    local_directory: str
    environment_variable: str
    aliases: tuple[str, ...] = ()
    family: str = "whisper"
    backend: str = "transformers-whisper"
    thai_only: bool = True
    openai_decoding_options: bool = False
    tasks: tuple[str, ...] = ("transcribe",)


THONBURIAN_MODELS = (
    WhisperModelSpec(
        "thonburian-medium", "Thonburian Medium",
        "biodatlab/whisper-th-medium-combined", "thonburian",
        "YTW_THONBURIAN_MEDIUM_MODEL", (LEGACY_THAI_MODEL,),
    ),
    WhisperModelSpec(
        "thonburian-large-v3", "Thonburian Large-v3",
        "biodatlab/whisper-th-large-v3-combined", "thonburian-large-v3",
        "YTW_THONBURIAN_LARGE_V3_MODEL",
    ),
    WhisperModelSpec(
        "thonburian-distill-large-v3", "Thonburian Distilled Large-v3",
        "biodatlab/distill-whisper-th-large-v3", "thonburian-distill-large-v3",
        "YTW_THONBURIAN_DISTILL_LARGE_V3_MODEL",
    ),
)
_CUSTOM_MODELS = {
    name: spec for spec in THONBURIAN_MODELS
    for name in (spec.id, spec.hf_repo, *spec.aliases)
}


def custom_model_spec(name):
    return _CUSTOM_MODELS.get(name)


def canonical_model_id(name):
    spec = custom_model_spec(name)
    return spec.id if spec else name


def supports_openai_options(name):
    spec = custom_model_spec(name)
    return spec is None or spec.openai_decoding_options


def model_ids():
    import whisper
    return whisper.available_models() + list(_CUSTOM_MODELS)


def model_choices():
    import whisper
    return [(f"OpenAI Whisper / {name}", name) for name in whisper.available_models()] + [
        (spec.display_name, spec.id) for spec in THONBURIAN_MODELS
    ]
