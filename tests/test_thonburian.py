"""Thonburian contracts without weights, network calls or a real GPU."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys
import weakref

import numpy as np
import pytest

from yt_whisper.audio import AudioSource
from yt_whisper.config import load_config
from yt_whisper.engine import Transcriber, validate_selection
from yt_whisper.models import (
    THONBURIAN_MODELS, canonical_model_id, custom_model_spec, model_choices,
)
from yt_whisper.options import decoding_options
from yt_whisper.results import save_result
from yt_whisper.storage import Storage


EXPECTED = {
    "thonburian-medium": "biodatlab/whisper-th-medium-combined",
    "thonburian-large-v3": "biodatlab/whisper-th-large-v3-combined",
    "thonburian-distill-large-v3": "biodatlab/distill-whisper-th-large-v3",
}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.delenv("YTW_THAI_MODEL", raising=False)
    for spec in THONBURIAN_MODELS:
        monkeypatch.delenv(spec.environment_variable, raising=False)
    return Storage(tmp_path / "data")


@pytest.fixture
def hf_backend(monkeypatch):
    import torch
    import whisper
    calls = SimpleNamespace(processors=[], models=[], pipelines=[], inference=[],
                            raw={"text": "sample", "chunks": []}, before_load=lambda: None)

    def processor(source, **kwargs):
        calls.processors.append((source, kwargs))
        return SimpleNamespace(tokenizer=object(), feature_extractor=object())

    def model(source, **kwargs):
        calls.before_load()
        calls.models.append((source, kwargs))
        return SimpleNamespace(config=SimpleNamespace(_commit_hash="fixture-revision"))

    class Pipeline:
        def __init__(self, task, **kwargs):
            self.model = kwargs["model"]
            calls.pipelines.append((task, {k: v for k, v in kwargs.items()
                                          if k not in ("model", "tokenizer", "feature_extractor")}))

        def __call__(self, waveform, **kwargs):
            calls.inference.append(kwargs)
            return copy.deepcopy(calls.raw)

    fake = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=processor),
                           WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=model),
                           pipeline=Pipeline)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(whisper, "load_audio", lambda path: np.zeros(16000 * 2, dtype=np.float32))
    monkeypatch.setattr(whisper, "load_model", Mock(side_effect=AssertionError("Unexpected OpenAI load")))
    return calls


@pytest.mark.parametrize("identifier,repo", EXPECTED.items())
def test_registry_resolves_exact_checkpoints(identifier, repo):
    spec = custom_model_spec(identifier)
    assert spec.hf_repo == repo
    assert custom_model_spec(repo) == spec
    assert spec.family == "whisper" and spec.backend == "transformers-whisper"
    assert spec.thai_only and not spec.openai_decoding_options
    assert spec.tasks == ("transcribe",)


def test_legacy_config_resolves_without_rewriting_file(storage):
    storage.config.parent.mkdir(parents=True)
    old = '{"model": "Thai_Thonburian", "language": "Auto"}'
    storage.config.write_text(old, encoding="utf-8")
    assert load_config(storage) == {"model": "thonburian-medium", "language": "Auto"}
    assert storage.config.read_text(encoding="utf-8") == old
    assert canonical_model_id("Thai_Thonburian") == "thonburian-medium"


@pytest.mark.parametrize("spec", THONBURIAN_MODELS, ids=lambda s: s.id)
@pytest.mark.parametrize("language", [None, "Auto", "th", "Thai"])
def test_auto_and_thai_generation(storage, hf_backend, spec, language):
    result = Transcriber(storage).transcribe("sample.wav", spec.id, language, device="cpu")
    assert hf_backend.models[0][0] == spec.hf_repo
    assert hf_backend.processors[0][0] == spec.hf_repo
    assert hf_backend.inference == [{"generate_kwargs": {"language": "th", "task": "transcribe"},
                                     "return_timestamps": True}]
    assert result["model"] == spec.id and result["language"] == "th"
    assert result["model_source"] == result["model_hf_repo"] == spec.hf_repo
    assert result["model_revision"] == "fixture-revision"


@pytest.mark.parametrize("spec", THONBURIAN_MODELS, ids=lambda s: s.id)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_loader_precision_safetensors_and_cache(storage, hf_backend, monkeypatch, spec, device):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    engine = Transcriber(storage)
    first = engine._load(spec.id, device)
    assert engine._load(spec.hf_repo, device) is first
    assert engine._key == (spec.id, device, spec.hf_repo)
    assert len(hf_backend.models) == 1
    kwargs = hf_backend.models[0][1]
    assert kwargs["dtype"] == (torch.float16 if device == "cuda" else torch.float32)
    assert kwargs["use_safetensors"] is True and kwargs["local_files_only"] is False
    assert kwargs["cache_dir"] == os.environ["HF_HUB_CACHE"]
    assert hf_backend.pipelines[0][1] == {"device": 0 if device == "cuda" else -1,
                                         "chunk_length_s": 30, "batch_size": 1}


def test_switching_releases_old_pipeline_before_new_load(storage, hf_backend, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    engine = Transcriber(storage)
    first = engine._load("Thai_Thonburian", "cuda")
    assert engine._load("thonburian-medium", "cuda") is first
    previous = weakref.ref(first)
    del first

    def check_released():
        assert previous() is None
        assert engine._model is None and engine._key is None

    hf_backend.before_load = check_released
    keys = []
    for spec in THONBURIAN_MODELS[1:]:
        loaded = engine._load(spec.id, "cuda")
        keys.append(engine._key)
        previous = weakref.ref(loaded)
        del loaded
    assert len(set(keys)) == 2
    assert [source for source, _ in hf_backend.models] == list(EXPECTED.values())
    assert torch.cuda.empty_cache.call_count == 3


def test_failed_load_leaves_cache_empty_and_can_retry(storage, hf_backend):
    engine = Transcriber(storage)
    engine._load("thonburian-medium", "cpu")
    hf_backend.before_load = Mock(side_effect=RuntimeError("load failed"))
    with pytest.raises(RuntimeError, match="load failed"):
        engine._load("thonburian-large-v3", "cpu")
    assert engine._model is None and engine._key is None
    hf_backend.before_load = lambda: None
    assert engine._load("thonburian-large-v3", "cpu") is not None


@pytest.mark.parametrize("spec", THONBURIAN_MODELS, ids=lambda s: s.id)
def test_variant_local_paths_and_offline_loading(storage, hf_backend, spec):
    for item in THONBURIAN_MODELS:
        directory = storage.root / "models" / item.local_directory
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")
    engine = Transcriber(storage)
    result = engine.transcribe("sample.wav", spec.id, device="cpu")
    expected = str((storage.root / "models" / spec.local_directory).resolve())
    assert hf_backend.models[0][0] == result["model_source"] == expected
    assert result["model_hf_repo"] == spec.hf_repo
    assert hf_backend.models[0][1]["local_files_only"] is True
    assert hf_backend.processors[0][1]["local_files_only"] is True


def test_legacy_override_is_medium_only_and_source_change_reloads(storage, hf_backend, tmp_path, monkeypatch):
    path = tmp_path / "custom-medium"
    path.mkdir()
    monkeypatch.setenv("YTW_THAI_MODEL", str(path))
    medium, large, distilled = THONBURIAN_MODELS
    assert storage.thonburian_source(large) == large.hf_repo
    assert storage.thonburian_source(distilled) == distilled.hf_repo
    engine = Transcriber(storage)
    engine._load(medium.id, "cpu")
    assert hf_backend.models[-1][0] == str(path.resolve())
    another = tmp_path / "another-medium"
    another.mkdir()
    monkeypatch.setenv(medium.environment_variable, str(another))
    engine._load(medium.id, "cpu")
    assert len(hf_backend.models) == 2
    assert hf_backend.models[-1][0] == str(another.resolve())


@pytest.mark.parametrize("spec", THONBURIAN_MODELS, ids=lambda s: s.id)
def test_missing_optional_dependency_has_install_hint(storage, monkeypatch, spec):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match=r'pip install .*\[thai\]'):
        Transcriber(storage).transcribe("sample.wav", spec.id, device="cpu")


@pytest.mark.parametrize("spec", THONBURIAN_MODELS, ids=lambda s: s.id)
@pytest.mark.parametrize("kwargs", [{"task": "translate"}, {"language": "en"}, {"language": "ja"}])
def test_unsupported_selection_fails_before_load(storage, monkeypatch, spec, kwargs):
    engine = Transcriber(storage)
    loader = Mock()
    monkeypatch.setattr(engine, "_load", loader)
    with pytest.raises(ValueError, match="Thai"):
        engine.transcribe("sample.wav", spec.id, **kwargs)
    loader.assert_not_called()


@pytest.mark.parametrize("identifier", [*EXPECTED, "Thai_Thonburian", *EXPECTED.values()])
def test_openai_options_remain_unavailable(identifier):
    assert decoding_options(identifier) == {"verbose": False}
    for opts in ({"word_timestamps": True}, {"preset": "official-cli"}, {"beam_size": 5}):
        with pytest.raises(ValueError, match="OpenAI"):
            decoding_options(identifier, **opts)


@pytest.mark.parametrize("spec", THONBURIAN_MODELS, ids=lambda s: s.id)
@pytest.mark.parametrize("chunks", [[], None, [{"timestamp": (0, None), "text": "sample"}],
                                    [{"timestamp": None, "text": ""}]])
def test_missing_timestamps_and_empty_chunks_export_text(storage, hf_backend, spec, chunks):
    hf_backend.raw = {"text": "sample", "chunks": chunks}
    result = Transcriber(storage).transcribe("sample.wav", spec.id, device="cpu")
    assert result["segments"] == [{"start": 0, "end": 2, "text": "sample"}]
    original = copy.deepcopy(result)
    audio = AudioSource(Path("sample.wav"), "same title", "source-id", "sample.wav")
    first = save_result(result, audio, storage.outputs)
    second = save_result(result, audio, storage.outputs)
    assert Path(first[0]).parent != Path(second[0]).parent
    for filename in first:
        assert "sample" in Path(filename).read_text(encoding="utf-8")
    payload = json.loads(Path(first[1]).read_text(encoding="utf-8"))
    assert payload["source_id"] == "source-id" and payload["model_hf_repo"] == spec.hf_repo
    assert result == original


def test_ui_choices_legacy_config_and_optional_import(storage, monkeypatch):
    from contextlib import contextmanager
    from yt_whisper.ui import build_demo
    @contextmanager
    def audio(*args):
        yield AudioSource(Path("sample.wav"), "sample", "id", "sample.wav")
    monkeypatch.setattr("yt_whisper.audio.prepare_audio", audio)
    storage.config.parent.mkdir(parents=True)
    storage.config.write_text('{"model":"Thai_Thonburian","language":"Auto"}', encoding="utf-8")
    monkeypatch.setattr("yt_whisper.ui.configure_storage", lambda: storage)
    monkeypatch.setitem(sys.modules, "transformers", None)
    demo = build_demo()
    try:
        components = demo.config["components"]
        selector = next(c["props"] for c in components if c["props"].get("label") == "Model")
        assert selector["value"] == "thonburian-medium"
        assert selector["allow_custom_value"] is True  # legacy Gradio API arguments
        assert dict(selector["choices"]) == dict(model_choices())
        advanced = next(c["props"] for c in components if
                        c["props"].get("label", "").startswith("Transcription options"))
        assert advanced["visible"] is False
        callback = next(f.fn for f in demo.fns.values() if f.fn.__name__ == "transcribe")
        with pytest.raises(Exception, match=r'pip install .*\[thai\]'):
            # Validate a legacy API model value without downloading/decoding audio.
            callback("sample.wav", "Thai_Thonburian", "Auto", "transcribe", "cpu")
    finally:
        demo.close()


def test_official_choices_and_language_behavior_unchanged():
    import whisper
    assert [value for _, value in model_choices() if not custom_model_spec(value)] == whisper.available_models()
    assert validate_selection("base", "Auto") == ("base", None)
    assert validate_selection("base.en", "th") == ("base.en", "en")
    assert validate_selection("large-v3", "ja", "translate") == ("large-v3", "ja")


@pytest.mark.parametrize("model", [*EXPECTED, "large-v3", "turbo"])
def test_benchmark_model_metadata_cer_and_cpu_fallback(tmp_path, monkeypatch, model):
    import runpy
    import torch
    import whisper
    from yt_whisper.storage import configure_storage
    # Synthetic Thai characters verify whitespace-only CER without external audio.
    text = "\u0e01 \u0e02"
    spec = custom_model_spec(model)
    result = {"model": model, "device": "cpu", "language": "th", "text": text,
              "segments": [{"start": 0, "end": 2, "text": text}]}
    if spec:
        result.update(model_hf_repo=spec.hf_repo, model_source=spec.hf_repo,
                      model_backend=spec.backend, model_revision="fixture-revision")
    calls = []
    class Engine:
        def __init__(self, storage):
            pass
        def _load(self, name, device):
            calls.append((name, device))
        def transcribe(self, audio, name, language, **kwargs):
            assert kwargs["device"] == "cpu" and language == "th"
            return result
    monkeypatch.setattr("yt_whisper.engine.Transcriber", Engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(whisper, "load_audio", lambda path: np.zeros(32000))
    original_storage = configure_storage()
    monkeypatch.setattr("yt_whisper.storage.configure_storage", lambda: original_storage)
    audio, reference, output = [tmp_path / name for name in ("audio.wav", "reference.txt", "run.json")]
    audio.write_bytes(b"mock audio")
    reference.write_text("\u0e01\u0e02", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--audio", str(audio), "--model", model,
                        "--device", "cuda", "--language", "th", "--reference", str(reference),
                        "--output", str(output)])
    with pytest.warns(UserWarning, match="CPU"):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/benchmark.py"), run_name="__main__")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [(model, "cpu")]
    assert report["model"] == model and report["audio_seconds"] == 2
    assert report["device"] == "cpu" and report["requested_device"] == "cuda"
    assert report["real_time_factor"] == report["transcribe_seconds"] / 2
    assert report["peak_cuda_allocated_gib"] is None
    assert report["character_error_rate"] == 0
    assert len(report["outputs"]) == 6 and all(Path(p).is_file() for p in report["outputs"])
    if spec:
        assert report["model_hf_repo"] == spec.hf_repo and report["model_revision"] == "fixture-revision"


@pytest.mark.parametrize("model", [*EXPECTED, "Thai_Thonburian"])
def test_cli_rejects_thai_translation_before_fetch(model, monkeypatch):
    from yt_whisper.cli import main
    fetch = Mock()
    monkeypatch.setattr("yt_whisper.audio.prepare_audio", fetch)
    with pytest.raises(SystemExit) as exc:
        main(["https://example.org/video", "--model", model, "--task", "translate"])
    assert exc.value.code == 1
    fetch.assert_not_called()
