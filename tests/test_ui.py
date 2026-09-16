"""UI modes and exports use a mocked engine; no audio/model downloads."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from yt_whisper.audio import AudioSource
from yt_whisper.storage import Storage


@pytest.fixture
def ui(tmp_path, monkeypatch):
    from yt_whisper.ui import build_demo
    storage = Storage(tmp_path / "data")
    monkeypatch.setattr("yt_whisper.ui.configure_storage", lambda: storage)
    monkeypatch.setattr("yt_whisper.config.load_config", lambda s: {"model": "base", "language": "en"})
    calls = []
    class Engine:
        def __init__(self, storage):
            pass
        def transcribe(self, path, model, language, task, device, **options):
            calls.append((model, language, task, device, options))
            return {"text": "First second", "model": model, "language": language,
                    "device": "cpu", "segments": [{"start": 0, "end": 2, "text": "First second"}]}
    monkeypatch.setattr("yt_whisper.engine.Transcriber", Engine)
    @contextmanager
    def audio(source, storage):
        yield AudioSource(Path("sample.wav"), "sample", "id", str(source))
    monkeypatch.setattr("yt_whisper.audio.prepare_audio", audio)
    demo = build_demo()
    callbacks = {f.fn.__name__: f.fn for f in demo.fns.values()}
    yield demo, callbacks, storage, calls
    demo.close()


def request():
    return ["sample.wav", "base", "en", "transcribe", "auto",
            "", False, False, True, None, "0", None, "", ["txt", "json", "srt", "vtt"],
            "default", None, "", "", "", False, None, None, None]


def test_simple_ignores_hidden_advanced_settings(ui):
    demo, callbacks, storage, calls = ui
    args = request()
    args[3:5] = ["translate", "cuda"]
    args[5] = "hidden prompt"
    args[6] = True
    args[9] = 2
    args[10] = "5,10"
    args[13] = []
    args[14] = "official-cli"
    runs = list(callbacks["transcribe_interactive"](*args, "Simple", 42, "../escape"))
    assert runs[0] == ("", [], "Processing...")
    text, files, status = runs[-1]
    assert text == "First second" and len(files) == 4
    assert status.startswith("Completed in ") and "seconds" in status
    assert calls[0][2:4] == ("transcribe", "auto")
    opts = calls[0][-1]
    assert not opts["word_timestamps"] and "initial_prompt" not in opts and "beam_size" not in opts
    assert opts["clip_timestamps"] == "0"
    assert all(Path(file).parent.parent == storage.outputs for file in files)


def test_advanced_settings_formats_and_subfolder(ui):
    _, callbacks, storage, calls = ui
    args = request()
    args[4] = "cpu"
    args[5] = "Names"
    args[13] = ["srt", "jsonl"]
    args[14] = "official-cli"
    args[12] = "0"
    text, files, status = list(callbacks["transcribe_interactive"](*args, "Advanced", 8, "session-a"))[-1]
    assert status.startswith("Completed in ")
    assert {Path(file).suffix for file in files} == {".srt", ".jsonl"}
    assert all(Path(file).is_relative_to(storage.outputs / "session-a") for file in files)
    assert calls[0][3] == "cpu"
    assert calls[0][-1]["initial_prompt"] == "Names" and calls[0][-1]["beam_size"] == 5


@pytest.mark.parametrize("line_length,folder", [(0, "../escape"), (-1, ""), (1.5, ""), (float("inf"), "")])
def test_invalid_export_settings_fail_before_inference(ui, line_length, folder):
    _, callbacks, _, calls = ui
    final = list(callbacks["transcribe_interactive"](*request(), "Advanced", line_length, folder))[-1]
    assert final[:2] == ("", []) and final[2].startswith("Failed after ")
    assert not calls


def test_legacy_api_still_returns_two_outputs(ui):
    _, callbacks, _, calls = ui
    response = callbacks["transcribe"](*request())
    assert len(response) == 2 and response[0] == "First second"
    assert len(response[1]) == 4 and len(calls) == 1


def test_copy_control_default_mode_and_thai_visibility(ui):
    demo, callbacks, *_ = ui
    props = [c["props"] for c in demo.config["components"]]
    transcript = next(p for p in props if p.get("label") == "Transcription")
    assert transcript["buttons"] == ["copy"] and transcript["interactive"] is True
    assert next(p for p in props if p.get("label") == "Mode")["value"] == "Simple"
    assert all(not update["visible"] for update in callbacks["show_mode"]("Simple", "base"))
    assert all(update["visible"] for update in callbacks["show_mode"]("Advanced", "base"))
    for model in ("thonburian-medium", "thonburian-large-v3", "thonburian-distill-large-v3"):
        updates = callbacks["show_mode"]("Advanced", model)
        assert not updates[2]["visible"] and updates[0]["visible"] and updates[3]["visible"]
