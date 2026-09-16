import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from yt_whisper.audio import AudioSource, prepare_audio
from yt_whisper.config import load_config, save_config
from yt_whisper.engine import Transcriber, normalize_language
from yt_whisper.results import save_result
from yt_whisper.storage import Storage


@pytest.fixture
def storage(tmp_path):
    root = Storage(tmp_path / "data")
    root.temp.mkdir(parents=True)
    root.models.mkdir(parents=True)
    root.config.parent.mkdir(parents=True)
    return root


@pytest.mark.parametrize("value", [None, "", " ", "Auto", "auto"])
def test_auto_language(value):
    assert normalize_language(value) is None


def test_language_names_and_codes():
    assert normalize_language("Thai") == normalize_language("th") == "th"
    with pytest.raises(ValueError):
        normalize_language("not-a-language")


def test_local_mp3_never_converted_or_removed(storage, tmp_path, monkeypatch):
    monkeypatch.setattr("yt_whisper.audio.require_ffmpeg", lambda: "ffmpeg")
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original audio")
    with prepare_audio(source, storage) as audio:
        assert audio.path == source
    assert source.read_bytes() == b"original audio"
    assert list(storage.temp.iterdir()) == []


def test_download_cleanup_on_failure(storage, monkeypatch):
    import yt_dlp
    monkeypatch.setattr("yt_whisper.audio.require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("yt_whisper.audio.downloader_options", lambda s, d: {"directory": d})
    folders = []
    class Downloader:
        def __init__(self, options):
            folders.append(Path(options["directory"]))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            (folders[-1] / "partial.wav").write_bytes(b"partial")
            raise RuntimeError("download interrupted")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Downloader)
    with pytest.raises(RuntimeError, match="interrupted"):
        with prepare_audio("https://example.org/video", storage):
            pass
    assert not folders[-1].exists()


def test_playlist_rejected_before_download(storage, monkeypatch):
    import yt_dlp
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)
    downloader.extract_info.return_value = {"_type": "playlist", "entries": []}
    monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda opts: downloader)
    monkeypatch.setattr("yt_whisper.audio.require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("yt_whisper.audio.downloader_options", lambda *args: {})
    with pytest.raises(ValueError, match="Playlists"):
        with prepare_audio("https://example.org/playlist", storage):
            pass
    downloader.extract_info.assert_called_once_with("https://example.org/playlist", download=False)
    downloader.process_ie_result.assert_not_called()
    assert list(storage.temp.iterdir()) == []


def test_config_none_and_invalid_json(storage):
    save_config(storage, "small", None)
    assert load_config(storage) == {"model": "small", "language": "Auto"}
    storage.config.write_text("broken", encoding="utf-8")
    with pytest.warns(UserWarning):
        assert load_config(storage)["model"] == "base"


def test_duplicate_titles_do_not_overwrite_and_exports_do_not_mutate(storage):
    audio = AudioSource(Path("a.wav"), "same title", "id", "https://example.org/a")
    result = {"text": " Thai text ", "segments": [{"text": " Thai text ", "start": 0, "end": 1}]}
    first = save_result(result, audio, storage.outputs, ("srt", "json"), line_length=5)
    second = save_result(result, audio, storage.outputs, ("json",))
    assert Path(first[0]).parent != Path(second[0]).parent
    payload = json.loads(Path(first[1]).read_text(encoding="utf-8"))
    assert payload["source_id"] == "id"
    assert payload["segments"][0]["text"] == " Thai text "
    assert result["segments"][0]["text"] == " Thai text "


def test_cpu_fallback_and_model_cache(storage, monkeypatch):
    import torch
    import whisper
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SimpleNamespace(transcribe=Mock(return_value={"text": "ok", "segments": []}))
    loader = Mock(return_value=model)
    monkeypatch.setattr(whisper, "load_model", loader)
    engine = Transcriber(storage)
    with pytest.warns(UserWarning, match="CPU"):
        result = engine.transcribe("a.wav", device="cuda", language=None)
    engine.transcribe("a.wav", device="cpu")
    assert result["device"] == "cpu"
    loader.assert_called_once_with("base", device="cpu", download_root=str(storage.models))
    assert model.transcribe.call_args.kwargs["fp16"] is False
    engine.transcribe("a.wav", model_name="small", device="cpu")
    assert loader.call_count == 2


def test_turbo_translation_rejected_before_load(storage, monkeypatch):
    engine = Transcriber(storage)
    monkeypatch.setattr(engine, "_load", Mock())
    with pytest.raises(ValueError, match="Turbo"):
        engine.transcribe("a.wav", model_name="turbo", task="translate")
    engine._load.assert_not_called()


def test_ui_build_without_optional_thai_import(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "transformers", None)
    from yt_whisper.ui import build_demo
    demo = build_demo()
    assert demo is not None
    demo.close()

def test_thai_missing_end_keeps_subtitle_text():
    from yt_whisper.engine import thai_segments
    segments = thai_segments([{"timestamp": (0.0, None), "text": "hello"}], 12.0, "hello")
    assert segments == [{"start": 0.0, "end": 12.0, "text": "hello"}]


def test_thai_no_chunks_still_exports_text():
    from yt_whisper.engine import thai_segments
    assert thai_segments([], 12.0, "hello") == [{"start": 0.0, "end": 12.0, "text": "hello"}]
