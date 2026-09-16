import json
import pytest
from yt_whisper.options import decoding_options
from yt_whisper.results import save_result
from yt_whisper.audio import AudioSource
from pathlib import Path


def test_options_preserve_defaults():
    opts = decoding_options("base")
    assert "temperature" not in opts
    assert "beam_size" not in opts
    assert opts["condition_on_previous_text"] is True


@pytest.mark.parametrize("kwargs", [
    {"temperature": "nan"}, {"temperature": "1.2"}, {"beam_size": 0},
    {"beam_size": 1.5}, {"beam_size": 5, "temperature": "0.4"},
    {"clip_timestamps": "5,2"}, {"clip_timestamps": "1,3,2,4"},
    {"hallucination_silence_threshold": 2},
    {"hallucination_silence_threshold": -1, "word_timestamps": True},
    {"carry_initial_prompt": True},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        decoding_options("base", **kwargs)


def test_options_parse_fallback_and_ranges():
    opts = decoding_options("base", temperature="0,0.2,0.4", beam_size=3,
                            clip_timestamps="1,3,5", word_timestamps=True,
                            hallucination_silence_threshold=2)
    assert opts["temperature"] == (0, .2, .4)
    assert opts["clip_timestamps"] == [1, 3, 5]


def test_thai_defaults_allowed_but_explicit_options_rejected():
    assert decoding_options("Thai_Thonburian") == {"verbose": False}
    with pytest.raises(ValueError, match="OpenAI"):
        decoding_options("Thai_Thonburian", condition_on_previous_text=False)


def test_tsv_and_jsonl_preserve_segments(tmp_path):
    segments = [{"start": .25, "end": 1.5, "text": "ไทย\tสอง\nบรรทัด", "words": [{"word": "ไทย", "start": .25, "end": .5}]}]
    files = save_result({"text": "ไทย", "segments": segments}, AudioSource(Path("a"), "title", "id", "a"),
                        tmp_path, ("tsv", "jsonl"))
    tsv = Path(files[0]).read_text(encoding="utf-8").splitlines()
    assert tsv == ["start\tend\ttext", "250\t1500\tไทย สอง บรรทัด"]
    assert json.loads(Path(files[1]).read_text(encoding="utf-8")) == segments[0]


def test_invalid_format_creates_no_output(tmp_path):
    with pytest.raises(ValueError):
        save_result({}, None, tmp_path / "out", ("bad",))
    assert not (tmp_path / "out").exists()


def test_cli_rejects_invalid_options_before_fetch(monkeypatch):
    from unittest.mock import Mock
    from yt_whisper.cli import main
    fetch = Mock()
    monkeypatch.setattr("yt_whisper.audio.prepare_audio", fetch)
    with pytest.raises(SystemExit) as exc:
        main(["https://example.org/video", "--hallucination-silence-threshold", "2"])
    assert exc.value.code == 1
    fetch.assert_not_called()
