import json
import pytest
from yt_whisper.options import decoding_options, timestamp_seconds
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


@pytest.mark.parametrize("value,seconds", [
    ("10:00", 600), ("12:30", 750), ("01:10:00", 4200),
    ("90:00", 5400), (" 1:02.5 ", 62.5), ("00:00:00.125", .125),
    ("750", 750), (2.5, 2.5), ("0:00", 0),
])
def test_clock_time_and_seconds(value, seconds):
    assert timestamp_seconds(value) == seconds


@pytest.mark.parametrize("value", [
    "10:60", "1:60:00", "1:02:99", "-1:00", "1.5:00", "1:2:3:4",
    "10:", "ten minutes", "nan", "inf", -1, True,
])
def test_invalid_clock_time(value):
    with pytest.raises(ValueError, match="MM:SS"):
        timestamp_seconds(value)


def test_clock_ranges_normalize_and_survive_engine_revalidation():
    options = decoding_options("base", clip_timestamps="10:00,12:30,00:30:00,1860.5")
    assert options["clip_timestamps"] == [600, 750, 1800, 1860.5]
    assert decoding_options("base", **options) == options
    assert decoding_options("base", clip_timestamps="10:00")["clip_timestamps"] == [600]
    assert decoding_options("base", clip_timestamps="0:00")["clip_timestamps"] == "0"
    for invalid in ("12:30,10:00", "10:00,600", "10:00,,12:30", "10:00,10:60"):
        with pytest.raises(ValueError):
            decoding_options("base", clip_timestamps=invalid)
    thai = decoding_options("thonburian-medium", clip_timestamps="10:00,12:30")
    assert thai == {"verbose": False, "clip_timestamps": [600, 750]}
    assert decoding_options("thonburian-medium", **thai) == thai


@pytest.mark.parametrize("model", ["base", "Thai_Thonburian", "thonburian-medium",
                                    "thonburian-large-v3", "thonburian-distill-large-v3"])
def test_cli_passes_clock_ranges_to_engine(monkeypatch, model):
    from contextlib import contextmanager
    from unittest.mock import Mock
    from yt_whisper.cli import main
    @contextmanager
    def audio(*args):
        yield AudioSource(Path("sample.wav"), "sample", "id", "sample.wav")
    engine = Mock()
    monkeypatch.setattr("yt_whisper.audio.prepare_audio", audio)
    monkeypatch.setattr("yt_whisper.engine.Transcriber", Mock(return_value=engine))
    monkeypatch.setattr("yt_whisper.results.save_result", Mock(return_value=[]))
    assert main(["sample.wav", "--model", model, "--clip-timestamps", "10:00,12:30"]) == 0
    assert engine.transcribe.call_args.kwargs["clip_timestamps"] == [600, 750]


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


def test_preset_overrides_survive_engine_revalidation():
    opts = decoding_options("base", preset="official-cli", beam_size=3,
                            no_speech_threshold=None, temperature=0)
    assert opts["beam_size"] == 3 and opts["best_of"] == 5
    assert opts["temperature"] == (0,)
    assert opts["no_speech_threshold"] is None
    assert decoding_options("base", **opts) == opts
    fallback = decoding_options("base", preset="official-cli")
    assert fallback["temperature"] == (0, .2, .4, .6, .8, 1)
    assert fallback["beam_size"] == fallback["best_of"] == 5
    assert "best_of" not in decoding_options("base")


@pytest.mark.parametrize("kwargs", [
    {"best_of": 0}, {"best_of": True}, {"best_of": 1.5},
    {"no_speech_threshold": 1.1}, {"no_speech_threshold": True},
    {"logprob_threshold": float("nan")}, {"compression_ratio_threshold": 0},
    {"preset": "missing"}, {"preset": []},
])
def test_new_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        decoding_options("base", **kwargs)


def test_thai_rejects_preset_and_disabled_threshold():
    for opts in ({"preset": "official-cli"}, {"no_speech_threshold": None}):
        with pytest.raises(ValueError, match="OpenAI"):
            decoding_options("Thai_Thonburian", **opts)


def test_word_subtitles_preserve_json_and_original_timing(tmp_path):
    import copy
    from yt_whisper.results import subtitle_options
    result = {"text": "One two three", "segments": [{
        "start": 0, "end": 4, "text": "One two three",
        "words": [{"word": " One", "start": .5, "end": 1},
                  {"word": " two", "start": 1.2, "end": 2},
                  {"word": " three", "start": 2.2, "end": 3}],
    }]}
    original = copy.deepcopy(result)
    files = save_result(result, AudioSource(Path("a"), "title", "id", "a"), tmp_path,
                        ("srt", "vtt", "json"),
                        subtitles=subtitle_options(True, True, max_words_per_line=1))
    srt, vtt = [Path(p).read_text(encoding="utf-8") for p in files[:2]]
    assert "00:00:00,500 --> 00:00:01,000" in srt
    assert "<u>One</u>" in srt and "<u>three</u>" in vtt
    assert srt.count("-->") == 3
    assert json.loads(Path(files[2]).read_text(encoding="utf-8"))["segments"] == original["segments"]
    assert result == original


@pytest.mark.parametrize("kwargs", [
    {"word_timestamps": False, "highlight_words": True},
    {"word_timestamps": True, "max_line_count": 2},
    {"word_timestamps": True, "max_line_width": 20, "max_words_per_line": 5},
    {"word_timestamps": True, "max_line_width": float("inf")},
    {"word_timestamps": True, "max_line_width": 20, "line_length": 20},
])
def test_invalid_subtitle_settings(kwargs):
    from yt_whisper.results import subtitle_options
    with pytest.raises(ValueError):
        subtitle_options(**kwargs)


def test_cli_new_settings_fail_before_fetch(monkeypatch):
    from unittest.mock import Mock
    from yt_whisper.cli import main
    fetch = Mock()
    monkeypatch.setattr("yt_whisper.audio.prepare_audio", fetch)
    for flags in (["--highlight-words"], ["--no-speech-threshold", "2"],
                  ["--clip-timestamps", "10:60,12:30"], ["--clip-timestamps", "12:30,10:00"]):
        with pytest.raises(SystemExit) as exc:
            main(["https://example.org/video"] + flags)
        assert exc.value.code == 1
    fetch.assert_not_called()


def test_word_subtitles_allow_empty_silence_segments(tmp_path):
    result = {"text": "", "segments": [{"start": 0, "end": 1, "text": "", "words": []}]}
    files = save_result(result, AudioSource(Path("a"), "title", "id", "a"), tmp_path,
                        ("srt",), subtitles={"highlight_words": True})
    assert Path(files[0]).read_text(encoding="utf-8") == ""
