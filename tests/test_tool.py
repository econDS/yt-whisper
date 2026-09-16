import json
from pathlib import Path

import pytest

from yt_whisper import tool
from yt_whisper.storage import Storage


@pytest.fixture
def setup_tool(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "data")
    storage.temp.mkdir(parents=True)
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"original audio")
    request = {"schema_version": 1, "request_id": "caller-123", "audio_path": "source.wav",
               "model": "base", "language": "ja", "device": "cpu",
               "ranges": [{"id": "a", "start": 10, "end": 12}, {"id": "b", "start": 20, "end": 22}]}
    monkeypatch.setattr(tool, "configure_storage", lambda: storage)
    monkeypatch.setattr(tool, "probe_audio", lambda path: 60)
    clips, calls, instances = [], [], []
    def extract(source, dest, start, end):
        assert source == audio
        dest.write_bytes(b"temporary clip")
        clips.append(dest)
        return end - start
    monkeypatch.setattr(tool, "extract_range", extract)
    class Engine:
        def __init__(self, storage):
            instances.append(self)
        def transcribe(self, path, model, language, task, device, **options):
            assert path.is_file()
            calls.append((model, language, task, device, options))
            print("Library progress must not enter JSON stdout")
            return {"text": "日本語", "model": model, "device": device, "language": language,
                    "segments": [{"start": .2, "end": 1.5, "text": "日本語",
                                  "words": [{"start": .3, "end": .8, "word": "日本語", "probability": .8}]}]}
    monkeypatch.setattr("yt_whisper.engine.Transcriber", Engine)
    path = tmp_path / "request.json"
    def run(payload=None, output=None):
        path.write_text(json.dumps(payload if payload is not None else request), encoding="utf-8")
        args = ["--request", str(path)]
        if output:
            args += ["--response", str(output)]
        return tool.main(args)
    return request, run, clips, calls, instances, audio, storage, path


def test_multi_range_offsets_reuse_cleanup_and_json_stdout(setup_tool, capsys):
    request, run, clips, calls, instances, audio, storage, path = setup_tool
    response_file = path.parent / "results" / "response.json"
    assert run(output=response_file) == 0
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response == json.loads(response_file.read_text(encoding="utf-8"))
    assert "Library progress" in captured.err
    assert response["request_id"] == request["request_id"]
    assert response["status"] == "complete"
    assert response["source"]["audio_path"] == str(audio)
    assert response["timestamp_basis"] == "source_audio_seconds"
    assert len(instances) == 1 and len(calls) == 2
    assert response["results"][0]["segments"][0]["start"] == 10.2
    assert response["results"][1]["segments"][0]["words"][0]["end"] == 20.8
    assert all(not p.exists() for p in clips)
    assert not list(storage.temp.iterdir())
    assert audio.read_bytes() == b"original audio"


def test_second_range_failure_keeps_first_and_skips_remaining(setup_tool, monkeypatch, capsys):
    request, run, clips, calls, instances, audio, storage, path = setup_tool
    request["ranges"].append({"id": "c", "start": 30, "end": 32})
    from yt_whisper.engine import Transcriber
    original = Transcriber.transcribe
    def fail_second(self, *args, **kwargs):
        if calls:
            raise RuntimeError("simulated GPU failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Transcriber, "transcribe", fail_second)
    assert run() == 4
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "partial"
    assert [r["status"] for r in response["results"]] == ["complete", "failed", "not_run"]
    assert response["error"]["code"] == "inference_error"
    assert response["results"][0]["text"] == "日本語"
    assert all(not p.exists() for p in clips)


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"model": "not-a-model"}, {"language": "xyz"},
    {"options": {"word_timestamps": "false"}}, {"options": {"clip_timestamps": "1,2"}},
    {"options": {"hallucination_silence_threshold": 2}},
    {"options": {"beam_size": True}}, {"options": {"temperature": [True]}},
    {"ranges": []}, {"ranges": [{"id": "a", "start": True, "end": 2}]},
    {"ranges": [{"id": "a", "start": 2, "end": 1}]},
    {"ranges": [{"id": "a", "start": 1, "end": 2}, {"id": "a", "start": 3, "end": 4}]},
    {"ranges": [{"id": "a", "start": 1, "end": 100}]},
    {"unknown_option": 1}, {"model": "turbo", "task": "translate"},
])
def test_invalid_requests_fail_before_model_or_decode(setup_tool, capsys, change):
    request, run, clips, calls, instances, *_ = setup_tool
    request.update(change)
    assert run() == 2
    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_request"
    assert not instances and not clips and not calls


@pytest.mark.parametrize("which", ["audio", "request"])
def test_response_cannot_overwrite_inputs(setup_tool, capsys, which):
    request, run, clips, calls, instances, audio, storage, path = setup_tool
    output = audio if which == "audio" else path
    before = audio.read_bytes()
    assert run(output=output) == 2
    response = json.loads(capsys.readouterr().out)
    assert "overwrite" in response["error"]["message"]
    assert audio.read_bytes() == before
    assert json.loads(path.read_text()) == request
    assert not calls


def test_missing_audio_returns_input_error(setup_tool, monkeypatch, capsys):
    _, run, _, calls, *_ = setup_tool
    def missing(path):
        raise tool.ToolError("input_error", "missing audio")
    monkeypatch.setattr(tool, "probe_audio", missing)
    assert run() == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "input_error"
    assert not calls


def test_response_write_failure_keeps_result_on_stdout(setup_tool, monkeypatch, capsys):
    _, run, _, _, _, _, _, path = setup_tool
    def fail(*args):
        raise PermissionError("cannot write response")
    monkeypatch.setattr(tool, "atomic_response", fail)
    assert run(output=path.parent / "out.json") == 5
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "partial"
    assert len(result["results"]) == 2
    assert result["error"]["code"] == "response_write_error"


def test_missing_or_invalid_times_are_flagged_without_fabrication():
    result = tool.normalize_segments([{"start": None, "end": float("nan"), "text": "a"},
                                      {"start": -1, "end": 10, "text": "b"}], 100, 105)
    assert result[0]["start"] is None and result[0]["end"] is None
    assert result[0]["timing_status"] == "missing"
    assert result[1]["start"] == 99 and result[1]["end"] == 110
    assert result[1]["timing_status"] == "out_of_range"


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":2}', '{"value":NaN}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(tool.ToolError, match="Duplicate|Non-finite"):
        tool.strict_json(text)


def test_malformed_json_and_cli_error_emit_json(tmp_path, capsys):
    path = tmp_path / "request.json"
    path.write_text("{broken", encoding="utf-8")
    assert tool.main(["--request", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_request"
    assert tool.main([]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_non_finite_model_scores_do_not_break_json_output():
    result = tool.normalize_segments([{"start": 0, "end": 1, "text": "a",
                                       "avg_logprob": float("-inf"),
                                       "words": [{"start": 0, "end": 1, "word": "a", "probability": float("nan")}]}], 0, 1)
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded)[0]["diagnostics"]["avg_logprob"] is None
    assert result[0]["words"][0]["probability"] is None


def test_decode_failure_cleans_temp_and_never_calls_model(setup_tool, monkeypatch, capsys):
    request, run, clips, calls, instances, audio, storage, path = setup_tool
    def fail(source, dest, start, end):
        dest.write_bytes(b"partial")
        raise tool.ToolError("input_error", "decode failed")
    monkeypatch.setattr(tool, "extract_range", fail)
    assert run() == 3
    result = json.loads(capsys.readouterr().out)
    assert [r["status"] for r in result["results"]] == ["failed", "not_run"]
    assert not calls and not list(storage.temp.iterdir())


def test_omitted_ranges_transcribes_full_source_once(setup_tool, capsys):
    request, run, clips, calls, instances, *_ = setup_tool
    request.pop("ranges")
    assert run() == 0
    response = json.loads(capsys.readouterr().out)
    result = response["results"]
    assert len(result) == 1 and result[0]["id"] == "full"
    assert (result[0]["start"], result[0]["end"]) == (0, 60)
    assert len(calls) == 1


def test_preset_and_null_threshold_reach_each_range(setup_tool, capsys):
    request, run, clips, calls, *_ = setup_tool
    request["options"] = {"preset": "official-cli", "beam_size": 3,
                          "no_speech_threshold": None, "word_timestamps": True}
    assert run() == 0
    response = json.loads(capsys.readouterr().out)
    opts = response["configuration"]["options"]
    assert opts["beam_size"] == 3 and opts["best_of"] == 5
    assert opts["no_speech_threshold"] is None
    assert opts["temperature"] == [0, .2, .4, .6, .8, 1]
    assert len(calls) == 2
    assert all(call[-1]["no_speech_threshold"] is None for call in calls)
    assert response["results"][1]["segments"][0]["start"] == 20.2
