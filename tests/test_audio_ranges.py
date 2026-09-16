"""Range decoding is local and does not load or download any ASR model."""
import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import numpy as np
import pytest

from yt_whisper.audio import load_audio_range, probe_audio_duration


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg/ffprobe required")
@pytest.mark.parametrize("extension", ["wav", "flac"])
def test_real_ffmpeg_seeks_to_requested_samples_without_changing_source(tmp_path, extension):
    samples = np.random.default_rng(0).integers(-16000, 16000, size=16000 * 3, dtype=np.int16)
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())
    if extension == "flac":
        encoded = source.with_suffix(".flac")
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-i", str(source), str(encoded)],
                       capture_output=True, check=True)
        source = encoded
    before = source.read_bytes()
    existing = set(tmp_path.iterdir())
    assert probe_audio_duration(source) == pytest.approx(3)
    actual = load_audio_range(source, 1.25, 2.5)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, samples[20000:40000].astype(np.float32) / 32768.0)
    assert source.read_bytes() == before and set(tmp_path.iterdir()) == existing


@pytest.mark.parametrize("stream_duration,expected", [("3.5", 3.5), ("N/A", 5), (None, 5)])
def test_duration_prefers_first_audio_stream_with_container_fallback(tmp_path, monkeypatch, stream_duration, expected):
    source = tmp_path / "audio"
    source.touch()
    monkeypatch.setattr("yt_whisper.audio.shutil.which", lambda name: name)
    metadata = {"streams": [{"codec_type": "video", "duration": "10"},
                             {"codec_type": "audio", "duration": stream_duration},
                             {"codec_type": "audio", "duration": "20"}], "format": {"duration": "5"}}
    monkeypatch.setattr("yt_whisper.audio.subprocess.run", Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata))))
    assert probe_audio_duration(source) == expected


@pytest.mark.parametrize("metadata", [
    {"streams": [], "format": {"duration": "5"}},
    {"streams": [{"codec_type": "audio", "duration": "nan"}], "format": {"duration": "inf"}},
    {},
])
def test_unknown_duration_is_a_clear_input_error(tmp_path, monkeypatch, metadata):
    from yt_whisper.tool import ToolError, probe_audio
    source = tmp_path / "audio"
    source.touch()
    monkeypatch.setattr("yt_whisper.audio.shutil.which", lambda name: name)
    monkeypatch.setattr("yt_whisper.audio.subprocess.run", Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata))))
    with pytest.raises(RuntimeError, match="Cannot read audio duration"):
        probe_audio_duration(source)
    with pytest.raises(ToolError) as error:
        probe_audio(source)
    assert error.value.code == "input_error"


@pytest.mark.parametrize("start,end", [(2, 1), (1, 1), (-1, 1), (0, float("inf"))])
def test_invalid_bounds_never_start_decoder(monkeypatch, start, end):
    run = Mock()
    monkeypatch.setattr("yt_whisper.audio.subprocess.run", run)
    with pytest.raises(ValueError, match="Audio ranges"):
        load_audio_range("audio.wav", start, end)
    run.assert_not_called()


@pytest.mark.parametrize("pcm", [b"", b"\x00\x00", b"\x00"])
def test_empty_truncated_or_invalid_pcm_is_rejected(monkeypatch, pcm):
    monkeypatch.setattr("yt_whisper.audio.require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("yt_whisper.audio.subprocess.run", Mock(return_value=SimpleNamespace(stdout=pcm)))
    with pytest.raises(RuntimeError, match="Cannot decode audio range"):
        load_audio_range("audio.wav", 1, 2)
