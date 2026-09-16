"""Configure storage before importing libraries that initialize caches."""
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class Storage:
    root: Path

    @property
    def temp(self):
        return self.root / "tmp"

    @property
    def models(self):
        return self.root / "models/whisper"

    @property
    def thai_model(self):
        return self.root / "models/thonburian"

    @property
    def outputs(self):
        return self.root / "outputs"

    @property
    def config(self):
        return self.root / "config/user_config.json"


def configure_storage(root=None):
    default = r"E:\yt-whisper" if os.name == "nt" else str(Path.home() / ".local/share/yt-whisper")
    path = Path(root or os.environ.get("YTW_DATA_DIR", default)).expanduser().resolve()
    if os.name == "nt" and not Path(path.anchor).exists():
        raise RuntimeError(f"Storage drive unavailable: {path}. Set YTW_DATA_DIR to a drive with enough space.")
    storage = Storage(path)
    folders = {
        "TEMP": storage.temp, "TMP": storage.temp, "TMPDIR": storage.temp,
        "GRADIO_TEMP_DIR": storage.temp / "gradio",
        "PIP_CACHE_DIR": path / "cache/pip",
        "CONDA_PKGS_DIRS": path / "cache/conda-pkgs",
        "XDG_CACHE_HOME": path / "cache",
        "HF_HOME": path / "cache/huggingface",
        "HF_HUB_CACHE": path / "cache/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": path / "cache/huggingface/hub",
        "TORCH_HOME": path / "cache/torch",
        "NUMBA_CACHE_DIR": path / "cache/numba",
        "MPLCONFIGDIR": path / "cache/matplotlib",
        "CUDA_CACHE_PATH": path / "cache/cuda",
    }
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    if "TRANSFORMERS_CACHE" in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = str(folders["HF_HUB_CACHE"])
    for folder in (storage.models, storage.outputs, storage.config.parent):
        folder.mkdir(parents=True, exist_ok=True)
    os.environ["YTW_DATA_DIR"] = str(path)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    tempfile.tempdir = str(storage.temp)
    return storage
