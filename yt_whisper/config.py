import json
from pathlib import Path
import tempfile
import warnings

DEFAULTS = {"model": "base", "language": "Auto"}


def load_config(storage):
    path = storage.config
    if not path.exists():
        # Read old repository settings once; never write to them.
        path = Path(__file__).resolve().parent.parent / "user_config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return {"model": data.get("model") or "base", "language": data.get("language") or "Auto"}
    except FileNotFoundError:
        return DEFAULTS.copy()
    except (OSError, ValueError) as exc:
        warnings.warn(f"Could not read settings from {path}: {exc}; using defaults.")
        return DEFAULTS.copy()


def save_config(storage, model, language):
    storage.config.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=storage.config.parent,
                                     suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump({"model": model, "language": language or "Auto"}, handle, ensure_ascii=False)
    try:
        temporary.replace(storage.config)
    finally:
        temporary.unlink(missing_ok=True)
