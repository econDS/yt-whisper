import os
from pathlib import Path
import tempfile

# Configure caches before pytest imports any machine-learning/UI libraries.
os.environ.setdefault("YTW_DATA_DIR", str(Path(tempfile.gettempdir()) / "yt-whisper-tests"))
from yt_whisper.storage import configure_storage
configure_storage()
