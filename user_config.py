"""Compatibility import for scripts using the old settings module."""
from yt_whisper.storage import configure_storage
from yt_whisper.config import load_config

user_config = load_config(configure_storage())
