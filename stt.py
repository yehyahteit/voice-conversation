"""
stt.py — Speech-to-Text using OpenAI's Whisper API.
Sends a WAV/MP3 file to the Whisper endpoint and returns the transcript.
"""

import os
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment.")
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe(audio_path: str, language: str = None) -> str:
    """
    Transcribe speech from an audio file using Whisper.

    Args:
        audio_path: Path to a WAV or MP3 file.
        language:   ISO-639-1 language code, or None to auto-detect (default).
                    e.g. "en" for English, "ar" for Arabic.

    Returns:
        The transcribed text (stripped of leading/trailing whitespace).
    """
    client = _get_client()

    with open(audio_path, "rb") as f:
        kwargs = {
            "model": "whisper-1",
            "file": f,
            "response_format": "text",
        }
        if language:
            kwargs["language"] = language

        text = client.audio.transcriptions.create(**kwargs)

    return text.strip() if isinstance(text, str) else str(text).strip()
