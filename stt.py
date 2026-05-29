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


def _is_arabic_audio(audio_path: str) -> bool:
    """
    Quick first-pass transcription with no language hint to detect Arabic.
    Returns True if the result contains Arabic script characters.
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    text = result.strip() if isinstance(result, str) else str(result).strip()
    # Arabic Unicode block: U+0600–U+06FF
    return bool(__import__('re').search(r'[؀-ۿ]', text))


def transcribe(audio_path: str, language: str = None) -> str:
    """
    Transcribe speech from an audio file using Whisper.
    Auto-detects Arabic vs English: runs a quick detection pass first,
    then transcribes with the correct language forced to avoid hallucination.

    Args:
        audio_path: Path to a WAV or MP3 file.
        language:   Force a specific language, or None for smart auto-detect.

    Returns:
        The transcribed text (stripped of leading/trailing whitespace).
    """
    client = _get_client()

    # If language not forced, detect Arabic vs English
    if not language:
        is_arabic = _is_arabic_audio(audio_path)
        language = "ar" if is_arabic else "en"
        print(f"🌐 Language detected: {'Arabic' if is_arabic else 'English'}")

    with open(audio_path, "rb") as f:
        kwargs = {
            "model": "whisper-1",
            "file": f,
            "response_format": "text",
            "language": language,
            "prompt": "WhatsApp, message, send, bibi, private, hello." if language == "en" else "",
        }
        if not kwargs["prompt"]:
            del kwargs["prompt"]

        text = client.audio.transcriptions.create(**kwargs)

    return text.strip() if isinstance(text, str) else str(text).strip()
