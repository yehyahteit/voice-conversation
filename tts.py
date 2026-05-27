"""
tts.py — Text-to-Speech using ElevenLabs API.
Uses eleven_flash_v2_5 (fastest model) and streams audio to a temp file.
"""

import os
import tempfile
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

_client: ElevenLabs | None = None

# Yehya — custom cloned voice
DEFAULT_VOICE_ID = "NvewI3hPEke44ohYVCpa"


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise EnvironmentError("ELEVENLABS_API_KEY is not set in environment.")
        _client = ElevenLabs(api_key=api_key)
    return _client


def speak_to_file(
    text: str,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = "eleven_turbo_v2_5",   # multilingual model — supports Arabic
) -> str:
    """
    Convert text to speech and save as a temporary MP3 file.
    Returns the path (caller is responsible for cleanup).
    """
    client = _get_client()

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        voice_settings=VoiceSettings(
            stability=0.75,
            similarity_boost=0.9,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    for chunk in audio_generator:
        tmp.write(chunk)
    tmp.close()

    return tmp.name
