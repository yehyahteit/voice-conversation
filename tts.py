"""
tts.py — Text-to-Speech using ElevenLabs API.
Uses eleven_turbo_v2_5 (multilingual model — supports Arabic).
Switches voice based on language: Arabic voice for Arabic text, default voice for English.
"""

import os
import re
import tempfile
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

_client: ElevenLabs | None = None

# Yehya — custom cloned voice (English)
DEFAULT_VOICE_ID = "NvewI3hPEke44ohYVCpa"

# Lebanese Arabic voice
ARABIC_VOICE_ID = "ZadBDdhKhprUwKSus5SD"

_ARABIC_RE = re.compile(r'[؀-ۿ]')


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
    voice_id: str = None,
    model_id: str = "eleven_turbo_v2_5",   # multilingual model — supports Arabic
) -> str:
    """
    Convert text to speech and save as a temporary MP3 file.
    Auto-selects Arabic voice when text contains Arabic script.
    Returns the path (caller is responsible for cleanup).
    """
    client = _get_client()

    # Auto-select voice based on language if not explicitly provided
    if voice_id is None:
        is_arabic = bool(_ARABIC_RE.search(text))
        voice_id = ARABIC_VOICE_ID if is_arabic else DEFAULT_VOICE_ID
        print(f"🔊 Voice: {'Arabic' if is_arabic else 'English'} ({voice_id})")

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
