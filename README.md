# Voice Conversation MVP

A real-time voice assistant that listens to you, thinks with Claude, and talks back.

**Pipeline:** Microphone → Whisper (STT) → Claude (LLM) → ElevenLabs (TTS) → Speaker

---

## Requirements

- Python 3.10+
- A microphone and speakers
- API keys for: **Anthropic**, **OpenAI**, **ElevenLabs**

On macOS, PyAudio needs PortAudio:
```bash
brew install portaudio
```

On Ubuntu/Debian:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

---

## Setup

**1. Clone / open the project folder**

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API keys**
```bash
cp .env.example .env
# Then open .env and fill in your keys
```

You need:
- `ANTHROPIC_API_KEY` — from https://console.anthropic.com
- `OPENAI_API_KEY` — from https://platform.openai.com/api-keys
- `ElevenLabs_API_KEY` — from https://elevenlabs.io/app

---

## Run

```bash
python voice.py
```

The assistant greets you, then listens. Speak naturally — it stops recording after ~1.5 seconds of silence, transcribes your speech, sends it to Claude, and reads the reply back to you.

Say **"exit"**, **"quit"**, or **"goodbye"** to end the session.

---

## Customisation

| File | What to change |
|------|----------------|
| `llm.py` | System prompt, Claude model, max response tokens |
| `tts.py` | ElevenLabs voice ID, TTS model (`eleven_turbo_v2` → `eleven_multilingual_v2` for other languages) |
| `audio.py` | `SILENCE_THRESHOLD` (mic sensitivity), `SILENCE_DURATION`, `MAX_RECORD_SECONDS` |
| `stt.py` | `language` parameter (default `"en"` — set `None` for auto-detect) |

### Changing the Claude model
In `llm.py`, update the `model` parameter in `Conversation.__init__`:
```python
model: str = "claude-opus-4-6"   # Most capable
# or
model: str = "claude-sonnet-4-6" # Faster / cheaper
```

### Changing the ElevenLabs voice
Find a voice ID at https://elevenlabs.io/voice-library, then update `DEFAULT_VOICE_ID` in `tts.py`.

---

## Project Structure

```
Voice Conversation/
├── voice.py          # Main conversation loop (entry point)
├── audio.py          # Mic recording & audio playback
├── stt.py            # Whisper speech-to-text
├── llm.py            # Claude conversation manager
├── tts.py            # ElevenLabs text-to-speech
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
└── README.md         # This file
```

---

## Troubleshooting

**No audio input detected** — check that your microphone is the system default input device and try lowering `SILENCE_THRESHOLD` in `audio.py`.

**PyAudio install fails on macOS** — run `brew install portaudio` first, then retry pip install.

**ElevenLabs quota exceeded** — you're on the free tier (10k chars/month). Upgrade or swap `tts.py` for a free alternative like `pyttsx3`.

**Whisper returns empty string** — the recording may have been too quiet. Try speaking louder or adjusting `SILENCE_THRESHOLD`.
