"""
voice.py — Main entry point for the Voice Conversation MVP.

Pipeline:
  1. Record microphone audio until the user stops speaking
  2. Transcribe speech with OpenAI Whisper (STT)
  3. Send transcript to Claude (LLM) and get a reply
  4. Synthesise the reply with ElevenLabs (TTS)
  5. Play the audio back to the user
  6. Repeat until the user says "exit", "quit", or "goodbye"

Usage:
    python voice.py
"""

import os
import sys
import signal
import tempfile

from dotenv import load_dotenv

# Load API keys from .env before importing sub-modules
load_dotenv()

from audio import record_until_silence, play_audio
from stt import transcribe
from llm import Conversation
from tts import speak_to_file

# Words that end the conversation
EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "stop", "end"}

# Greeting the assistant says at startup
GREETING = "Hello! I'm your voice assistant. How can I help you today?"


def cleanup_file(path: str) -> None:
    """Delete a temporary file if it exists."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def run() -> None:
    print("=" * 50)
    print("  🎙️  Voice Conversation MVP")
    print("  Powered by Whisper · Claude · ElevenLabs")
    print("  Say 'exit' or 'quit' to end the session.")
    print("=" * 50)

    conversation = Conversation()

    # --- Greet the user ---
    print(f"\n🤖 Assistant: {GREETING}")
    greeting_audio = speak_to_file(GREETING)
    play_audio(greeting_audio)
    cleanup_file(greeting_audio)

    # --- Main conversation loop ---
    while True:
        wav_path = None
        mp3_path = None

        try:
            # Step 1: Record
            wav_path = record_until_silence()

            # Step 2: Transcribe
            print("📝 Transcribing...", end=" ", flush=True)
            user_text = transcribe(wav_path)
            print(f"Done.\n👤 You: {user_text}")

            if not user_text.strip():
                print("  (no speech detected, try again)")
                continue

            # Check for exit intent
            if any(phrase in user_text.lower() for phrase in EXIT_PHRASES):
                farewell = "Goodbye! Have a great day!"
                print(f"\n🤖 Assistant: {farewell}")
                farewell_audio = speak_to_file(farewell)
                play_audio(farewell_audio)
                cleanup_file(farewell_audio)
                break

            # Step 3: Get Claude's reply
            print("🧠 Thinking...", end=" ", flush=True)
            reply = conversation.send(user_text)
            print(f"Done.\n🤖 Assistant: {reply}")

            # Step 4: Synthesise speech
            print("🔊 Synthesising...", end=" ", flush=True)
            mp3_path = speak_to_file(reply)
            print("Done.")

            # Step 5: Play back
            play_audio(mp3_path)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting.")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Continuing to next turn...\n")

        finally:
            cleanup_file(wav_path)
            cleanup_file(mp3_path)

    print("\n👋 Session ended.")


def _handle_sigterm(sig, frame):
    print("\nReceived SIGTERM. Exiting cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    run()
