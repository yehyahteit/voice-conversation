"""
audio.py — Microphone recording and audio playback utilities.
Uses PyAudio for recording and pygame (or sounddevice) for playback.
"""

import os
import wave
import tempfile
import pyaudio
import pygame
import threading

# Recording settings
SAMPLE_RATE = 16000       # Hz — optimal for Whisper
CHANNELS = 1              # Mono
CHUNK = 1024              # Frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio
SILENCE_THRESHOLD = 500   # RMS amplitude below this = silence
SILENCE_DURATION = 0.8    # Seconds of silence before stopping recording
MAX_RECORD_SECONDS = 30   # Hard cap to avoid runaway recording


def _rms(data: bytes) -> float:
    """Compute root-mean-square of a raw PCM chunk (Int16, little-endian)."""
    import struct
    count = len(data) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"<{count}h", data)
    return (sum(s * s for s in shorts) / count) ** 0.5


def record_until_silence(prompt: str = "🎤 Listening...") -> str:
    """
    Record from the default microphone until the user stops speaking.
    Returns the path to a temporary WAV file.
    """
    print(f"\n{prompt}", flush=True)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    silent_chunks = 0
    required_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
    max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK)
    speaking_started = False

    for _ in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        rms = _rms(data)

        if rms > SILENCE_THRESHOLD:
            speaking_started = True
            silent_chunks = 0
        elif speaking_started:
            silent_chunks += 1
            if silent_chunks >= required_silent_chunks:
                break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Write to a temp WAV file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print("✅ Captured.", flush=True)
    return tmp.name


def play_audio(file_path: str) -> str | None:
    """
    Play an audio file (MP3 or WAV).
    Stops immediately if the user starts speaking (interrupt detection).
    Returns path to a WAV file containing the captured interrupt speech,
    or None if playback finished normally.
    """
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Listen for user interruption while playing
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT, channels=CHANNELS,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=CHUNK
    )

    INTERRUPT_THRESHOLD = 600
    INTERRUPT_CHUNKS = 2
    loud_count = 0
    captured_frames = []   # keep ALL frames during playback
    interrupted = False

    while pygame.mixer.music.get_busy():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = _rms(data)
            captured_frames.append(data)

            if rms > INTERRUPT_THRESHOLD:
                loud_count += 1
                if loud_count >= INTERRUPT_CHUNKS:
                    pygame.mixer.music.stop()
                    print("⚡ Interrupted!", flush=True)
                    interrupted = True
                    break
            else:
                loud_count = 0
        except Exception:
            pass
        pygame.time.wait(20)

    # Continue recording until silence (finish capturing what user is saying)
    if interrupted:
        silent_chunks = 0
        required_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
        max_extra = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK)
        print("🎤 Continuing to capture...", flush=True)
        for _ in range(max_extra):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                captured_frames.append(data)
                rms = _rms(data)
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= required_silent_chunks:
                        break
                else:
                    silent_chunks = 0
            except Exception:
                break

    stream.stop_stream()
    stream.close()
    pa.terminate()
    pygame.mixer.music.unload()

    if interrupted and captured_frames:
        # Save captured speech to a temp WAV for transcription
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(captured_frames))
        print("✅ Interrupt speech captured.", flush=True)
        return tmp.name

    return None
