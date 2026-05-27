"""
server.py — Stable WebSocket server for Voice Conversation avatar UI.

Pipeline:
  Browser mic → binary WebSocket → Whisper STT → Claude → ElevenLabs TTS → binary WebSocket → Browser plays

States sent to browser (JSON text frames):
  {"state": "idle"}
  {"state": "listening"}
  {"state": "thinking"}
  {"state": "speaking", "text": "..."}
  {"state": "error", "message": "..."}

Audio from browser: binary WebSocket frame (webm/ogg blob)
Audio to browser:   binary WebSocket frame (MP3 bytes)
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from dotenv import load_dotenv

load_dotenv()

from stt import transcribe
from llm import Conversation
from tts import speak_to_file

# Thread pool for blocking I/O (Whisper + ElevenLabs are sync)
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()

EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "stop", "end"}
GREETING = "Hello! I'm Yehya, and I'm here to support you!"

# ── Helpers ───────────────────────────────────────────────────────────────────

def cleanup(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass

async def send_json(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass

async def send_audio(ws: WebSocket, mp3_path: str):
    try:
        with open(mp3_path, "rb") as f:
            data = f.read()
        await ws.send_bytes(data)
    except Exception as e:
        print(f"❌ send_audio error: {e}")

async def run_in_thread(fn, *args):
    """Run a blocking function in a thread pool without freezing the server."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fn, *args)

# ── Voice pipeline ────────────────────────────────────────────────────────────

async def handle_audio(ws: WebSocket, conv: Conversation, audio_bytes: bytes):
    """Full pipeline for one user utterance."""
    tmp_audio = None
    mp3_path  = None

    try:
        # Save browser audio blob to temp file
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        tmp_audio.write(audio_bytes)
        tmp_audio.close()

        # 1. Transcribe (runs in thread — non-blocking)
        await send_json(ws, {"state": "thinking"})
        user_text = await run_in_thread(transcribe, tmp_audio.name)
        user_text = user_text.strip()
        print(f"👤 You: {user_text!r}")

        if not user_text:
            await send_json(ws, {"state": "idle"})
            return

        # 2. Exit check
        if any(p in user_text.lower() for p in EXIT_PHRASES):
            farewell = "Goodbye! Have a great day!"
            await send_json(ws, {"state": "speaking", "text": farewell})
            mp3_path = await run_in_thread(speak_to_file, farewell)
            await send_audio(ws, mp3_path)
            await send_json(ws, {"state": "idle"})
            return

        # 3. Claude LLM
        reply = await run_in_thread(conv.send, user_text)
        reply = reply.strip()
        print(f"🤖 Yehya: {reply!r}")

        # 4. TTS → send audio to browser
        await send_json(ws, {"state": "speaking", "text": reply})
        mp3_path = await run_in_thread(speak_to_file, reply)
        await send_audio(ws, mp3_path)
        await send_json(ws, {"state": "idle"})

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback; traceback.print_exc()
        await send_json(ws, {"state": "error", "message": str(e)})
        await send_json(ws, {"state": "idle"})
    finally:
        if tmp_audio: cleanup(tmp_audio.name)
        if mp3_path:  cleanup(mp3_path)

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    conv = Conversation()
    processing = False   # prevent overlapping pipeline calls
    print(f"🔌 Client connected")

    try:
        # Send greeting
        await send_json(ws, {"state": "speaking", "text": GREETING})
        mp3_path = await run_in_thread(speak_to_file, GREETING)
        await send_audio(ws, mp3_path)
        cleanup(mp3_path)
        await send_json(ws, {"state": "idle"})

        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                if processing:
                    print("⚠️  Still processing previous — skipping audio blob")
                    continue
                processing = True
                try:
                    await handle_audio(ws, conv, message["bytes"])
                finally:
                    processing = False

            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                except Exception:
                    msg = {}
                if msg.get("action") == "ping":
                    await send_json(ws, {"action": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print(f"🔌 Client disconnected")

# ── Static routes ─────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())

@app.get("/avatar_video")
async def avatar_video(request: Request):
    base = Path(__file__).parent
    video_path = None
    for name in ("yehyavideo1.mp4", "yehyavideo1.mov", "avatar.mp4", "avatar.mov"):
        p = base / name
        if p.exists():
            video_path = p
            break
    if video_path is None:
        return Response(content="Video not found", status_code=404)

    file_size    = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start      = int(byte_range[0])
        end        = int(byte_range[1]) if byte_range[1] else file_size - 1
        end        = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(iter_file(), status_code=206, headers={
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type":   "video/mp4",
        })
    else:
        def iter_whole():
            with open(video_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data: break
                    yield data
        return StreamingResponse(iter_whole(), status_code=200, headers={
            "Accept-Ranges":  "bytes",
            "Content-Length": str(file_size),
            "Content-Type":   "video/mp4",
        })

@app.get("/logo")
async def logo():
    p = Path(__file__).parent / "logo.svg"
    if p.exists():
        return FileResponse(str(p), media_type="image/svg+xml")
    return Response(content="Not found", status_code=404)

@app.get("/avatar")
async def avatar():
    base = Path(__file__).parent
    for ext in ("jpg", "jpeg", "png"):
        p = base / f"avatar.{ext}"
        if p.exists():
            return FileResponse(str(p))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="260" height="300"><rect width="260" height="300" fill="#2a2a3a" rx="130"/><text x="130" y="160" text-anchor="middle" font-size="80" fill="#555">🤖</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🎙️  Voice Conversation — Avatar UI")
    print("   Open http://localhost:8000 in your browser")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
