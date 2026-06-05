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
import re
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response, JSONResponse
from dotenv import load_dotenv

load_dotenv()

from stt import transcribe
from llm import Conversation, generate_suggestions
from tts import speak_to_file
from whatsapp import build_deep_link

# ── Emotion detection ─────────────────────────────────────────────────────────
_EMOTION_MAP = {
    "happy":   ["happy","great","awesome","love","haha","lol","yay","amazing","wonderful","excited","glad","joy","fantastic","good","nice","مبسوط","كتير منيح","رائع","يسلمو","الحمدلله"],
    "sad":     ["sad","miss","lonely","cry","crying","depressed","upset","sorry","hurt","pain","حزين","زعلان","بكي","وحيد","مش منيح"],
    "angry":   ["angry","mad","hate","stupid","annoying","frustrated","shut","damn","ugh","كرهت","غاضب","زعلان","انرفزت"],
    "excited": ["wow","omg","wait","seriously","no way","really","can't believe","finally","يا إلهي","جد","مش معقول","والله جد"],
    "thinking":["hmm","let me","wonder","maybe","think","not sure","يمكن","بفكر","مش عارف"],
}

def detect_emotion(text: str) -> str:
    """Return one of: happy, sad, angry, excited, thinking, neutral."""
    t = text.lower()
    scores = {e: 0 for e in _EMOTION_MAP}
    for emotion, keywords in _EMOTION_MAP.items():
        for kw in keywords:
            if kw in t:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"

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

# ── WhatsApp intent detection ─────────────────────────────────────────────────
_WA_KW = r"(?:whatsapp|whats\s*app|watsapp|what'?s\s*app|واتساب|واتس|وتساب)"

# "WA <NAME> <MESSAGE>"  — shortest form e.g. "whatsapp private hello" (commas optional)
_WA_SHORT = re.compile(
    _WA_KW + r"[,\s]+(\w+)[,\s]+(.+)",
    re.IGNORECASE,
)
# "send/message <NAME> <MESSAGE>" — no whatsapp keyword needed (commas optional)
_WA_MSG_NAME = re.compile(
    r"(?:message|msg|text)[,\s]+(\w+)[,\s]+(.+)",
    re.IGNORECASE,
)
# "send a whatsapp to <NAME> saying <MESSAGE>"
_WA_TO_SAYING = re.compile(
    r"(?:send|write|compose)\s+(?:a\s+)?" + _WA_KW +
    r"(?:\s+(?:message|msg|text))?"
    r"\s+to\s+(\w+)"
    r"\s+(?:saying|that says?|with\s+(?:the\s+)?(?:message|text)?)?\s*[\"']?(.+)[\"']?",
    re.IGNORECASE,
)
# "send a whatsapp saying <MESSAGE>"
_WA_NO_TO = re.compile(
    r"(?:send|write|compose)\s+(?:a\s+)?" + _WA_KW +
    r"(?:\s+(?:message|msg|text))?"
    r"(?:\s+to\s+)?"
    r"\s+(?:saying|that says?|with\s+(?:the\s+)?(?:message|text)?)\s*[\"']?(.+)[\"']?",
    re.IGNORECASE,
)
# Arabic with recipient
_WA_AR_TO = re.compile(
    r"(?:ابعت|أرسل|بعت|ارسل).*?" + _WA_KW +
    r".*?(?:لـ?|إلى\s*)(\w+)"
    r".*?(?:قلو|قلها|قللو|رسالة|إنو|انو)?\s*(.+)",
    re.IGNORECASE,
)
_WA_AR = re.compile(
    r"(?:ابعت|أرسل|بعت|ارسل).*?" + _WA_KW +
    r".*?(?:قلو|قلها|قللو|رسالة|إنو|انو)?\s*(.+)",
    re.IGNORECASE,
)

def detect_whatsapp_intent(text: str):
    """
    Returns (message: str, recipient_name: str|None) if WhatsApp intent detected, else None.
    recipient_name is the spoken name (e.g. 'Mom', 'John') or None for default contact.
    """
    # Strip punctuation (commas, periods, etc.) that Whisper often inserts
    t = re.sub(r'[,،.!?]', ' ', text.strip())
    t = re.sub(r'\s+', ' ', t).strip()

    # Shortest: "whatsapp private hello"
    m = _WA_SHORT.search(t)
    if m:
        name, msg = m.group(1).strip(), m.group(2).strip().strip('"\'')
        if msg:
            return msg, name

    # "message private hello"
    m = _WA_MSG_NAME.search(t)
    if m:
        name, msg = m.group(1).strip(), m.group(2).strip().strip('"\'')
        if msg:
            return msg, name

    # "send a whatsapp to Mom saying I'll be late"
    m = _WA_TO_SAYING.search(t)
    if m:
        name, msg = m.group(1).strip(), m.group(2).strip().strip('"\'')
        if msg:
            return msg, name

    # "send a whatsapp saying hello"
    m = _WA_NO_TO.search(t)
    if m:
        msg = m.group(1).strip().strip('"\'')
        if msg:
            return msg, None

    # Arabic with recipient
    m = _WA_AR_TO.search(t)
    if m:
        name, msg = m.group(1).strip(), m.group(2).strip().strip('"\'')
        if msg:
            return msg, name

    # Arabic without recipient
    m = _WA_AR.search(t)
    if m:
        msg = m.group(1).strip().strip('"\'')
        if msg:
            return msg, None

    # Broad fallback: contains whatsapp keyword
    if re.search(r"whatsapp|whats\s*app|watsapp|واتساب|واتس", t, re.IGNORECASE):
        after = re.search(
            r"(?:saying|that says?|message saying|قلو|قلها|رسالة|إنو|انو)\s+(.+)",
            t, re.IGNORECASE
        )
        if after:
            return after.group(1).strip().strip('"\''), None

    return None


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

        # 3. WhatsApp intent — detected from user's speech directly (no Claude JSON needed)
        print(f"🔍 Checking WhatsApp intent for: {user_text!r}")
        wa_result = detect_whatsapp_intent(user_text)
        print(f"🔍 WhatsApp result: {wa_result!r}")
        if wa_result:
            wa_message, wa_recipient = wa_result
            print(f"📱 WhatsApp intent detected. To: {wa_recipient!r} | Message: {wa_message!r}")
            is_arabic = bool(re.search(r"[؀-ۿ]", user_text))
            if is_arabic:
                spoken_reply = f"تمام! افتح واتساب وابعت الرسالة{' لـ' + wa_recipient if wa_recipient else ''}!"
            else:
                to_str = f" to {wa_recipient}" if wa_recipient else ""
                spoken_reply = f"Opening WhatsApp{to_str} — just tap Send!"
            deep_link = build_deep_link(wa_message, wa_recipient)
            conv.history.append({"role": "user", "content": user_text})
            conv.history.append({"role": "assistant", "content": spoken_reply})
            await send_json(ws, {"state": "speaking", "text": spoken_reply, "whatsapp_link": deep_link})
            mp3_path = await run_in_thread(speak_to_file, spoken_reply)
            await send_audio(ws, mp3_path)
            await send_json(ws, {"state": "idle"})
            return

        # 4. Claude LLM (normal conversation)
        reply = await run_in_thread(conv.send, user_text)
        reply = reply.strip()
        print(f"🤖 Yehya: {reply!r}")

        # 5. TTS → send audio to browser
        emotion = detect_emotion(user_text)
        await send_json(ws, {"state": "speaking", "text": reply, "user_text": user_text, "emotion": emotion})
        mp3_path = await run_in_thread(speak_to_file, reply)
        await send_audio(ws, mp3_path)

        # 6. Generate suggestions after audio sent
        suggestions = await run_in_thread(generate_suggestions, reply, user_text)
        await send_json(ws, {"state": "idle", "suggestions": suggestions})

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback; traceback.print_exc()
        await send_json(ws, {"state": "error", "message": str(e)})
        await send_json(ws, {"state": "idle"})
    finally:
        if tmp_audio: cleanup(tmp_audio.name)
        if mp3_path:  cleanup(mp3_path)

# ── Text pipeline (typed input, same as audio but no STT) ────────────────────

async def handle_text(ws: WebSocket, conv: Conversation, user_text: str):
    """Full pipeline for one typed user message."""
    mp3_path = None
    try:
        print(f"⌨️  Text: {user_text!r}")
        await send_json(ws, {"state": "thinking", "user_text": user_text})

        # WhatsApp intent check
        wa_result = detect_whatsapp_intent(user_text)
        if wa_result:
            wa_message, wa_recipient = wa_result
            is_arabic = bool(re.search(r"[؀-ۿ]", user_text))
            if is_arabic:
                spoken_reply = f"تمام! افتح واتساب وابعت الرسالة{' لـ' + wa_recipient if wa_recipient else ''}!"
            else:
                to_str = f" to {wa_recipient}" if wa_recipient else ""
                spoken_reply = f"Opening WhatsApp{to_str} — just tap Send!"
            deep_link = build_deep_link(wa_message, wa_recipient)
            conv.history.append({"role": "user", "content": user_text})
            conv.history.append({"role": "assistant", "content": spoken_reply})
        else:
            spoken_reply = await run_in_thread(conv.send, user_text)
            spoken_reply = spoken_reply.strip()

        print(f"🤖 Yehya: {spoken_reply!r}")

        emotion = detect_emotion(user_text)
        extra = {"whatsapp_link": deep_link} if wa_result else {}
        await send_json(ws, {"state": "speaking", "text": spoken_reply, "emotion": emotion, **extra})
        mp3_path = await run_in_thread(speak_to_file, spoken_reply)
        await send_audio(ws, mp3_path)

        # Generate suggestions after audio sent
        suggestions = await run_in_thread(generate_suggestions, spoken_reply, user_text)
        await send_json(ws, {"state": "idle", "suggestions": suggestions})

    except Exception as e:
        print(f"❌ Text pipeline error: {e}")
        import traceback; traceback.print_exc()
        await send_json(ws, {"state": "error", "message": str(e)})
        await send_json(ws, {"state": "idle"})
    finally:
        if mp3_path: cleanup(mp3_path)


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
                elif msg.get("action") == "text_input" and msg.get("text"):
                    if processing:
                        continue
                    processing = True
                    try:
                        await handle_text(ws, conv, msg["text"].strip())
                    finally:
                        processing = False

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

# ── Contacts management ───────────────────────────────────────────────────────

CONTACTS_FILE = Path(__file__).parent / "contacts.json"
CONTACTS_PASSCODE = "987987987"

def _load_contacts_file() -> dict:
    """Load contacts from contacts.json file."""
    if CONTACTS_FILE.exists():
        try:
            return json.loads(CONTACTS_FILE.read_text())
        except Exception:
            pass
    # Fall back to parsing WHATSAPP_CONTACTS env var
    contacts = {}
    raw = os.environ.get("WHATSAPP_CONTACTS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            parts = entry.split(":", 1)
            name   = parts[0].strip()
            number = parts[1].strip()
            contacts[name] = number
    return contacts

def _save_contacts_file(contacts: dict):
    """Save contacts to contacts.json file."""
    CONTACTS_FILE.write_text(json.dumps(contacts, ensure_ascii=False, indent=2))

@app.get("/contacts")
async def contacts_page():
    html_path = Path(__file__).parent / "contacts.html"
    return HTMLResponse(html_path.read_text())

@app.get("/api/contacts")
async def api_get_contacts(passcode: str = ""):
    if passcode != CONTACTS_PASSCODE:
        return JSONResponse({"error": "Invalid passcode"}, status_code=401)
    contacts = _load_contacts_file()
    return JSONResponse({"contacts": [{"name": k, "number": v} for k, v in contacts.items()]})

@app.post("/api/contacts")
async def api_add_contact(request: Request):
    body = await request.json()
    if body.get("passcode") != CONTACTS_PASSCODE:
        return JSONResponse({"error": "Invalid passcode"}, status_code=401)
    name   = body.get("name", "").strip()
    number = body.get("number", "").strip()
    if not name or not number:
        return JSONResponse({"error": "Name and number are required"}, status_code=400)
    # Normalise number
    if not number.startswith("+"):
        number = "+" + number
    contacts = _load_contacts_file()
    contacts[name] = number
    _save_contacts_file(contacts)
    print(f"📋 Contact added: {name} → {number}")
    return JSONResponse({"ok": True, "name": name, "number": number})

@app.delete("/api/contacts/{name}")
async def api_delete_contact(name: str, passcode: str = ""):
    if passcode != CONTACTS_PASSCODE:
        return JSONResponse({"error": "Invalid passcode"}, status_code=401)
    contacts = _load_contacts_file()
    if name not in contacts:
        return JSONResponse({"error": "Contact not found"}, status_code=404)
    del contacts[name]
    _save_contacts_file(contacts)
    print(f"📋 Contact deleted: {name}")
    return JSONResponse({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("🎙️  Voice Conversation — Avatar UI")
    print(f"   Open http://localhost:{port} in your browser")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
