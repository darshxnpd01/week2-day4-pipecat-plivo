"""
Day 4 - Pipecat + Plivo: AI Receptionist Over Real Phone Calls
==============================================================
Manual WebSocket handler — no pipecat transport layer.
Direct control over Plivo's bidirectional audio protocol.

Pipeline per call:
  Plivo audio (mulaw 8kHz) → PCM 16kHz → Deepgram STT
  Deepgram transcript → OpenAI LLM → text
  text → ElevenLabs TTS → PCM 16kHz → mulaw 8kHz → Plivo

How to run:
  source ~/venv-pipecat/bin/activate
  python server.py
  # In second terminal: ngrok http 8000
  # Plivo Answer URL: https://xxx.ngrok.io/answer (POST)
"""

import os
import json
import base64
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse
import uvicorn
from dotenv import load_dotenv
from openai import AsyncOpenAI
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_KEY   = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"
POSTGRES_URL     = os.getenv("POSTGRES_URL_NON_POOLING") or os.getenv("POSTGRES_URL")
REDIS_URL        = os.getenv("KV_URL") or os.getenv("REDIS_URL")
WEBSOCKET_BASE_URL = os.getenv("WEBSOCKET_BASE_URL", "")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ─── Optional DB/Cache ────────────────────────────────────────────────────────
try:
    import asyncpg
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

db_pool = None
redis_client = None

# ─── Audio helpers ────────────────────────────────────────────────────────────
import audioop
import math
import struct

def generate_test_tone(freq: int = 440, duration_ms: int = 500) -> bytes:
    """Generate a pure sine tone as mulaw 8kHz bytes (for format testing)."""
    n = int(8000 * duration_ms / 1000)
    pcm = bytearray()
    for i in range(n):
        v = int(16000 * math.sin(2 * math.pi * freq * i / 8000))
        pcm.extend(struct.pack('<h', v))
    return audioop.lin2ulaw(bytes(pcm), 2)

def mulaw_to_pcm(mulaw_bytes: bytes) -> bytes:
    """mulaw 8kHz → PCM 16-bit 16kHz"""
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k

def pcm_to_mulaw(pcm_bytes: bytes) -> bytes:
    """PCM 16-bit 16kHz → mulaw 8kHz"""
    pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(pcm_8k, 2)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly AI receptionist for Mario's Italian Kitchen.
You answer phone calls to help with reservations and questions.
Keep ALL responses to 1-2 short sentences — this is a phone call.
Be warm and natural. Never use bullet points.
For reservations collect: name, date, time, party size — one at a time.
Available times: 5:30 PM, 6:00 PM, 6:30 PM, 7:00 PM, 7:30 PM, 8:00 PM.
Once you have ALL four details (name, date, time, party size), call save_reservation immediately."""

RESERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "save_reservation",
        "description": "Save a confirmed reservation to the database. Call this once you have collected name, date, time, and party size.",
        "parameters": {
            "type": "object",
            "properties": {
                "name":       {"type": "string",  "description": "Guest name"},
                "date":       {"type": "string",  "description": "Reservation date e.g. 'February 10th'"},
                "time":       {"type": "string",  "description": "Reservation time e.g. '6:00 PM'"},
                "party_size": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["name", "date", "time", "party_size"],
        },
    },
}

# ─── DB helpers ───────────────────────────────────────────────────────────────
async def log_call(caller: str, called: str) -> Optional[int]:
    if not db_pool:
        return None
    try:
        return await db_pool.fetchval(
            """INSERT INTO call_logs (caller_number, called_number, call_status, created_at)
               VALUES ($1, $2, 'started', NOW()) RETURNING id""",
            caller, called,
        )
    except Exception as e:
        logger.warning(f"DB log_call error: {e}")
        return None

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client
    if POSTGRES_URL and HAS_POSTGRES:
        try:
            db_pool = await asyncpg.create_pool(POSTGRES_URL)
            await db_pool.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id SERIAL PRIMARY KEY, caller_number TEXT, called_number TEXT,
                    call_status TEXT DEFAULT 'started', transcript_summary TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await db_pool.execute("""
                CREATE TABLE IF NOT EXISTS reservations (
                    id SERIAL PRIMARY KEY, name TEXT, party_size INTEGER,
                    date TEXT, time TEXT, confirmation_number TEXT UNIQUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ PostgreSQL connected")
        except Exception as e:
            logger.error(f"❌ PostgreSQL: {e}")

    if REDIS_URL and HAS_REDIS:
        try:
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.error(f"❌ Redis: {e}")

    yield

    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.aclose()


app = FastAPI(title="Plivo Voice AI", lifespan=lifespan)

# ─── HTTP Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "postgres": "connected" if db_pool else "not configured",
        "redis": "connected" if redis_client else "not configured",
    }


@app.post("/answer")
async def answer_call(request: Request):
    form = await request.form()
    caller = form.get("From", "unknown")
    called = form.get("To", "unknown")
    call_uuid = form.get("CallUUID", "")
    logger.info(f"Incoming call: {caller} → {called} (UUID: {call_uuid})")

    if db_pool:
        asyncio.create_task(log_call(caller, called))

    if redis_client:
        try:
            await redis_client.setex(
                f"call:{call_uuid}", 1800,
                json.dumps({"caller": caller, "started": datetime.now(timezone.utc).isoformat()}),
            )
        except Exception as e:
            logger.warning(f"Redis error: {e}")

    if WEBSOCKET_BASE_URL:
        ws_url = f"wss://{WEBSOCKET_BASE_URL}/ws"
    else:
        host = request.headers.get("host", "localhost:8000")
        ws_url = f"wss://{host}/ws"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream streamTimeout="86400" keepCallAlive="true" bidirectional="true"
          contentType="audio/x-mulaw;rate=8000">
    {ws_url}
  </Stream>
</Response>"""
    return Response(content=xml, media_type="application/xml")


@app.get("/reservations")
async def list_reservations():
    if not db_pool:
        return JSONResponse({"error": "Database not configured"}, status_code=503)
    rows = await db_pool.fetch("SELECT * FROM reservations ORDER BY created_at DESC LIMIT 50")
    return [dict(r) for r in rows]


# ─── WebSocket — Manual Plivo Handler ─────────────────────────────────────────

async def stream_tts_to_plivo(websocket: WebSocket, text: str, send_lock: asyncio.Lock):
    """ElevenLabs TTS → mulaw 8kHz → Plivo playAudio (correct format per docs)."""
    logger.info(f"[TTS] starting: '{text[:60]}'")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}/stream"
    headers = {"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_turbo_v2_5"}
    params  = {"output_format": "pcm_16000", "optimize_streaming_latency": 2}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, params=params) as resp:
                logger.info(f"[TTS] ElevenLabs status: {resp.status}")
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[TTS] ElevenLabs error: {resp.status} {body}")
                    return

                pcm_buf    = bytearray()
                CHUNK      = 640   # 20ms of PCM 16kHz → 160 bytes mulaw
                n_sent     = 0
                pace_start = None

                async for raw in resp.content.iter_any():
                    if not raw:
                        continue
                    pcm_buf.extend(raw)
                    while len(pcm_buf) >= CHUNK:
                        frame = bytes(pcm_buf[:CHUNK])
                        pcm_buf = pcm_buf[CHUNK:]
                        mulaw = pcm_to_mulaw(frame)
                        b64   = base64.b64encode(mulaw).decode()
                        # ── Correct Plivo playAudio format (per official docs) ──
                        msg = json.dumps({
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": b64,
                            },
                        })
                        async with send_lock:
                            await websocket.send_text(msg)
                        n_sent += 1

                        # Real-time pacing: 20ms per chunk
                        if pace_start is None:
                            pace_start = asyncio.get_event_loop().time()
                        expected = pace_start + n_sent * 0.020
                        delay = expected - asyncio.get_event_loop().time()
                        if delay > 0.001:
                            await asyncio.sleep(delay)

                # flush tail
                if len(pcm_buf) >= 2:
                    if len(pcm_buf) % 2 != 0:
                        pcm_buf = pcm_buf[:-1]
                    mulaw = pcm_to_mulaw(bytes(pcm_buf))
                    b64   = base64.b64encode(mulaw).decode()
                    async with send_lock:
                        await websocket.send_text(json.dumps({
                            "event": "playAudio",
                            "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": b64},
                        }))
                    n_sent += 1

                logger.info(f"[TTS] done — sent {n_sent} chunks to Plivo")

    except asyncio.CancelledError:
        logger.info("[TTS] cancelled")
    except Exception as e:
        logger.error(f"[TTS] error: {e}", exc_info=True)


async def save_reservation_db(name: str, date: str, time: str, party_size: int) -> str:
    import random, string
    confirmation = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    if db_pool:
        try:
            await db_pool.execute(
                """INSERT INTO reservations (name, party_size, date, time, confirmation_number, created_at)
                   VALUES ($1, $2, $3, $4, $5, NOW())""",
                name, party_size, date, time, confirmation,
            )
            logger.info(f"[DB] Reservation saved: {name} {date} {time} x{party_size} → {confirmation}")
        except Exception as e:
            logger.error(f"[DB] save_reservation error: {e}")
    return confirmation


async def get_ai_response(conversation: list) -> str:
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        tools=[RESERVATION_TOOL],
        tool_choice="auto",
        max_tokens=150,
        temperature=0.7,
    )
    choice = resp.choices[0]

    # If the LLM wants to call save_reservation, do it and get a confirmation
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        tool_call = choice.message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        confirmation = await save_reservation_db(
            name=args["name"],
            date=args["date"],
            time=args["time"],
            party_size=args["party_size"],
        )
        # Feed result back to LLM so it can confirm to the caller
        conversation.append(choice.message)
        conversation.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Reservation saved. Confirmation number: {confirmation}",
        })
        follow_up = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            max_tokens=100,
            temperature=0.7,
        )
        return follow_up.choices[0].message.content

    return choice.message.content


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, call_uuid: str = ""):
    await websocket.accept()
    logger.info(f"Plivo WebSocket connected call_uuid={call_uuid}")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    send_lock    = asyncio.Lock()
    active_tts   = None
    ws_open      = True

    async def handle_transcript(text: str):
        nonlocal active_tts
        if not ws_open:
            return
        text = text.strip()
        if not text:
            return
        logger.info(f"[USER] '{text}'")
        if active_tts and not active_tts.done():
            active_tts.cancel()
        conversation.append({"role": "user", "content": text})
        try:
            logger.info("[LLM] calling OpenAI...")
            ai_text = await get_ai_response(conversation)
            logger.info(f"[LLM] response: '{ai_text}'")
        except Exception as e:
            logger.error(f"[LLM] error: {e}", exc_info=True)
            return
        if not ws_open:
            return
        conversation.append({"role": "assistant", "content": ai_text})
        active_tts = asyncio.create_task(stream_tts_to_plivo(websocket, ai_text, send_lock))

    # ── Deepgram setup ──────────────────────────────────────────────────────
    dg_client = DeepgramClient(DEEPGRAM_API_KEY)
    dg_conn   = dg_client.listen.asynclive.v("1")

    async def on_transcript(self, result, **kwargs):
        alt = result.channel.alternatives[0]
        transcript = alt.transcript
        if transcript:
            logger.info(f"[DG] final={result.is_final} text='{transcript}'")
        if result.is_final and transcript:
            task = asyncio.create_task(handle_transcript(transcript))
            task.add_done_callback(lambda t: logger.error(f"handle_transcript error: {t.exception()}") if not t.cancelled() and t.exception() else None)

    dg_conn.on(LiveTranscriptionEvents.Transcript, on_transcript)

    await dg_conn.start(LiveOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        encoding="linear16",
        sample_rate=16000,
        endpointing=400,
    ))

    # ── Main receive loop ───────────────────────────────────────────────────
    try:
        async for raw_msg in websocket.iter_text():
            try:
                msg = json.loads(raw_msg)
            except Exception:
                continue

            event = msg.get("event", "")
            if event != "media":
                logger.info(f"[PLIVO] event={event} raw={raw_msg[:200]}")

            if event == "start":
                logger.info("Stream started — sending greeting")
                active_tts = asyncio.create_task(
                    stream_tts_to_plivo(websocket, "Hi, thank you for calling Mario's Italian Kitchen! How can I help you today?", send_lock)
                )

            elif event == "media":
                media = msg.get("media", {})
                if media.get("track", "inbound") != "inbound":
                    continue
                payload = media.get("payload", "")
                if payload:
                    mulaw = base64.b64decode(payload)
                    pcm   = mulaw_to_pcm(mulaw)
                    await dg_conn.send(pcm)

            elif event == "stop":
                logger.info("Plivo sent stop")
                break

    except WebSocketDisconnect:
        logger.info("Caller disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        ws_open = False
        if active_tts and not active_tts.done():
            active_tts.cancel()
        await dg_conn.finish()
        logger.info("Call ended")


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n{'='*55}")
    print(f"  Plivo Voice AI Server (Manual WebSocket)")
    print(f"  http://0.0.0.0:{port}")
    print(f"  Health: http://localhost:{port}/health")
    print(f"{'='*55}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
