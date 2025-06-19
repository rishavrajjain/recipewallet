"""
quotation_bot.py  –  Twilio call ↔︎ OpenAI audio (batch version)

Requires:
  pip install fastapi uvicorn python-multipart websockets pydub audioop \
              twilio==8.* openai>=1.14 google-api-python-client google-auth \
              google-auth-httplib2 google-auth-oauthlib
.env keys:
  OPENAI_API_KEY, GOOGLE_SERVICE_JSON, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
  TWILIO_FROM_NUMBER, NGROK_HOST, GOOGLE_DOCS_TEMPLATE_ID, EMAIL_RECIPIENT
Optional .env keys:
  DEFAULT_TEST_CALL_NUMBER, GOOGLE_DRIVE_FOLDER_ID, KEEP_GOOGLE_DOCS_COPY, PORT, LOG_LEVEL
"""

import os, io, base64, json, datetime, audioop, asyncio, logging, websockets, time
from typing import Dict, List, Optional, Any

import dotenv; dotenv.load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s %(asctime)s %(module)s:%(lineno)d %(funcName)s: %(message)s"
)
log = logging.getLogger("bot")

# --- FastAPI server ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
app = FastAPI()

# --- Twilio Client ---
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
twilio_client: Optional[TwilioClient] = None
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    log.warning("Twilio credentials not found. Twilio functionality will be disabled.")

# --- OpenAI Client ---
from openai import OpenAI, RateLimitError, APIError
openai_client: Optional[OpenAI] = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    log.warning("OPENAI_API_KEY not found. OpenAI functionality will be disabled.")

STT_MODEL, TTS_MODEL, LLM_MODEL = ("whisper-1", "tts-1", "gpt-4o-mini")
OPENAI_TTS_VOICE = "alloy"
OPENAI_TTS_EXPECTED_SAMPLERATE = 24000

# --- Google API Clients ---
from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError as GoogleHttpError
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders

SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/gmail.send"]
CREDS_FILE = os.getenv("GOOGLE_SERVICE_JSON")
CREDS: Optional[service_account.Credentials] = None
DOCS: Optional[Resource] = None
DRIVE: Optional[Resource] = None
GMAIL: Optional[Resource] = None

if CREDS_FILE:
    try:
        CREDS = service_account.Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPES)
        DOCS  = build("docs","v1",credentials=CREDS,cache_discovery=False, static_discovery=False)
        DRIVE = build("drive","v3",credentials=CREDS,cache_discovery=False, static_discovery=False)
        GMAIL = build("gmail","v1",credentials=CREDS,cache_discovery=False, static_discovery=False)
        log.info("Google API clients initialized successfully.")
    except Exception as e:
        log.error(f"Error loading Google service account credentials or building services: {e}")
else:
    log.warning("GOOGLE_SERVICE_JSON env var not set. Google API services will be disabled.")

TEMPLATE_ID = os.getenv("GOOGLE_DOCS_TEMPLATE_ID", "1rtfulGXBRSY_E9RT1g4tOB8kvE-mOuHA9kP1hUn8xMQ")
EMAIL_TO = os.getenv("EMAIL_RECIPIENT", "rishavrajjain@gmail.com")

# --- Pydantic Models ---
from pydantic import BaseModel, Field, ValidationError
class Item(BaseModel):
    name:str
    qty:int
    rate:float
    total:float

class Quote(BaseModel):
    qno:str|None=None
    date:str = Field(default_factory=lambda: datetime.date.today().isoformat())
    address_lines:List[str]=[]
    subject:str|None=None
    items:List[Item]=[]

# --- OpenAI Tools ---
TOOLS=[
  {"type":"function","function":{
     "name":"ask_user","description":"Ask user a clarifying question when information is missing or ambiguous for the quotation.",
     "parameters":{"type":"object",
                   "properties":{"question":{"type":"string", "description": "The question to ask the user."}},
                   "required":["question"]}}},
  {"type":"function","function":{
     "name":"quotation_complete",
     "description":"Called ONLY when ALL necessary quotation fields (address, subject, and at least one item with name, quantity, and rate) have been collected and confirmed with the user.",
     "parameters": Quote.model_json_schema()}}
]

# --- Audio Conversion Utility ---
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

def convert_audio_for_twilio(audio_bytes: bytes, input_samplerate: int = OPENAI_TTS_EXPECTED_SAMPLERATE) -> bytes | None:
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        log.debug(f"Original audio for Twilio: {audio_segment.frame_rate}Hz, {audio_segment.channels}ch, {audio_segment.sample_width*8}-bit")
        audio_segment = audio_segment.set_frame_rate(8000).set_channels(1)
        if audio_segment.sample_width != 2:
             audio_segment = audio_segment.set_sample_width(2)
        pcm_8khz_mono_16bit_data = audio_segment.raw_data
        ulaw_data = audioop.lin2ulaw(pcm_8khz_mono_16bit_data, 2)
        log.debug(f"Converted audio for Twilio to μ-law. Output size: {len(ulaw_data)} bytes")
        return ulaw_data
    except CouldntDecodeError:
        log.error("Pydub couldn't decode audio bytes for Twilio conversion.")
        return None
    except Exception as e:
        log.error(f"Error converting audio for Twilio: {e}", exc_info=True)
        return None

async def text_to_speech_for_twilio(text: str, voice: str = OPENAI_TTS_VOICE) -> bytes | None:
    if not openai_client:
        log.error("OpenAI client not available for TTS.")
        return None
    try:
        log.debug(f"Requesting TTS from OpenAI for text: '{text}'")
        response = await asyncio.to_thread(
            openai_client.audio.speech.create,
            model=TTS_MODEL, voice=voice, input=text, response_format="wav"
        )
        tts_audio_bytes = response.read()
        log.debug(f"Received {len(tts_audio_bytes)} bytes of WAV audio from OpenAI TTS.")
        return convert_audio_for_twilio(tts_audio_bytes)
    except Exception as e:
        log.error(f"Error in text_to_speech_for_twilio: {e}", exc_info=True)
        return None

# --- PDF and Email Utilities (Using placeholders, replace with your full logic) ---
def pdf_from_state(q:Quote)->bytes|None:
    if not DRIVE or not DOCS or not TEMPLATE_ID:
        log.error("Google Drive/Docs API not initialized or TEMPLATE_ID missing for pdf_from_state.")
        return None
    log.info(f"SIMULATING PDF creation for quotation: {q.qno or 'DRAFT'}")
    # Replace this with your actual Google Docs PDF generation logic
    pdf_content = f"SIMULATED PDF\nQNO: {q.qno or 'DRAFT'}\nDate: {q.date}\nSubject: {q.subject}\nItems: {len(q.items)}"
    return pdf_content.encode('utf-8')

def email_pdf(qno:str,pdf_data:bytes):
    if not GMAIL:
        log.error("Gmail API not initialized for email_pdf.")
        return
    log.info(f"SIMULATING Emailing PDF for quotation: {qno} to {EMAIL_TO}")
    # Replace this with your actual Gmail sending logic

# --- In-memory Session Store ---
SESS:Dict[str,Dict[str, Any]] = {}

# --- Public WebSocket URL ---
NGROK_HOST = os.getenv('NGROK_HOST')
PUBLIC_WS = f"wss://{NGROK_HOST}/twilio/audio" if NGROK_HOST else ""
if not NGROK_HOST: log.error("NGROK_HOST env var not set. WebSocket URL will be invalid.")

# --- Twilio Call Endpoint ---
@app.api_route("/call",methods=["GET","POST"])
async def make_call_endpoint(to_number:str|None=None):
    if not twilio_client or not TWILIO_FROM_NUMBER:
        raise HTTPException(status_code=500, detail="Twilio not configured.")
    num_to_call = to_number or os.getenv("DEFAULT_TEST_CALL_NUMBER", "+918586980704")
    if not num_to_call.startswith("+"):
        raise HTTPException(status_code=400,detail="To number must be E.164 format.")
    try:
        call_instance = twilio_client.calls.create(
                            to=num_to_call, from_=TWILIO_FROM_NUMBER,
                            twiml=f"<Response><Connect><Stream url='{PUBLIC_WS}'/></Connect></Response>"
                        )
        log.info(f"Initiating call to {num_to_call}. Call SID: {call_instance.sid}")
        return {"status": "calling", "to": num_to_call, "call_sid": call_instance.sid}
    except TwilioRestException as e:
        raise HTTPException(status_code=500, detail=f"Twilio error: {e.msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not initiate call: {str(e)}")

# --- Constants for Audio Processing ---
TWILIO_EXPECTED_SAMPLERATE = 8000
STT_TARGET_SAMPLERATE = 16000  # Whisper prefers 16kHz

# --- IMPROVED AUDIO PROCESSING CONSTANTS ---
# Larger buffer to gather more audio before processing
SPEECH_MIN_BUFFER_SIZE = 3 * TWILIO_EXPECTED_SAMPLERATE  # 3 seconds minimum
SPEECH_MAX_BUFFER_SIZE = 10 * TWILIO_EXPECTED_SAMPLERATE  # 10 seconds maximum

# Silence detection parameters
SILENCE_THRESHOLD = 400  # Value to consider as silence (in 16-bit PCM)
SILENCE_DURATION_SECONDS = 1.5  # Duration of silence to trigger processing

# --- WebSocket Handler ---
@app.websocket("/twilio/audio")
async def audio_stream_handler(ws:WebSocket):
    await ws.accept()
    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None
    user_audio_buffer = b''
    
    # Silence detection variables
    silence_start_time = None
    last_audio_processing_time = 0
    is_bot_speaking = False
    
    log.info(f"WebSocket connection accepted from {ws.client.host}:{ws.client.port}")

    try:
        while True:
            message_str = await ws.receive_text()
            try: message = json.loads(message_str)
            except json.JSONDecodeError:
                log.error(f"Failed to decode JSON: {message_str}"); continue
            event_type = message.get("event")

            if event_type == "start":
                start_data = message.get("start", {})
                call_sid = start_data.get("callSid")
                stream_sid = message.get("streamSid")
                if not call_sid or not stream_sid:
                    log.error(f"Start event missing callSid/streamSid. Data: {message}")
                    await ws.close(code=1008); return
                log.info(f"Call started: CallSID {call_sid}, StreamSID {stream_sid}")
                SESS[call_sid] = {
                    "state": Quote(), 
                    "msgs": [],
                    "is_processing": False,  # Flag to prevent overlapping processing
                    "last_transcription_text": "",  # Store last transcription to avoid duplicates
                    "last_bot_response_time": time.time()  # Track when bot last responded
                }
                
                greeting_text = f"Hello! I'm QuotationBot. How can I help you create a quote today? Please tell me the customer's full address to begin."
                SESS[call_sid]["msgs"].extend([
                    {"role":"system", "content": (
                        "You are 'QuotationBot', a friendly AI for creating quotations. Collect customer address, subject, and items (name, qty, rate, total). "
                        "Confirm each piece of info. Ask one thing at a time. Use 'ask_user' for clarifications. Call 'quotation_complete' ONLY when all data is collected and confirmed. "
                        "Keep responses brief and focused. Listen carefully to the user's responses."
                    )},
                    {"role": "assistant", "content": greeting_text}
                ])
                
                is_bot_speaking = True
                greeting_audio_ulaw = await text_to_speech_for_twilio(greeting_text)
                if greeting_audio_ulaw and stream_sid:
                    await ws.send_json({
                        "event": "media", "streamSid": stream_sid,
                        "media": {"payload": base64.b64encode(greeting_audio_ulaw).decode()}
                    })
                    log.info(f"[{call_sid}] Sent initial greeting TTS.")
                    SESS[call_sid]["last_bot_response_time"] = time.time()
                    is_bot_speaking = False
                else: 
                    log.error(f"[{call_sid}] Failed to generate/convert greeting audio.")
                    is_bot_speaking = False

            elif event_type == "media":
                if not call_sid or not stream_sid or not openai_client:
                    log.warning(f"Media event premature or OpenAI missing (CallSID: {call_sid})."); continue
                
                # Skip processing if the bot is currently speaking or recently finished speaking
                session = SESS.get(call_sid, {})
                if is_bot_speaking or (time.time() - session.get("last_bot_response_time", 0) < 0.5):
                    continue
                
                if session.get("is_processing", False):
                    # Skip if we're already processing audio
                    continue
                
                media_payload = message.get("media", {}).get("payload")
                if not media_payload: 
                    log.warning(f"[{call_sid}] Media event no payload."); 
                    continue
                
                # Decode and add to buffer
                audio_chunk = base64.b64decode(media_payload)
                user_audio_buffer += audio_chunk
                
                # Convert ulaw to linear PCM for silence detection
                pcm_data = audioop.ulaw2lin(audio_chunk, 2)
                
                # Check for silence in the current chunk
                rms = audioop.rms(pcm_data, 2)
                current_time = time.time()
                
                if rms < SILENCE_THRESHOLD:
                    # Start tracking silence if not already doing so
                    if silence_start_time is None:
                        silence_start_time = current_time
                    
                    # If silence has lasted long enough and we have enough audio, process it
                    if (current_time - silence_start_time >= SILENCE_DURATION_SECONDS and 
                            len(user_audio_buffer) >= SPEECH_MIN_BUFFER_SIZE):
                        
                        # Only process if we have enough data and enough time has passed since last processing
                        if current_time - last_audio_processing_time >= 1.0:  # At least 1 second between processing
                            session["is_processing"] = True
                            try:
                                await process_audio_buffer(ws, call_sid, stream_sid, user_audio_buffer)
                                user_audio_buffer = b''  # Clear buffer after processing
                                last_audio_processing_time = current_time
                            finally:
                                session["is_processing"] = False
                            
                            # Reset silence tracking
                            silence_start_time = None
                else:
                    # Reset silence tracking when sound is detected
                    silence_start_time = None
                
                # Force processing if buffer gets too large
                if len(user_audio_buffer) >= SPEECH_MAX_BUFFER_SIZE and not session.get("is_processing", False):
                    session["is_processing"] = True
                    try:
                        await process_audio_buffer(ws, call_sid, stream_sid, user_audio_buffer)
                        user_audio_buffer = b''  # Clear buffer after processing
                        last_audio_processing_time = current_time
                    finally:
                        session["is_processing"] = False
                    
                    # Reset silence tracking
                    silence_start_time = None

            elif event_type == "stop":
                log.info(f"Call stopped: {message.get('stop',{}).get('callSid', call_sid)}")
                break
            elif event_type == "mark":
                log.info(f"[{call_sid}] Mark event: {message.get('mark',{}).get('name')}")
            else: 
                log.warning(f"[{call_sid}] Unhandled event: {event_type}, Data: {message}")
                
    except WebSocketDisconnect: 
        log.info(f"WS disconnected: {call_sid} (Client: {ws.client.host}:{ws.client.port})")
    except Exception as e: 
        log.error(f"Critical WS error: {call_sid}: {e}", exc_info=True)
    finally:
        if call_sid and call_sid in SESS: 
            SESS.pop(call_sid)
            log.info(f"Session cleared: {call_sid}")
        if ws.client_state != websockets.protocol.State.CLOSED: 
            await ws.close()
            log.info(f"WS closed: {call_sid}")

# --- Process Audio Buffer Function (new) ---
async def process_audio_buffer(ws: WebSocket, call_sid: str, stream_sid: str, audio_buffer: bytes) -> None:
    if not call_sid or not stream_sid or len(audio_buffer) == 0:
        return
    
    session = SESS.get(call_sid)
    if not session:
        log.warning(f"[{call_sid}] Session not found for audio processing")
        return
    
    log.info(f"[{call_sid}] Processing audio buffer of {len(audio_buffer)} bytes")
    
    try:
        # Convert ulaw to PCM and resample from 8kHz to 16kHz for Whisper
        pcm_data_8khz_16bit = audioop.ulaw2lin(audio_buffer, 2)
        pcm_data_16khz_16bit = audioop.ratecv(pcm_data_8khz_16bit, 2, 1, TWILIO_EXPECTED_SAMPLERATE, STT_TARGET_SAMPLERATE, None)[0]
        
        # Create audio segment for STT
        audio_segment_for_stt = AudioSegment(
            data=pcm_data_16khz_16bit, sample_width=2,
            frame_rate=STT_TARGET_SAMPLERATE, channels=1
        )
        
        # Export as WAV for OpenAI's Whisper
        stt_input_wav_obj = io.BytesIO()
        audio_segment_for_stt.export(stt_input_wav_obj, format="wav")
        stt_input_wav_obj.name = "user_speech.wav"  # filename is important for OpenAI client
        stt_input_wav_obj.seek(0)  # Rewind for reading
        
        # Transcribe with Whisper
        transcription_response = await asyncio.to_thread(
            openai_client.audio.transcriptions.create,
            model=STT_MODEL, file=stt_input_wav_obj
        )
        
        transcribed_text = transcription_response.text.strip()
        
        # Skip if transcription is empty or too short
        if not transcribed_text or len(transcribed_text) < 2:
            log.info(f"[{call_sid}] Ignoring short/empty transcription: '{transcribed_text}'")
            return
            
        # Skip if this is a duplicate of the last transcription
        if transcribed_text == session.get("last_transcription_text", ""):
            log.info(f"[{call_sid}] Ignoring duplicate transcription: '{transcribed_text}'")
            return
            
        # Store this transcription to check for duplicates
        session["last_transcription_text"] = transcribed_text
        
        log.info(f"[{call_sid}] Transcribed: '{transcribed_text}'")
        
        # Process with LLM
        reply_text = await handle_text(call_sid, transcribed_text)
        log.info(f"[{call_sid}] LLM Reply: '{reply_text}'")
        
        if reply_text and stream_sid:
            # Flag that the bot is about to speak
            session["is_bot_speaking"] = True
            
            bot_audio_ulaw = await text_to_speech_for_twilio(reply_text)
            if bot_audio_ulaw:
                await ws.send_json({
                    "event": "media", "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(bot_audio_ulaw).decode()}
                })
                log.info(f"[{call_sid}] Sent TTS reply to Twilio.")
                
                # Update the last response time
                session["last_bot_response_time"] = time.time()
            else: 
                log.error(f"[{call_sid}] Failed to generate/convert bot reply audio.")
                
            # Bot has finished speaking
            session["is_bot_speaking"] = False
        else: 
            log.info(f"[{call_sid}] LLM no reply text or stream_sid missing, not sending TTS.")
            
    except RateLimitError as e: 
        log.error(f"[{call_sid}] OpenAI API rate limit: {e}")
    except APIError as e: 
        log.error(f"[{call_sid}] OpenAI API error (STT/TTS): {e.message if hasattr(e, 'message') else e}")
    except Exception as e: 
        log.error(f"[{call_sid}] Error in audio processing: {e}", exc_info=True)

# --- LLM Interaction Logic (handle_text) ---
async def handle_text(call_sid:str, text:str) -> str:
    if not openai_client: return "OpenAI client not ready."
    if call_sid not in SESS: return "Session not found."
    session_data = SESS[call_sid]
    message_history: List[Dict[str, Any]] = session_data['msgs']
    message_history.append({"role":"user","content":text})
    log.debug(f"[{call_sid}] History for LLM: {json.dumps(message_history, indent=2)}")
    try:
        completion = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=LLM_MODEL, messages=message_history,
            tools=TOOLS, tool_choice="auto", temperature=0.3,
            max_tokens=150  # Limit response length to keep bot concise
        )
        rsp_message = completion.choices[0].message
    except Exception as e:
        log.error(f"[{call_sid}] LLM API error: {e}", exc_info=True)
        return "Sorry, I couldn't process that. Could you please repeat?"

    llm_response_dict = {"role": rsp_message.role}
    if rsp_message.content: llm_response_dict["content"] = rsp_message.content
    if rsp_message.tool_calls:
        llm_response_dict["tool_calls"] = [
            {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in rsp_message.tool_calls
        ]
    message_history.append(llm_response_dict)

    if rsp_message.tool_calls:
        tool_call = rsp_message.tool_calls[0]
        tool_function_name = tool_call.function.name
        tool_arguments_str = tool_call.function.arguments or "{}"
        log.info(f"[{call_sid}] LLM tool: {tool_function_name}, Args: {tool_arguments_str}")
        try: tool_args = json.loads(tool_arguments_str)
        except json.JSONDecodeError:
            message_history.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_function_name, "content": "Error: Malformed JSON."})
            return "I'm having trouble processing your information. Let's try again."

        tool_response_content = ""
        assistant_reply_after_tool = ""
        if tool_function_name == "quotation_complete":
            try:
                session_data['state'] = Quote(**tool_args)
                pdf_bytes = email_and_get(session_data['state'])
                if pdf_bytes:
                    tool_response_content = "Quote PDF sent."
                    assistant_reply_after_tool = "Great! Your quotation has been finalized and emailed. Thank you for using our service."
                else:
                    tool_response_content = "Failed to send PDF."
                    assistant_reply_after_tool = "I've created your quote, but there was an issue sending the PDF. Our team will contact you shortly."
            except ValidationError as ve:
                tool_response_content = f"Validation Error: {ve.errors()}"
                missing_fields = [e['loc'][0] for e in ve.errors()]
                assistant_reply_after_tool = f"We're missing some important details: {', '.join(missing_fields)}. Could you provide those?"
            except Exception as e:
                tool_response_content = f"Error: {str(e)}"
                assistant_reply_after_tool = "I'm having trouble finalizing your quote. Let's continue with the information you need to provide."
        elif tool_function_name == "ask_user":
            question_to_ask = tool_args.get("question", "Could you clarify that for me?")
            tool_response_content = f"Asking: {question_to_ask}"
            assistant_reply_after_tool = question_to_ask
        else:
            tool_response_content = f"Unknown tool: {tool_function_name}"
            assistant_reply_after_tool = "I didn't understand that. Let's continue with your quotation details."
        message_history.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_function_name, "content": tool_response_content})
        if assistant_reply_after_tool: message_history.append({"role": "assistant", "content": assistant_reply_after_tool})
        return assistant_reply_after_tool
    return rsp_message.content or "Could you repeat that, please?"


def email_and_get(q_state:Quote) -> bytes | None:
    if not q_state.qno: q_state.qno = f"Q{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    pdf_data = pdf_from_state(q_state)
    if pdf_data: email_pdf(q_state.qno, pdf_data); return pdf_data
    return None

# --- Main Execution ---
if __name__=="__main__":
    import uvicorn
    essential_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY, "GOOGLE_SERVICE_JSON": CREDS_FILE,
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID, "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
        "TWILIO_FROM_NUMBER": TWILIO_FROM_NUMBER, "NGROK_HOST": NGROK_HOST,
        "GOOGLE_DOCS_TEMPLATE_ID": TEMPLATE_ID
    }
    missing = [k for k, v in essential_vars.items() if not v]
    if missing: log.critical(f"MISSING CRITICAL ENV VARS: {', '.join(missing)}.")
    else: log.info("All critical environment variables appear set.")
    log.info(f"Quotation Bot Starting... WebSocket URL for Twilio: {PUBLIC_WS}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))