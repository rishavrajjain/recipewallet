# main.py – FastAPI backend (June 2025) - Render Ready
# Reels → recipe JSON + on-demand GPT-4.1 step-image generation
# deps: openai>=1.21.0 fastapi uvicorn yt-dlp pysrt python-dotenv

import os, json, time, tempfile, asyncio, base64, uuid
from pathlib import Path
from typing import List, Dict

import yt_dlp, pysrt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

# Get environment variables with defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client  = OpenAI(api_key=OPENAI_API_KEY)
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

CHAT_MODEL  = "gpt-4.1"
IMAGE_MODEL = "gpt-image-1"
MAX_STEPS   = 10

# Get the server URL from environment or use default
SERVER_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")

app = FastAPI(
    title="Recipe Extractor API",
    description="Extract recipes from video reels and generate step images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Create temp directory if it doesn't exist
TEMP_DIR = "/tmp/recipe_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount static files for serving generated images
app.mount("/images", StaticFiles(directory=TEMP_DIR), name="images")

class ImageGenerationRequest(BaseModel):
    instructions: List[str]
    recipe_title: str = "Recipe"

# ── YouTube helpers ───────────────────────────────────────────────────
def run_yt_dlp(url: str, dst: Path) -> dict:
    out = dict(audio=None, subs=None, thumb=None, caption="")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "hi", ""],
        "subtitlesformat": "srt",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }],
        "quiet": True, 
        "no_warnings": True
    }
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise HTTPException(400, f"Failed to download video: {str(e)}")

    base = dst / info["id"]
    out["audio"]   = next(base.parent.glob(f"{info['id']}.mp3"), None)
    out["subs"]    = next(base.parent.glob(f"{info['id']}*.srt"), None)
    out["thumb"]   = next(base.parent.glob(f"{info['id']}.jpg"), None)
    out["caption"] = (info.get("description") or "").strip()
    return out

def srt_to_text(path: Path) -> str:
    try:
        return " ".join(
            s.text.replace("\n", " ").strip()
            for s in pysrt.open(str(path), encoding="utf-8")
        )
    except Exception:
        return ""

def transcribe(audio_path: Path) -> str:
    PRIMARY, FALLBACK = "gpt-4o-transcribe", "whisper-1"
    model = PRIMARY if audio_path.stat().st_size <= 25 * 1024 * 1024 else FALLBACK
    
    try:
        with audio_path.open("rb") as f:
            return client.audio.transcriptions.create(model=model, file=f).text
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""

# ── Recipe extraction ─────────────────────────────────────────────────
def gpt_json(prompt: str, temp: float) -> dict:
    try:
        rsp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=temp,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user",    "content": prompt}
            ]
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        print(f"GPT JSON extraction failed: {e}")
        raise

def extract_recipe(caption: str, srt_text: str, speech: str) -> dict:
    prompt = (
        "Build one recipe. Return JSON keys: title, description, "
        "ingredients (list), steps (list).\n\n"
        f"POST_CAPTION:\n{caption}\n\nCLOSED_CAPTIONS:\n{srt_text}\n\n"
        f"SPEECH_TRANSCRIPT:\n{speech}"
    )
    for t in (0.1, 0.5):
        try:
            data = gpt_json(prompt, t)
            if data.get("ingredients") and data.get("steps"):
                return data
        except Exception as e:
            print(f"Recipe extraction attempt failed: {e}")
            continue
    
    return {
        "title": "Imported Recipe",
        "description": "Recipe from Reel",
        "ingredients": ["Add ingredients manually."],
        "steps": ["Add steps manually."]
    }

# ── Step parsing (single GPT-4.1 call) ────────────────────────────────
async def parse_steps_async(steps: List[str]) -> List[Dict[str, str]]:
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    prompt = (
        "Return ONE JSON object {\"steps\": [...]}. "
        "Array length must equal number of instructions; "
        "each item has verb, mainIngredient, vessel, visible_change "
        "(<=3 words each).\n\nINSTRUCTIONS:\n" + joined
    )
    
    try:
        rsp = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return {'steps':[...]} only."},
                {"role": "user",    "content": prompt}
            ]
        )
        data  = json.loads(rsp.choices[0].message.content)
        items = data.get("steps", [])
        if isinstance(items, str):
            items = json.loads(items)
        if not isinstance(items, list) or len(items) != len(steps):
            items = [{"verb": "prepare",
                      "mainIngredient": "ingredients",
                      "vessel": "vessel",
                      "visible_change": "ready"} for _ in steps]
        return items
    except Exception as e:
        print(f"Step parsing failed: {e}")
        return [{"verb": "prepare",
                 "mainIngredient": "ingredients",
                 "vessel": "vessel",
                 "visible_change": "ready"} for _ in steps]

# ── Image generation ─────────────────────────────────────────────────
async def generate_step_image(idx: int, comp: Dict[str, str]) -> Dict[str, str]:
    prompt = (
        f"High-resolution food photo, cinematic color grade.\n"
        f"Scene: Step {idx}. {comp['verb']} {comp['mainIngredient']} "
        f"in {comp['vessel']}; {comp['visible_change']}.\n"
        "Camera: 50 mm prime, f/2.8, ISO 200, 1/125 s. "
        "Angle: top-down 90°. Surface: rustic dark-oak board. "
        "Props: neutral linen napkin, pinch bowl sea salt. "
        "Lighting: soft window light from left, natural shadows. "
        "Aspect: 1:1. Negative: hands, faces, brand logos, text."
    )

    try:
        rsp = await aclient.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        url = rsp.data[0].url
        if url is None and hasattr(rsp.data[0], "b64_json"):
            image_filename = f"{uuid.uuid4()}.png"
            fname = os.path.join(TEMP_DIR, image_filename)
            with open(fname, "wb") as f:
                f.write(base64.b64decode(rsp.data[0].b64_json))
            
            # Use the server URL from environment
            url = f"{SERVER_URL}/images/{image_filename}"

        return {"step_number": idx, "image_url": url}
    except Exception as e:
        raise Exception(f"Image generation failed for step {idx}: {str(e)}")

# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Recipe Extractor API", 
        "version": "1.0.0",
        "endpoints": ["/health", "/import-recipe", "/generate-step-images"]
    }

@app.post("/import-recipe")
async def import_recipe(req: Request):
    try:
        body = await req.json()
        link = body.get("link", "").strip()
        if not link:
            raise HTTPException(400, "link is required")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp   = Path(tmpdir)
            info  = run_yt_dlp(link, tmp)
            cap   = info["caption"]
            srt   = srt_to_text(info["subs"]) if info["subs"] else ""
            speech= transcribe(info["audio"]) if info["audio"] else ""
            recipe= extract_recipe(cap, srt, speech)
        
        return {"success": True, "recipe": recipe}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/generate-step-images")
async def generate_step_images(req: ImageGenerationRequest):
    try:
        steps = req.instructions[:MAX_STEPS]
        if not steps:
            raise HTTPException(400, "instructions list is empty")
        
        comps = await parse_steps_async(steps)
        sem   = asyncio.Semaphore(3)  # Reduced concurrency for Render

        async def worker(i, c):
            async with sem:
                return await generate_step_image(i, c)

        res = await asyncio.gather(
            *[worker(i+1, c) for i, c in enumerate(comps)],
            return_exceptions=True
        )

        good, bad = [], []
        for i, r in enumerate(res, 1):
            if isinstance(r, dict):
                r["step_text"] = steps[i-1]
                good.append(r)
            else:
                bad.append({"step_number": i, "error": str(r)})

        return {"success": len(bad) == 0,
                "generated_images": good,
                "failed_steps": bad}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)