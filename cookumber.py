# main.py – FastAPI backend (Production-ready for Render)
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
client  = OpenAI()
aclient = AsyncOpenAI()

CHAT_MODEL  = "gpt-4.1"
IMAGE_MODEL = "gpt-image-1"
MAX_STEPS   = 10

# Get port from environment (Render sets this)
PORT = int(os.environ.get("PORT", 8000))

# Create images directory
IMAGES_DIR = Path("/tmp/images")
IMAGES_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Updated CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Serve static files from /tmp/images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

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
        "quiet": True, "no_warnings": True
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    base = dst / info["id"]
    out["audio"]   = next(base.parent.glob(f"{info['id']}.mp3"), None)
    out["subs"]    = next(base.parent.glob(f"{info['id']}*.srt"), None)
    out["thumb"]   = next(base.parent.glob(f"{info['id']}.jpg"), None)
    out["caption"] = (info.get("description") or "").strip()
    return out

def srt_to_text(path: Path) -> str:
    return " ".join(
        s.text.replace("\n", " ").strip()
        for s in pysrt.open(str(path), encoding="utf-8")
    )

def transcribe(audio_path: Path) -> str:
    PRIMARY, FALLBACK = "gpt-4o-transcribe", "whisper-1"
    model = PRIMARY if audio_path.stat().st_size <= 25 * 1024 * 1024 else FALLBACK
    with audio_path.open("rb") as f:
        return client.audio.transcriptions.create(model=model, file=f).text

# ── Recipe extraction ─────────────────────────────────────────────────
def gpt_json(prompt: str, temp: float) -> dict:
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
        except Exception:
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

    rsp = await aclient.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        n=1
    )

    url = rsp.data[0].url
    if url is None and hasattr(rsp.data[0], "b64_json"):
        image_filename = f"{uuid.uuid4()}.png"
        fname = IMAGES_DIR / image_filename
        with open(fname, "wb") as f:
            f.write(base64.b64decode(rsp.data[0].b64_json))
        
        # Use request to get the correct host
        # This will be replaced with actual Render URL
        url = f"/images/{image_filename}"  # Relative URL for now

    return {"step_number": idx, "image_url": url}

# ── Endpoints ────────────────────────────────────────────────────────
@app.post("/import-recipe")
async def import_recipe(req: Request):
    link = (await req.json()).get("link", "").strip()
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

@app.post("/generate-step-images")
async def generate_step_images(req: ImageGenerationRequest):
    steps = req.instructions[:MAX_STEPS]
    if not steps:
        raise HTTPException(400, "instructions list is empty")
    comps = await parse_steps_async(steps)
    sem   = asyncio.Semaphore(5)

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

@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/")
async def root():
    return {"message": "Recipe API is running", "status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)