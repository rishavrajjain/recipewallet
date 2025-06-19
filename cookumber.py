# main.py – FastAPI backend (June 2025) - Render Ready
# Reels → recipe JSON + on-demand GPT-4o step-image generation
# deps: openai>=1.30.0 fastapi uvicorn yt-dlp pysrt python-dotenv

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

CHAT_MODEL  = "gpt-4o"
IMAGE_MODEL = "dall-e-3"
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
    
    # Base options for all platforms
    base_opts = {
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
    
    # Instagram-specific options
    if "instagram.com" in url:
        base_opts.update({
            "cookiefile": None,  # We'll try without cookies first
            "http_headers": {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        })
    
    # Try multiple strategies for Instagram
    strategies = [base_opts]
    
    # If Instagram, add fallback strategies
    if "instagram.com" in url:
        # Strategy 2: Try with cookies from browser (if available)
        strategy2 = base_opts.copy()
        strategy2.update({
            "cookiesfrombrowser": ("chrome", None, None, None),
            "ignoreerrors": True
        })
        strategies.append(strategy2)
        
        # Strategy 3: Try with different format
        strategy3 = base_opts.copy()
        strategy3.update({
            "format": "worst",  # Try lower quality
            "ignoreerrors": True
        })
        strategies.append(strategy3)
    
    last_error = None
    for i, opts in enumerate(strategies):
        try:
            print(f"Attempting download strategy {i+1}/{len(strategies)} for {url}")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                break
        except Exception as e:
            last_error = e
            print(f"Strategy {i+1} failed: {str(e)}")
            continue
    else:
        # All strategies failed
        if "instagram.com" in url:
            raise HTTPException(400, 
                "Instagram content requires authentication. Please try with a public Instagram reel or use a different platform. "
                "Instagram has strict rate limits and may require login for some content.")
        else:
            raise HTTPException(400, f"Failed to download video: {str(last_error)}")

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
    # Use whisper-1 as it's the most reliable model
    model = "whisper-1"
    
    try:
        with audio_path.open("rb") as f:
            response = client.audio.transcriptions.create(
                model=model, 
                file=f,
                response_format="text"
            )
            # Handle both string response and object response
            if isinstance(response, str):
                return response
            else:
                return response.text if hasattr(response, 'text') else str(response)
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

# ── Step parsing (single GPT-4o call) ────────────────────────────────
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
        print("=== Import Recipe Request Started ===")
        body = await req.json()
        link = body.get("link", "").strip()
        print(f"Processing URL: {link}")
        
        if not link:
            raise HTTPException(400, "link is required")
        
        # Check if it's an Instagram URL and provide helpful message
        if "instagram.com" in link:
            print("Detected Instagram URL, attempting extraction...")
            # Try to extract anyway, but prepare for failure
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp   = Path(tmpdir)
                    print(f"Created temp directory: {tmp}")
                    info  = run_yt_dlp(link, tmp)
                    print(f"yt-dlp extraction successful: {info}")
                    cap   = info["caption"]
                    srt   = srt_to_text(info["subs"]) if info["subs"] else ""
                    speech= transcribe(info["audio"]) if info["audio"] else ""
                    print(f"Transcription completed. Caption length: {len(cap)}, SRT length: {len(srt)}, Speech length: {len(speech)}")
                    recipe= extract_recipe(cap, srt, speech)
                    print(f"Recipe extraction successful: {recipe.get('title', 'No title')}")
                
                return {"success": True, "recipe": recipe}
            except HTTPException as he:
                print(f"HTTP Exception during Instagram processing: {he.status_code} - {he.detail}")
                # For Instagram failures, provide a more helpful response
                if he.status_code == 400 and "Instagram content requires authentication" in str(he.detail):
                    return {
                        "success": False,
                        "error": "instagram_auth_required",
                        "message": "Instagram content requires authentication. Please try one of these alternatives:",
                        "suggestions": [
                            "Use a public Instagram reel that doesn't require login",
                            "Try a YouTube video instead",
                            "Copy the recipe text manually and paste it into the app",
                            "Use a different social media platform (TikTok, YouTube Shorts, etc.)"
                        ],
                        "fallback_recipe": {
                            "title": "Manual Recipe Entry",
                            "description": "Please enter your recipe details manually",
                            "ingredients": ["Add your ingredients here"],
                            "steps": ["Add your cooking steps here"]
                        }
                    }
                else:
                    raise he
            except Exception as inner_e:
                print(f"Unexpected error during Instagram processing: {type(inner_e).__name__}: {str(inner_e)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(500, f"Instagram processing failed: {str(inner_e)}")
        else:
            print("Processing non-Instagram URL...")
            # Non-Instagram URL - process normally
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp   = Path(tmpdir)
                print(f"Created temp directory: {tmp}")
                info  = run_yt_dlp(link, tmp)
                print(f"yt-dlp extraction successful: {info}")
                cap   = info["caption"]
                srt   = srt_to_text(info["subs"]) if info["subs"] else ""
                speech= transcribe(info["audio"]) if info["audio"] else ""
                print(f"Transcription completed. Caption length: {len(cap)}, SRT length: {len(srt)}, Speech length: {len(speech)}")
                recipe= extract_recipe(cap, srt, speech)
                print(f"Recipe extraction successful: {recipe.get('title', 'No title')}")
            
            return {"success": True, "recipe": recipe}
            
    except HTTPException as he:
        print(f"HTTP Exception: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
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
    uvicorn.run("cookumber:app", host="0.0.0.0", port=port)