# main.py – FastAPI backend
# Instagram Reels & TikTok → recipe JSON + on-demand GPT-4 step-image generation
# deps: openai>=1.21.0 fastapi uvicorn yt-dlp pysrt python-dotenv python-multipart aiofiles

import os, json, time, tempfile, asyncio, base64, uuid, random
from pathlib import Path
from typing import List, Dict, Union, Optional
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

import yt_dlp, pysrt, aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
import httpx
import re, urllib.parse

# --- Configuration ---
load_dotenv()
client = OpenAI()
aclient = AsyncOpenAI()

CHAT_MODEL  = "gpt-4.1"
IMAGE_MODEL = "gpt-image-1"
MAX_STEPS = 10

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
USER_UPLOADS_DIR = Path("/tmp/user_uploads")
USER_UPLOADS_DIR.mkdir(exist_ok=True)
# --- End Configuration ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the upload directory exists on startup."""
    USER_UPLOADS_DIR.mkdir(exist_ok=True)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error on {request.method} {request.url}")
    body = await request.body()
    print(f"Request body: {body.decode()}")
    print(f"Validation errors: {exc.errors()}")
    return await request_validation_exception_handler(request, exc)

app.mount("/images", StaticFiles(directory=USER_UPLOADS_DIR), name="images")

# --- Pydantic Models (NEW CONTRACTS) ---
class ImageGenerationRequest(BaseModel):
    instructions: List[str]
    recipe_title: str = "Recipe"

class NutritionInfo(BaseModel):
    calories: Optional[int] = Field(None, description="Estimated total calories for the recipe.")
    protein: Optional[int] = Field(None, description="Estimated total protein in grams.")
    carbs: Optional[int] = Field(None, description="Estimated total carbohydrates in grams.")
    fats: Optional[int] = Field(None, description="Estimated total fats in grams.")
    portions: Optional[int] = Field(1, description="Estimated number of portions this recipe makes.")

class IngredientCategory(str, Enum):
    FRUIT_VEG = "Fruit & Vegetables"
    MEAT_POULTRY_FISH = "Meat, Poultry, Fish"
    PASTA_RICE_GRAINS = "Pasta, Rice & Grains"
    HERBS_SPICES = "Herbs & Spices"
    CUPBOARD_STAPLES = "Cupboard Staples"
    DAIRY = "Dairy"
    CANNED_JARRED = "Canned & Jarred"
    OTHER = "Other"

class CategorizedIngredient(BaseModel):
    name: str
    category: IngredientCategory = Field(..., description="The category of the ingredient.")

class Recipe(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    imageUrl: str
    prepTime: int = Field(..., description="Preparation time in minutes.")
    cookTime: int = Field(..., description="Cooking time in minutes.")
    difficulty: str = Field(..., description="Difficulty rating, e.g., 'Easy', 'Medium', 'Hard'.")
    nutrition: NutritionInfo
    ingredients: List[CategorizedIngredient]
    steps: List[str]
    isFromReel: bool = True
    extractedFrom: str = Field(..., description="Platform source: 'instagram', 'tiktok', 'youtube', or 'website'")
    creatorHandle: Optional[str] = Field(None, description="Creator's username/handle (e.g., @chefname)")
    creatorName: Optional[str] = Field(None, description="Creator's display name")
    createdAt: Union[str, datetime] = Field(default_factory=datetime.utcnow)

    @field_validator('createdAt', mode='before')
    @classmethod
    def parse_created_at(cls, v):
        if isinstance(v, datetime): return v.isoformat()
        if isinstance(v, str): return v
        return str(v)

class HealthAnalysisRequest(BaseModel):
    recipe: Recipe
    blood_test_id: Optional[str] = None
    include_blood_test: bool = False

class BloodMarker(BaseModel):
    marker: str
    current_level: float
    predicted_impact: str
    target_range: str
    is_out_of_range: bool

class HealthCulprit(BaseModel):
    ingredient: str
    impact: str
    severity: str

class HealthBooster(BaseModel):
    ingredient: str
    impact: str
    severity: str

class Recommendations(BaseModel):
    should_avoid: bool
    modifications: List[str]
    alternative_recipes: List[str]

class HealthAnalysis(BaseModel):
    overall_score: int
    risk_level: str
    personal_message: str
    main_culprits: List[HealthCulprit]
    health_boosters: List[HealthBooster]
    recommendations: Recommendations
    blood_markers_affected: List[BloodMarker]

class HealthAnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[HealthAnalysis] = None
    error: Optional[str] = None

# --- End Pydantic Models ---

# --- Core Logic Functions ---
def run_yt_dlp(url: str, dst: Path) -> dict:
    out = dict(audio=None, subs=None, thumb=None, caption="", thumbnail_url="", platform="", creator_handle="", creator_name="")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst / "%(id)s.%(ext)s"),
        "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": ["en"], "subtitlesformat": "srt",
        "writethumbnail": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True, "no_warnings": True
    }
    
    # Add platform-specific headers for better success rates
    if "tiktok.com" in url.lower():
        opts["http_headers"] = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    elif "instagram.com" in url.lower():
        opts["http_headers"] = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    base = dst / info["id"]
    out["audio"] = next(base.parent.glob(f"{info['id']}.mp3"), None)
    out["subs"] = next(base.parent.glob(f"{info['id']}*.srt"), None)
    out["thumb"] = next(base.parent.glob(f"{info['id']}.jpg"), None) or next(base.parent.glob(f"{info['id']}.webp"), None)
    out["caption"] = (info.get("description") or info.get("title") or "").strip()
    out["thumbnail_url"] = info.get("thumbnail", "")
    
    # Extract platform and creator information
    if "instagram.com" in url.lower():
        out["platform"] = "instagram"
    elif "tiktok.com" in url.lower():
        out["platform"] = "tiktok"
    elif "youtube.com" in url.lower() or "youtu.be" in url.lower():
        out["platform"] = "youtube"
    else:
        out["platform"] = "website"
    
    # Extract creator info from yt-dlp metadata
    out["creator_handle"] = info.get("uploader_id", "") or info.get("channel_id", "")
    out["creator_name"] = info.get("uploader", "") or info.get("channel", "")
    
    # Add @ prefix to handle if it doesn't have one
    if out["creator_handle"] and not out["creator_handle"].startswith("@"):
        out["creator_handle"] = f"@{out['creator_handle']}"
    
    return out

def srt_to_text(path: Path) -> str:
    try:
        return " ".join(s.text.replace("\n", " ").strip() for s in pysrt.open(str(path), encoding="utf-8"))
    except Exception:
        return ""

def transcribe(audio_path: Path) -> str:
    with audio_path.open("rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f).text

def gpt_json(prompt: str, temp: float) -> dict:
    rsp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temp,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a culinary assistant. Your task is to accurately parse video transcripts into a structured JSON recipe format. Adhere strictly to the requested schema."},
            {"role": "user", "content": prompt}
        ]
    )
    return json.loads(rsp.choices[0].message.content)

def extract_recipe(caption: str, srt_text: str, speech: str, thumbnail_url: str = "", platform: str = "", creator_handle: str = "", creator_name: str = "") -> dict:
    categories_list = ", ".join(f'"{cat.value}"' for cat in IngredientCategory)
    prompt = (
        "Based on the following video content, generate a detailed recipe. "
        "Return a SINGLE JSON object. Do not include any text outside the JSON object.\n\n"
        "**JSON Structure Requirements:**\n"
                 "- `title`: string (Recipe title)\n"
        "- `description`: string (A brief, engaging summary)\n"
        "- `prepTime`: integer (Preparation time in minutes)\n"
        "- `cookTime`: integer (Cooking time in minutes)\n"
        "- `difficulty`: string ('Easy', 'Medium', or 'Hard')\n"
        "- `nutrition`: object with keys `calories`, `protein`, `carbs`, `fats`, `portions` (all integers, estimate if not mentioned, use null if unknown)\n"
        "- `ingredients`: array of objects. Each object must have:\n"
        "  - `name`: string (The full ingredient text, e.g., '1 cup of basmati rice')\n"
        "  - `category`: string (Must be one of: " + categories_list + ")\n"
        "- `steps`: array of strings (Cooking instructions)\n\n"
        f"**Video Content:**\n"
        f"POST_CAPTION:\n{caption}\n\n"
        f"CLOSED_CAPTIONS:\n{srt_text}\n\n"
        f"SPEECH_TRANSCRIPT:\n{speech}"
    )
    for t in (0.1, 0.5): # Try with low temperature first for accuracy
        try:
            data = gpt_json(prompt, t)
            if data.get("ingredients") and data.get("steps") and data.get("title"):
                data["imageUrl"] = thumbnail_url or ""
                data["id"] = str(uuid.uuid4())
                data["isFromReel"] = True
                data["extractedFrom"] = platform or "website"
                data["creatorHandle"] = creator_handle or None
                data["creatorName"] = creator_name or None
                data["createdAt"] = datetime.utcnow().isoformat()
                
                # Validate with Pydantic to ensure full compliance before returning
                Recipe.model_validate(data)
                return data
        except Exception as e:
            print(f"GPT extraction attempt failed with temp={t}. Error: {e}")
            continue

    # Fallback if GPT fails
    return {
        "id": str(uuid.uuid4()),
        "title": "Imported Recipe",
        "description": "Could not automatically extract recipe details from the video.",
        "imageUrl": thumbnail_url or "",
        "prepTime": 10, "cookTime": 20, "difficulty": "Medium",
        "nutrition": {"calories": None, "protein": None, "carbs": None, "fats": None, "portions": 2},
        "ingredients": [{
            "name": "Please add ingredients manually.",
            "category": "Other",
        }],
        "steps": ["Could not extract steps. Please add them manually."],
        "isFromReel": True,
        "extractedFrom": platform or "website",
        "creatorHandle": creator_handle or None,
        "creatorName": creator_name or None,
        "createdAt": datetime.utcnow().isoformat()
    }

async def parse_steps_async(steps: List[str]) -> List[Dict[str, str]]:
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    prompt = (
        'Return ONE JSON object: {"steps": [...]}. '
        'Each item in the array must have these keys: "verb", "mainIngredient", "vessel", "visible_change". '
        "Values should be 3 words or less. Array length must match the number of instructions.\n\n"
        "INSTRUCTIONS:\n" + joined
    )
    rsp = await aclient.chat.completions.create(
        model=CHAT_MODEL, temperature=0.1, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON matching the user's schema."},
            {"role": "user",   "content": prompt}
        ]
    )
    try:
        data = json.loads(rsp.choices[0].message.content)
        items = data.get("steps", [])
        if not isinstance(items, list) or len(items) != len(steps):
            raise ValueError("Parsed steps do not match expected structure.")
        return items
    except (json.JSONDecodeError, ValueError):
        return [{"verb": "prepare", "mainIngredient": "ingredients", "vessel": "vessel", "visible_change": "ready"} for _ in steps]

async def generate_step_image(idx: int, comp: Dict[str, str]) -> Dict[str, str]:
    prompt = (
        f"High-resolution food photography, cinematic color grade. "
        f"Scene: Step {idx}. {comp['verb'].capitalize()} {comp['mainIngredient']} "
        f"in {comp['vessel']}; showing {comp['visible_change']}. "
        "Camera: 50 mm prime lens, f/2.8. Angle: top-down 90°. "
        "Surface: rustic dark-oak board. Props: wooden spoon, ceramic ramekin of herbs. "
        "Lighting: soft window light from left. Aspect: 1:1. "
        "Negative prompts: no hands, no faces, no brand logos, no text."
    )
    rsp = await aclient.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1, response_format="b64_json")
    if not rsp.data or not rsp.data[0].b64_json:
        raise ValueError("Image generation failed, no b64_json data returned.")
    image_filename = f"{uuid.uuid4()}.png"
    fname = USER_UPLOADS_DIR / image_filename
    async with aiofiles.open(fname, "wb") as f:
        await f.write(base64.b64decode(rsp.data[0].b64_json))
    url = f"{BASE_URL}/images/{image_filename}"
    return {"step_number": idx, "image_url": url}

def _extract_recipe_from_url_sync(link: str, tmpdir: str) -> Dict:
    """Synchronous helper for video processing to run in a thread."""
    tmp = Path(tmpdir)
    info = run_yt_dlp(link, tmp)
    cap = info["caption"]
    srt = srt_to_text(info["subs"]) if info["subs"] else ""
    speech = transcribe(info["audio"]) if info["audio"] else ""
    thumbnail_url = info["thumbnail_url"]
    platform = info["platform"]
    creator_handle = info["creator_handle"]
    creator_name = info["creator_name"]
    recipe = extract_recipe(cap, srt, speech, thumbnail_url, platform, creator_handle, creator_name)
    return recipe

# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/import-recipe")
async def import_recipe(req: Request):
    link = (await req.json()).get("link", "").strip()
    if not link:
        raise HTTPException(400, "link is required")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            recipe = await asyncio.to_thread(_extract_recipe_from_url_sync, link, tmpdir)
            return {"success": True, "recipe": recipe, "source": "yt_dlp"}
    except Exception as e:
        print(f"yt_dlp failed: {e}. Falling back to basic scrape…")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(link, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                html = resp.text

            import html as html_lib
            thumb_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            fallback_thumb = html_lib.unescape(thumb_match.group(1)) if thumb_match else ""
            desc_match = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            caption = html_lib.unescape(desc_match.group(1)) if desc_match else "Could not extract description."

            # Determine platform for fallback
            if "instagram.com" in link.lower():
                platform = "instagram"
            elif "tiktok.com" in link.lower():
                platform = "tiktok"
            elif "youtube.com" in link.lower() or "youtu.be" in link.lower():
                platform = "youtube"
            else:
                platform = "website"

            recipe = extract_recipe(caption, "", "", fallback_thumb, platform, "", "")
            return {"success": True, "recipe": recipe, "source": "fallback", "warning": "Used fallback extraction."}
        except Exception as e2:
            print(f"Fallback scrape failed: {e2}")
            raise HTTPException(500, f"Failed to process link: {e}")

@app.post("/generate-step-images")
async def generate_step_images(req: ImageGenerationRequest):
    steps = req.instructions[:MAX_STEPS]
    if not steps:
        raise HTTPException(400, "instructions list is empty")

    comps = await parse_steps_async(steps)
    sem = asyncio.Semaphore(5) # Limit concurrent requests to OpenAI

    async def worker(i, c):
        async with sem:
            return await generate_step_image(i + 1, c)

    tasks = [worker(i, c) for i, c in enumerate(comps)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    good, bad = [], []
    for i, r in enumerate(results):
        if isinstance(r, dict):
            r["step_text"] = steps[i]
            good.append(r)
        else:
            bad.append({"step_number": i + 1, "error": str(r)})
    return {"success": len(bad) == 0, "generated_images": good, "failed_steps": bad}

async def analyze_recipe_general_health(recipe: Recipe) -> Dict:
    """Analyzes a recipe for general health without personal data."""
    ingredients_list = [ing.name for ing in recipe.ingredients]
    
    prompt = f"""
Analyze this recipe for general health. Return ONLY valid JSON with this exact structure:

{{
  "overall_score": 85,
  "risk_level": "low",
  "personal_message": "This recipe is generally healthy...",
  "main_culprits": [
    {{"ingredient": "honey", "impact": "high sugar content", "severity": "medium"}}
  ],
  "health_boosters": [
    {{"ingredient": "garlic", "impact": "anti-inflammatory properties", "severity": "high"}}
  ],
  "recommendations": {{
    "should_avoid": false,
    "modifications": ["Reduce honey by half", "Add more vegetables"],
    "alternative_recipes": ["Grilled chicken with steamed vegetables"]
  }},
  "blood_markers_affected": []
}}

RECIPE: Name: {recipe.title}
Ingredients: {', '.join(ingredients_list)}

Provide realistic health analysis with scores 1-100, risk_level as "low"/"medium"/"high", and practical recommendations.
"""
    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": "You are a nutrition expert. Return only valid JSON matching the exact structure provided."}, {"role": "user", "content": prompt}],
            temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"General health analysis error: {e}")
        return {
            "overall_score": 70,
            "risk_level": "medium",
            "personal_message": "Could not complete general analysis.",
            "main_culprits": [],
            "health_boosters": [],
            "recommendations": {
                "should_avoid": False,
                "modifications": [],
                "alternative_recipes": []
            },
            "blood_markers_affected": []
        }

@app.post("/analyze-health-impact", response_model=HealthAnalysisResponse)
async def analyze_health_impact(request: HealthAnalysisRequest):
    try:
        # For now, we'll just do general health analysis since blood test functionality was removed
        analysis_result = await analyze_recipe_general_health(request.recipe)
        return HealthAnalysisResponse(success=True, analysis=analysis_result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Health analysis error: {e}")
        raise HTTPException(500, f"Failed to analyze health impact: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print("✅ Backend service is starting.")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)