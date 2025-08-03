# main.py – FastAPI backend
# Instagram Reels & TikTok → recipe JSON + on-demand GPT-4 step-image generation
# deps: openai>=1.21.0 fastapi uvicorn yt-dlp pysrt python-dotenv python-multipart aiofiles httpx

import os, json, time, tempfile, asyncio, base64, uuid, random
from pathlib import Path
from typing import List, Dict, Union, Optional
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

import yt_dlp, pysrt, aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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

CHAT_MODEL = "gpt-4.1"
IMAGE_MODEL = "gpt-image-1"
MAX_STEPS = 10

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
USER_UPLOADS_DIR = Path("/tmp/user_uploads")
USER_UPLOADS_DIR.mkdir(exist_ok=True)
# --- End Configuration ---

@asynccontextmanager
async def lifespan(app: FastAPI):
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

# ----------------- Instagram handle extraction utilities -----------------

_ALLOWED = re.compile(r"^[a-z0-9._]{2,30}$")

def normalize_handle(s: str) -> str:
    """
    Lowercase, strip @, slashes, spaces, and pipes. Keep only [a-z0-9._].
    Require at least one letter. Return '' if invalid.
    """
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("@", "").replace(" ", "").replace("|", "").strip("/")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._")
    
    if not _ALLOWED.fullmatch(s):
        return ""
    if s.isdigit():  # reject pure numeric ids like 52845553550
        return ""
    if not re.search(r"[a-z]", s):  # must contain a letter
        return ""
    
    return s

def extract_from_uploader_url(uploader_url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(uploader_url)
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            return normalize_handle(parts[0])
    except Exception:
        pass
    return ""

def public_oembed_handle_sync(url: str) -> str:
    try:
        resp = httpx.get(
            "https://www.instagram.com/oembed/",
            params={"url": url, "omitscript": "true"},
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        data = resp.json()
        author_url = data.get("author_url") or ""
        if author_url:
            return extract_from_uploader_url(author_url)
    except Exception:
        pass
    return ""

def graph_oembed_handle_sync(url: str) -> str:
    token = os.getenv("INSTAGRAM_OEMBED_ACCESS_TOKEN", "").strip()
    if not token:
        return ""
    try:
        resp = httpx.get(
            "https://graph.facebook.com/v16.0/instagram_oembed",
            params={"url": url, "omitscript": "true", "access_token": token},
            timeout=6
        )
        resp.raise_for_status()
        data = resp.json()
        author_url = data.get("author_url") or ""
        if author_url:
            return extract_from_uploader_url(author_url)
    except Exception:
        pass
    return ""

def scrape_instagram_page_handle_sync(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=8, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        # Multiple approaches to find the username
        patterns_to_try = [
            # Look for "username":"..." in JSON
            r'"username"\s*:\s*"([A-Za-z0-9._]+)"',
            # Look for @username in title or meta tags  
            r'<title[^>]*>.*?@([A-Za-z0-9._]+)',
            r'content="[^"]*@([A-Za-z0-9._]+)[^"]*"',
            # Look for profile links
            r'href="https://www\.instagram\.com/([A-Za-z0-9._]+)/"',
            r'/([A-Za-z0-9._]+)/\?' '"[^"]*profile',
            # Look for og:url
            r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']https://www\.instagram\.com/([A-Za-z0-9._]+)',
        ]

        for pattern in patterns_to_try:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                h = normalize_handle(match)
                if h:
                    return h

        # JSON-LD blocks: author.alternateName or author.name
        for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL | re.IGNORECASE):
            try:
                block = m.group(1)
                data = json.loads(block.strip())
                objs = data if isinstance(data, list) else [data]
                for obj in objs:
                    author = obj.get("author")
                    if isinstance(author, dict):
                        cand = author.get("alternateName") or author.get("name") or ""
                        h = normalize_handle(cand)
                        if h:
                            return h
                    elif isinstance(author, list):
                        for a in author:
                            cand = a.get("alternateName") or a.get("name") or ""
                            h = normalize_handle(cand)
                            if h:
                                return h
            except Exception:
                continue

    except Exception:
        pass
    return ""

def best_effort_instagram_handle(url: str, info: dict) -> str:
    """
    Resolution order:
      1) uploader_url/channel_url
      2) public oEmbed
      3) Graph oEmbed
      4) HTML scrape
      5) uploader_id/channel_id only if looks like a real handle (not digits-only)
    Returns normalized handle without '@'.
    """
    for key in ("uploader_url", "channel_url"):
        h = extract_from_uploader_url(info.get(key, "") or "")
        if h:
            return h

    h = public_oembed_handle_sync(url)
    if h:
        return h

    h = graph_oembed_handle_sync(url)
    if h:
        return h

    h = scrape_instagram_page_handle_sync(url)
    if h:
        return h

    for key in ("uploader_id", "channel_id"):
        h = normalize_handle(info.get(key, "") or "")
        if h:
            return h

    return ""

# ----------------- Pydantic Models -----------------
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
    prepTime: int
    cookTime: int
    difficulty: str
    nutrition: NutritionInfo
    ingredients: List[CategorizedIngredient]
    steps: List[str]
    isFromReel: bool = True
    extractedFrom: str
    creatorHandle: Optional[str] = None
    creatorName: Optional[str] = None
    originalUrl: Optional[str] = Field(None, description="Original URL used to import this recipe")
    createdAt: Union[str, datetime] = Field(default_factory=datetime.utcnow)

    @field_validator('createdAt', mode='before')
    @classmethod
    def parse_created_at(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str):
            return v
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

# ----------------- Core Logic -----------------
def run_yt_dlp(url: str, dst: Path) -> dict:
    out = dict(audio=None, subs=None, thumb=None, caption="", thumbnail_url="", platform="", creator_handle="", creator_name="")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "srt",
        "writethumbnail": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True,
        "no_warnings": True
    }

    if "tiktok.com" in url.lower() or "instagram.com" in url.lower():
        opts["http_headers"] = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
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

    if info.get("thumbnails"):
        out["thumbnail_url"] = info["thumbnails"][-1]["url"]
    else:
        out["thumbnail_url"] = info.get("thumbnail", "")

    if "instagram.com" in url.lower():
        out["platform"] = "instagram"
    elif "tiktok.com" in url.lower():
        out["platform"] = "tiktok"
    elif "youtube.com" in url.lower():
        out["platform"] = "youtube"
    else:
        out["platform"] = "website"

    creator_name = info.get("uploader", "") or info.get("channel", "")
    creator_handle = ""
    if "instagram.com" in url.lower():
        creator_handle = best_effort_instagram_handle(url, info)
    else:
        creator_handle = normalize_handle(info.get("uploader_id", "") or info.get("channel_id", ""))

    # Try to salvage from display name like "Foo @bar"
    if not creator_handle:
        m = re.search(r'@([A-Za-z0-9._]+)', creator_name or "")
        if m:
            creator_handle = normalize_handle(m.group(1))

    # Add @ prefix to creator_handle if it exists
    if creator_handle:
        out["creator_handle"] = f"@{creator_handle}"
    else:
        out["creator_handle"] = ""
    
    out["creator_name"] = creator_name or (creator_handle if creator_handle else "")

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

def _clamp_time(value, default, lo, hi) -> int:
    try:
        v = int(value)
        if v <= 0:  # Explicitly handle 0 and negative values
            return default
    except Exception:
        v = default
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v

def extract_explicit_times(text: str) -> tuple:
    """Extract explicit time mentions from recipe text."""
    import re
    
    text_lower = text.lower()
    max_prep_time = 0
    max_cook_time = 0
    
    # More comprehensive time patterns
    time_patterns = [
        # Hours patterns - more flexible
        (r'(\d+(?:\.\d+)?)\s*(?:to|-)?\s*(\d+(?:\.\d+)?)?\s*hours?', 'hours'),
        (r'(\d+)\s*hrs?', 'hours'),
        # Minutes patterns - more flexible  
        (r'(\d+(?:\.\d+)?)\s*(?:to|-)?\s*(\d+(?:\.\d+)?)?\s*(?:mins?|minutes?)', 'minutes'),
        # Special slow cooker patterns
        (r'slow\s+cook(?:er)?.*?(\d+)\s*hours?', 'hours'),
        (r'cook.*?(?:for|about).*?(\d+)\s*hours?', 'hours'),
        (r'(\d+)\s*hours?\s*(?:\((\d+)\s*minutes?\))?', 'hours_with_minutes'),
    ]
    
    for pattern, unit_type in time_patterns:
        matches = re.findall(pattern, text_lower)
        
        for match in matches:
            try:
                if isinstance(match, tuple):
                    if unit_type == 'hours_with_minutes' and len(match) == 2:
                        # Handle "4 hours (240 minutes)" format
                        hours = int(float(match[0])) if match[0] else 0
                        extra_minutes = int(float(match[1])) if match[1] else 0
                        minutes = hours * 60 + extra_minutes
                    elif len(match) == 2 and match[1]:
                        # Handle ranges like "2-3 hours"
                        if unit_type == 'hours':
                            minutes = int(float(match[1])) * 60
                        else:
                            minutes = int(float(match[1]))
                    else:
                        # Single value
                        time_val = match[0] if match[0] else match[1] if len(match) > 1 else '0'
                        if unit_type == 'hours':
                            minutes = int(float(time_val)) * 60
                        else:
                            minutes = int(float(time_val))
                else:
                    # Single match
                    if unit_type == 'hours':
                        minutes = int(float(match)) * 60
                    else:
                        minutes = int(float(match))
                
                # Determine if this is prep or cook time based on context
                match_pos = text_lower.find(str(match[0]) if isinstance(match, tuple) else str(match))
                context_start = max(0, match_pos - 150)
                context_end = min(len(text_lower), match_pos + 150)
                context = text_lower[context_start:context_end]
                
                # Better context classification with priority rules
                # FIRST: Exclude marination times completely
                if any(word in context for word in ['marinate', 'marinade', 'marinating', 'marinated']):
                    continue  # Skip marination times entirely
                elif minutes >= 120:  # 2+ hours is almost always cook time
                    max_cook_time = max(max_cook_time, minutes)
                elif any(word in context for word in ['slow cooker', 'slow cook', 'crockpot', 'cook for', 'bake', 'roast', 'simmer', 'boil', 'fry']):
                    max_cook_time = max(max_cook_time, minutes)
                elif any(word in context for word in ['prep', 'prepare', 'chop', 'dice', 'slice']):
                    max_prep_time = max(max_prep_time, minutes)
                else:
                    # For ambiguous short times, default to cook time
                    if minutes > 60:
                        max_cook_time = max(max_cook_time, minutes)
                    else:
                        max_prep_time = max(max_prep_time, minutes)
                    
            except (ValueError, IndexError) as e:
                continue
    
    return max_prep_time, max_cook_time

def smart_timing_fallback(ingredients: list, steps: list, recipe_text: str) -> tuple:
    """Calculate realistic prep and cook times based on recipe complexity."""
    
    # First, try to extract explicit time mentions
    explicit_prep, explicit_cook = extract_explicit_times(recipe_text)
    
    # Base times
    prep_time = max(5, explicit_prep) if explicit_prep > 0 else 5
    cook_time = max(10, explicit_cook) if explicit_cook > 0 else 10
    
    # If we found explicit times, use them as base and add complexity
    if explicit_cook > 0:
        cook_time = explicit_cook
    if explicit_prep > 0:
        prep_time = explicit_prep
    
    # Analyze ingredients for prep complexity only if no explicit prep time
    if explicit_prep == 0 and ingredients:
        ingredient_count = len(ingredients)
        prep_time = max(5, min(25, ingredient_count * 2))
        
        # Add time for specific prep-heavy ingredients
        for ing in ingredients:
            ing_text = ing.get("name", "").lower() if isinstance(ing, dict) else str(ing).lower()
            if any(word in ing_text for word in ["onion", "garlic", "ginger"]):
                prep_time += 3
            if any(word in ing_text for word in ["marinate", "marinade"]):
                prep_time += 20
            if any(word in ing_text for word in ["dice", "chop", "slice", "cut"]):
                prep_time += 2
    
    # Analyze steps for cooking complexity only if no explicit cook time
    if explicit_cook == 0 and steps:
        steps_count = len(steps)
        cook_time = max(8, min(45, steps_count * 5))
        
        # Analyze recipe text for cooking methods
        recipe_lower = recipe_text.lower()
        cooking_methods = {
            "slow cook": 240, "slow cooker": 240, "crockpot": 240,  # Default to 4 hours for slow cooker
            "bake": 30, "roast": 35, "oven": 25,
            "grill": 15, "bbq": 15,
            "stir fry": 10, "stir-fry": 10, "saute": 8,
            "boil": 12, "steam": 10,
            "fry": 10, "pan fry": 12,
            "simmer": 20, "braise": 45
        }
        
        for method, time in cooking_methods.items():
            if method in recipe_lower:
                cook_time = max(cook_time, time)
                break
    
    # Final bounds check - but allow longer times for slow cooking
    prep_time = max(3, min(60, prep_time))
    cook_time = max(5, min(480, cook_time))  # Allow up to 8 hours for slow cooking
    
    return prep_time, cook_time

def extract_recipe(caption: str, srt_text: str, speech: str, thumbnail_url: str = "", platform: str = "", creator_handle: str = "", creator_name: str = "", original_url: str = "") -> dict:
    categories_list = ", ".join(f'"{cat.value}"' for cat in IngredientCategory)
    prompt = (
        "Based on the following video content, generate a detailed recipe. "
        "Return a SINGLE JSON object. Do not include any text outside the JSON object.\n\n"
        "🔥 CRITICAL TIMING REQUIREMENTS (NEVER IGNORE):\n"
        "You MUST calculate REALISTIC prep and cook times like a world-class chef:\n"
        "🚫 MARINATION TIME IS NEVER INCLUDED IN PREP OR COOK TIME - ONLY ACTIVE COOKING TIME!\n\n"
        "PREP TIME CALCULATION:\n"
        "- Chopping 1 onion: 1 minute\n"
        "- Mincing garlic: 1 minute per clove\n"
        "- Cutting vegetables: 1 minute per item, limit to 8 minutes total\n"
        "- Cutting meat/protein: 5 minutes\n"
        "- Include marination ingredient prep in 'Prep Time' but NOT actual marinating time\n"
        "ADD UP ALL PREP STEPS = prepTime (minimum 2, maximum 60 minutes)\n\n"
        "COOK TIME CALCULATION:\n"
        "- Sautéing vegetables: 5-8 minutes\n"
        "- Frying chicken/meat: 8-15 minutes\n"
        "- Boiling pasta: 8-12 minutes\n"
        "- Cooking rice: 12-20 minutes\n"
        "- Baking (oven): 20-45 minutes\n"
        "- Slow cooking: 120-360 minutes (2-8 hours)\n"
        "- Grilling: 5-20 minutes\n"
        "- Steaming: 5-15 minutes\n"
        "- Making sauce/dressing/salsa: 3-5 minutes\n"
        "- Do NOT include marination time in Prep or Cook Time calculation\n"
        "- If recipe mentions marinating, mention it in steps but EXCLUDE from timing calculations\n"
        "ADD UP ALL COOKING STEPS = cookTime (minimum 3 minutes)\n\n"
        "COOKING CONTEXT:\n"
        "- If raw meat is added to a broth, it can be boiled in 4-20 minutes dependent on the meat\n"
        "- If we do not supply a specific cut of meat in the ingredient list, do not mention specific cooking times in the instructions, instead supply options or say to cook the meat dependent on the cut\n\n"
        "⚠️ MANDATORY: prepTime and cookTime MUST be positive integers. NEVER null, undefined, or 0!\n\n"
        "JSON Structure Requirements:\n"
        "- `title`: string (Recipe title)\n"
        "- `description`: string (Brief, engaging summary)\n"
        "- `prepTime`: integer (REQUIRED - calculated prep time in minutes)\n"
        "- `cookTime`: integer (REQUIRED - calculated cook time in minutes)\n"
        "- `difficulty`: string ('Easy', 'Medium', 'Hard')\n"
        "- `nutrition`: object with keys `calories`, `protein`, `carbs`, `fats`, `portions` (integers or null)\n"
        "- Every recipe should have a protein, carbohydrates and total fat value which is based on accumulative values of all the ingredients divided by the number of portions\n"
        "- `ingredients`: array of objects with `name` and `category` in [" + categories_list + "]\n"
        "- `steps`: array of strings (Clear cooking instructions)\n\n"
        "TIMING EXAMPLES:\n"
        "- Simple stir-fry: prepTime: 10, cookTime: 14\n"
        "- Pasta with sauce: prepTime: 8, cookTime: 15\n"
        "- Roasted chicken: prepTime: 10, cookTime: 45\n"
        "- Slow cooker meal: prepTime: 20, cookTime: 360\n"
        "- Quick salad: prepTime: 8, cookTime: 3\n\n"
        f"VIDEO CONTENT TO ANALYZE:\n"
        f"POST_CAPTION:\n{caption}\n\n"
        f"CLOSED_CAPTIONS:\n{srt_text}\n\n"
        f"SPEECH_TRANSCRIPT:\n{speech}"
    )
    for t in (0.1, 0.5):
        try:
            data = gpt_json(prompt, t)
            if data.get("ingredients") and data.get("steps") and data.get("title"):
                # ALWAYS check for explicit times and override GPT if found
                ingredients = data.get("ingredients", [])
                steps = data.get("steps", [])
                recipe_text = f"{caption} {srt_text} {speech}"
                
                # Include steps text in the analysis
                if steps:
                    recipe_text += " " + " ".join(steps)
                
                smart_prep, smart_cook = smart_timing_fallback(ingredients, steps, recipe_text)
                
                # Use explicit times if found, otherwise use GPT + smart fallback
                explicit_prep, explicit_cook = extract_explicit_times(recipe_text)
                
                final_prep = explicit_prep if explicit_prep > 0 else _clamp_time(data.get("prepTime"), default=smart_prep, lo=3, hi=60)
                final_cook = explicit_cook if explicit_cook > 0 else _clamp_time(data.get("cookTime"), default=smart_cook, lo=5, hi=480)
                
                data["prepTime"] = final_prep
                data["cookTime"] = final_cook

                data["imageUrl"] = thumbnail_url or ""
                data["id"] = str(uuid.uuid4())
                data["isFromReel"] = True
                data["extractedFrom"] = platform or "website"
                data["creatorHandle"] = creator_handle or None
                data["creatorName"] = creator_name or None
                data["originalUrl"] = original_url or None
                data["createdAt"] = datetime.utcnow().isoformat()
                Recipe.model_validate(data)
                return data
        except Exception as e:
            print(f"GPT extraction attempt failed with temp={t}. Error: {e}")
            continue

    # hard fallback if GPT fails completely
    recipe_text = f"{caption} {srt_text} {speech}"
    fallback_prep, fallback_cook = smart_timing_fallback([], [], recipe_text)
    
    return {
        "id": str(uuid.uuid4()),
        "title": "Imported Recipe",
        "description": "Could not automatically extract recipe details from the video.",
        "imageUrl": thumbnail_url or "",
        "prepTime": fallback_prep,
        "cookTime": fallback_cook,
        "difficulty": "Easy",
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
        "originalUrl": original_url or None,
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
        model=CHAT_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON matching the user's schema."},
            {"role": "user", "content": prompt}
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
    tmp = Path(tmpdir)
    info = run_yt_dlp(link, tmp)
    cap = info["caption"]
    srt = srt_to_text(info["subs"]) if info["subs"] else ""
    speech = transcribe(info["audio"]) if info["audio"] else ""
    thumbnail_url = info["thumbnail_url"]
    platform = info["platform"]
    creator_handle = info["creator_handle"]
    creator_name = info["creator_name"]
    recipe = extract_recipe(cap, srt, speech, thumbnail_url, platform, creator_handle, creator_name, link)
    return recipe

# ----------------- API Endpoints -----------------
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
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client_:
                resp = await client_.get(link, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                html = resp.text

            import html as html_lib
            thumb_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            fallback_thumb = html_lib.unescape(thumb_match.group(1)) if thumb_match else ""
            desc_match = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            caption = html_lib.unescape(desc_match.group(1)) if desc_match else "Could not extract description."

            if "instagram.com" in link.lower():
                platform = "instagram"
            elif "tiktok.com" in link.lower():
                platform = "tiktok"
            elif "youtube.com" in link.lower() or "youtu.be" in link.lower():
                platform = "youtube"
            else:
                platform = "website"

            handle = public_oembed_handle_sync(link) or graph_oembed_handle_sync(link) or scrape_instagram_page_handle_sync(link)

            recipe = extract_recipe(caption, "", "", fallback_thumb, platform, handle, "", link)
            return {"success": True, "recipe": recipe, "source": "fallback", "warning": "Used fallback extraction."}
        except Exception as e2:
            print(f"Fallback scrape failed: {e2}")
            raise HTTPException(500, f"Failed to process link: {e2}")

@app.post("/generate-step-images")
async def generate_step_images(req: ImageGenerationRequest):
    steps = req.instructions[:MAX_STEPS]
    if not steps:
        raise HTTPException(400, "instructions list is empty")

    comps = await parse_steps_async(steps)
    sem = asyncio.Semaphore(5)

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
            temperature=0.3,
            response_format={"type": "json_object"}
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