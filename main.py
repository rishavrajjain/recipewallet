# main.py – FastAPI backend (Updated for Render Testing)
# Reels, TikTok, Website → recipe JSON + on-demand GPT-4.1 step-image generation
# User Info → kitchen photos + blood test PDF upload handling
# deps: openai>=1.21.0 fastapi uvicorn yt-dlp pysrt python-dotenv python-multipart aiofiles beautifulsoup4 lxml

import os, json, time, tempfile, asyncio, base64, uuid, random, html
from pathlib import Path
from typing import List, Dict, Union
from contextlib import asynccontextmanager
from datetime import datetime

import yt_dlp, pysrt, aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, validator
import httpx
import re, urllib.parse
from bs4 import BeautifulSoup


# --- Configuration ---
load_dotenv()
client  = OpenAI()
aclient = AsyncOpenAI()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

CHAT_MODEL  = "gpt-4.1"
IMAGE_MODEL = "gpt-image-1"
MAX_STEPS   = 10

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
    print(f"Request body: {await request.body()}")
    print(f"Validation errors: {exc.errors()}")
    return await request_validation_exception_handler(request, exc)

app.mount("/images", StaticFiles(directory=USER_UPLOADS_DIR), name="images")

# --- Pydantic Models ---
class ImageGenerationRequest(BaseModel):
    instructions: List[str]
    recipe_title: str = "Recipe"

class IngredientItem(BaseModel):
    name: str
    imageUrl: str

class Recipe(BaseModel):
    id: str
    name: str
    description: str
    imageUrl: str
    ingredients: List[str]
    cookTime: int
    isFromReel: bool
    steps: List[str]
    createdAt: Union[str, datetime]

    @validator('createdAt', pre=True)
    def parse_created_at(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, (int, float)):
            # Handle timestamp
            return datetime.fromtimestamp(v).isoformat()
        elif isinstance(v, str):
            return v
        else:
            return str(v)

class HealthAnalysisRequest(BaseModel):
    recipe: Recipe
    blood_test_id: str = None  # Make optional
    include_blood_test: bool = False  # Flag to control blood test analysis

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

class BloodMarkerImpact(BaseModel):
    marker: str
    current_level: float
    predicted_impact: str
    target_range: str

class IngredientImpact(BaseModel):
    ingredient: str
    impact: str
    severity: str

class HealthRecommendations(BaseModel):
    should_avoid: bool
    modifications: List[str]
    alternative_recipes: List[str] = []

class HealthAnalysisResponse(BaseModel):
    success: bool
    analysis: Dict = {}
# --- End Pydantic Models ---


# --- Core Logic Functions ---
def run_yt_dlp(url: str, dst: Path) -> dict:
    """Universal video downloader with Instagram-specific workarounds."""
    out = dict(audio=None, subs=None, thumb=None, caption="", thumbnail_url="")
    
    # Instagram-specific strategies to bypass rate limiting
    instagram_strategies = [
        # Strategy 1: Use different user agents and headers
        {
            "format": "bestaudio/best",
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "hi", ""],
            "subtitlesformat": "srt",
            "writethumbnail": True,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }],
            "quiet": True, "no_warnings": True
        },
        # Strategy 2: Mobile user agent, no audio processing
        {
            "format": "best",
            "writesubtitles": True,
            "writethumbnail": True,
            "http_headers": {
                "User-Agent": "Instagram 219.0.0.12.117 Android",
            },
            "quiet": True, "no_warnings": True
        },
        # Strategy 3: Minimal extraction with different extractor
        {
            "format": "worst",
            "writesubtitles": False,
            "writethumbnail": True,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            },
            "quiet": True, "no_warnings": True
        },
        # Strategy 4: Info only (metadata extraction)
        {
            "skip_download": True,
            "writesubtitles": False,
            "writethumbnail": False,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            "quiet": True, "no_warnings": True
        }
    ]
    
    # Use Instagram strategies if it's an Instagram URL
    is_instagram = "instagram.com" in url.lower()
    strategies = instagram_strategies if is_instagram else [instagram_strategies[0]]  # Use first strategy for non-Instagram
    
    for i, opts in enumerate(strategies, 1):
        try:
            print(f"🎬 DEBUG: Trying {'Instagram' if is_instagram else 'video'} strategy {i}/{len(strategies)}")
            opts["outtmpl"] = str(dst / "%(id)s.%(ext)s")
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=not opts.get("skip_download", False))
            
            if not info:
                print(f"⚠️ DEBUG: Strategy {i} returned no info")
                continue
                
            print(f"✅ DEBUG: Strategy {i} succeeded, got video info")
            
            # Extract files if downloaded
            if not opts.get("skip_download", False):
                base = dst / info["id"]
                out["audio"] = next(base.parent.glob(f"{info['id']}.mp3"), None)
                out["subs"] = next(base.parent.glob(f"{info['id']}*.srt"), None)
                out["thumb"] = next(base.parent.glob(f"{info['id']}.jpg"), None) or next(base.parent.glob(f"{info['id']}.webp"), None)
            
            # Extract metadata (always available)
            out["caption"] = (info.get("description") or info.get("title") or "").strip()
            out["thumbnail_url"] = info.get("thumbnail", "")
            
            print(f"🎬 DEBUG: Extracted - Caption: {len(out['caption'])} chars, Thumbnail: {'✅' if out['thumbnail_url'] else '❌'}")
            return out
            
        except Exception as e:
            print(f"⚠️ DEBUG: Strategy {i} failed: {str(e)}")
            if i == len(strategies):  # Last strategy failed
                raise e
            continue
    
    raise Exception("All yt-dlp strategies failed")

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

def gpt_json(prompt: str, temp: float) -> dict:
    rsp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temp,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user",   "content": prompt}
        ]
    )
    return json.loads(rsp.choices[0].message.content)

def extract_recipe(caption: str, srt_text: str, speech: str, thumbnail_url: str = "") -> dict:
    # Combine all available text
    all_text = f"{caption}\n{srt_text}\n{speech}".strip()
    
    if not all_text or len(all_text) < 10:
        print(f"⚠️ DEBUG: Insufficient text for recipe extraction: {len(all_text)} chars")
        return {}
    
    print(f"🤖 DEBUG: Sending to GPT - Total text: {len(all_text)} chars")
    
    prompt = (
        "Extract a recipe from the following content. Return JSON with keys: title, description (1 sentence), "
        "ingredients (list of strings), steps (list of strings).\n\n"
        "If the content doesn't contain a clear recipe, try to infer one from cooking-related content.\n\n"
        f"CONTENT:\n{all_text[:10000]}"  # Limit to 10k chars to avoid token limits
    )
    
    # Try multiple temperatures for better success rate
    for t in (0.1, 0.3, 0.5):
        try:
            print(f"🤖 DEBUG: Trying GPT with temperature {t}")
            data = gpt_json(prompt, t)
            
            if data and isinstance(data, dict):
                # Validate required fields
                ingredients = data.get("ingredients", [])
                steps = data.get("steps", [])
                
                if ingredients and steps and len(ingredients) > 0 and len(steps) > 0:
                    data["thumbnailUrl"] = thumbnail_url
                    data.setdefault("cookTime", 0)
                    print(f"✅ DEBUG: GPT extraction successful - {len(ingredients)} ingredients, {len(steps)} steps")
                    return data
                else:
                    print(f"⚠️ DEBUG: GPT returned incomplete recipe - ingredients: {len(ingredients)}, steps: {len(steps)}")
            else:
                print(f"⚠️ DEBUG: GPT returned invalid data type: {type(data)}")
                
        except Exception as e:
            print(f"❌ DEBUG: GPT extraction failed at temp {t}: {str(e)}")
            continue
    
    print(f"💥 DEBUG: All GPT extraction attempts failed")
    return {} # Return empty dict on failure

async def extract_instagram_fallback(url: str, client: httpx.AsyncClient) -> dict:
    """Fallback Instagram extraction when yt-dlp fails."""
    try:
        print(f"📱 DEBUG: Attempting Instagram fallback extraction")
        
        # Try multiple user agents
        user_agents = [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ]
        
        for ua in user_agents:
            try:
                headers = {
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
                
                response = await client.get(url, headers=headers, timeout=15, follow_redirects=True)
                if response.status_code == 200:
                    html = response.text
                    
                    # Extract basic metadata from HTML
                    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
                    description_match = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                    image_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                    
                    title = html.unescape(title_match.group(1)) if title_match else "Instagram Recipe"
                    description = html.unescape(description_match.group(1)) if description_match else ""
                    thumbnail = html.unescape(image_match.group(1)) if image_match else ""
                    
                    if description and len(description) > 20:
                        print(f"📱 DEBUG: Instagram fallback found content: {len(description)} chars")
                        # Use GPT to extract recipe from the description
                        return extract_recipe("", "", description, thumbnail)
                    
            except Exception as e:
                print(f"📱 DEBUG: Instagram fallback attempt failed with {ua[:20]}...: {str(e)}")
                continue
        
        return {}
    except Exception as e:
        print(f"📱 DEBUG: Instagram fallback completely failed: {str(e)}")
        return {}

async def extract_recipe_from_spoonacular(url: str, client: httpx.AsyncClient) -> dict:
    """Extracts a recipe using the Spoonacular API."""
    # Skip Spoonacular for Instagram URLs (it doesn't work)
    if "instagram.com" in url.lower():
        print(f"🥄 DEBUG: Skipping Spoonacular for Instagram URL")
        return None
        
    if not SPOONACULAR_API_KEY or SPOONACULAR_API_KEY == "YOUR_API_KEY_HERE":
        return None
    api_url = "https://api.spoonacular.com/recipes/extract"
    params = {"url": url, "apiKey": SPOONACULAR_API_KEY}
    try:
        response = await client.get(api_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data.get("extendedIngredients") or not data.get("analyzedInstructions"):
            return None

        return {
            "title": data.get("title", "Imported Recipe"),
            "description": re.sub('<[^<]+?>', '', data.get("summary", "Recipe from Website")).split('.')[0],
            "thumbnailUrl": data.get("image", ""),
            "ingredients": [ing.get("original") for ing in data.get("extendedIngredients", [])],
            "cookTime": data.get("readyInMinutes", 0),
            "steps": [step.get("step") for inst in data.get("analyzedInstructions", []) for step in inst.get("steps", [])]
        }
    except Exception as e:
        print(f"Spoonacular API call failed for {url}: {e}")
        return None

def extract_recipe_from_html(html_content: str) -> dict:
    """Tries to extract a recipe from HTML, prioritizing JSON-LD, then falls back to GPT."""
    soup = BeautifulSoup(html_content, "lxml")
    
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            graph = data.get("@graph", [data]) 
            for item in graph:
                item_type = item.get("@type", "")
                is_recipe = item_type == "Recipe" or (isinstance(item_type, list) and "Recipe" in item_type)
                if is_recipe:
                    steps_raw = item.get("recipeInstructions", [])
                    steps = [re.sub('<[^<]+?>', '', step.get("text", step)) for step in steps_raw if step]

                    image = item.get("image", "")
                    img_url = ""
                    if isinstance(image, dict): img_url = image.get("url", "")
                    elif isinstance(image, list) and image: img_url = image[0]
                    else: img_url = image

                    return {
                        "title": item.get("name", "Imported Recipe"),
                        "description": item.get("description", "Recipe from Website"),
                        "thumbnailUrl": img_url,
                        "ingredients": item.get("recipeIngredient", []),
                        "steps": steps,
                        "cookTime": 0
                    }
        except Exception:
            continue

    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    text_content = ' '.join(main_content.get_text(separator=' ', strip=True).split()) if main_content else ""
    og_image = soup.find("meta", property="og:image")
    thumb_url = og_image["content"] if og_image else ""
    
    if not text_content: return {}
    
    # Fallback to GPT for unstructured text
    return extract_recipe(caption="", srt_text="", speech=text_content[:15000], thumbnail_url=thumb_url)


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
            {"role": "user",   "content": prompt}
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

async def generate_step_image(idx: int, comp: Dict[str, str]) -> Dict[str, str]:
    # --- Dynamic prompt generation with varied props for realism & wow-factor ---
    prop_choices = [
        "wooden spoon and vintage measuring cups", "ceramic ramekin of chopped fresh herbs",
        "small glass bowl of colorful spices", "tiny jug of extra-virgin olive oil",
        "marble mortar and pestle with crushed pepper", "sprig of fresh rosemary on the side",
        "chef\'s knife with patina finish", "linen napkin and copper spoon",
        "cast-iron skillet handle peeking in",
    ]
    chosen_props = random.choice(prop_choices)

    prompt = (
        f"High-resolution food photography, cinematic color grade.\n"
        f"Scene: Step {idx}. {comp['verb'].capitalize()} {comp['mainIngredient']} "
        f"in {comp['vessel']}; showing {comp['visible_change']}.\n"
        "Camera: 50 mm prime lens, f/2.8, ISO 200, 1/125 s. "
        "Angle: top-down 90°. Surface: rustic dark-oak board. "
        f"Props: {chosen_props}. "
        "Lighting: soft window light from left, gentle natural shadows. "
        "Aspect: 1:1. Negative: hands, faces, brand logos, text."
    )

    rsp = await aclient.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="b64_json"
    )

    if not rsp.data[0].b64_json:
        raise ValueError("Image generation failed, no b64_json data returned.")

    image_filename = f"{uuid.uuid4()}.png"
    fname = USER_UPLOADS_DIR / image_filename
    
    async with aiofiles.open(fname, "wb") as f:
        await f.write(base64.b64decode(rsp.data[0].b64_json))
    
    url = f"{BASE_URL}/images/{image_filename}"

    return {"step_number": idx, "image_url": url}

# ── Health Analysis Functions ─────────────────────────────────────────
def parse_blood_test_data(blood_test_id: str) -> Dict:
    """Parse blood test PDF and extract key health markers"""
    # In production, this would use OCR/PDF parsing
    # For now, return mock data based on common blood test markers
    return {
        "cholesterol": {
            "total": 220, "ldl": 140, "hdl": 45, "triglycerides": 180,
            "target_ranges": { "total": "< 200 mg/dL", "ldl": "< 100 mg/dL", "hdl": "> 40 mg/dL (men), > 50 mg/dL (women)", "triglycerides": "< 150 mg/dL" }
        },
        "blood_sugar": {
            "fasting_glucose": 105, "hba1c": 5.8,
            "target_ranges": { "fasting_glucose": "70-99 mg/dL", "hba1c": "< 5.7%" }
        },
        "vitamins": {
            "vitamin_d": 18, "b12": 250,
            "target_ranges": { "vitamin_d": "30-100 ng/mL", "b12": "200-900 pg/mL" }
        },
        "minerals": {
            "iron": 85, "calcium": 9.5,
            "target_ranges": { "iron": "60-170 mcg/dL", "calcium": "8.5-10.5 mg/dL" }
        }
    }

def get_recipe_data(recipe_id: str) -> Dict:
    """Get recipe data - in production this would query a database"""
    # Mock recipe data for testing
    return {
        "title": "Butter Chicken with Rice",
        "ingredients": [ "2 lbs chicken thighs", "1 cup heavy cream", "4 tbsp butter", "2 tbsp ghee", "1 cup basmati rice", "2 tbsp sugar", "1 tsp salt", "Tomato sauce", "Garam masala", "Ginger garlic paste" ],
        "nutrition_per_serving": { "calories": 650, "saturated_fat": 25, "cholesterol": 120, "sodium": 890, "carbs": 45, "sugar": 12, "protein": 35 }
    }

async def analyze_health_impact_ai(recipe_data: Dict, blood_data: Dict) -> Dict:
    """Use OpenAI to analyze health impact of recipe on user's blood markers"""
    prompt = f"""
You are a medical nutrition expert analyzing how a specific recipe will impact someone's blood test results.

BLOOD TEST RESULTS:
- Total Cholesterol: {blood_data['cholesterol']['total']} mg/dL (target: {blood_data['cholesterol']['target_ranges']['total']})
- LDL Cholesterol: {blood_data['cholesterol']['ldl']} mg/dL (target: {blood_data['cholesterol']['target_ranges']['ldl']})
- HDL Cholesterol: {blood_data['cholesterol']['hdl']} mg/dL (target: {blood_data['cholesterol']['target_ranges']['hdl']})
- Triglycerides: {blood_data['cholesterol']['triglycerides']} mg/dL (target: {blood_data['cholesterol']['target_ranges']['triglycerides']})
- Fasting Glucose: {blood_data['blood_sugar']['fasting_glucose']} mg/dL (target: {blood_data['blood_sugar']['target_ranges']['fasting_glucose']})
- HbA1c: {blood_data['blood_sugar']['hba1c']}% (target: {blood_data['blood_sugar']['target_ranges']['hba1c']})
- Vitamin D: {blood_data['vitamins']['vitamin_d']} ng/mL (target: {blood_data['vitamins']['target_ranges']['vitamin_d']})

RECIPE ANALYSIS:
Title: {recipe_data['title']}
Ingredients: {', '.join([str(i) for i in recipe_data['ingredients']])}

Analyze and return JSON with:
{{
  "overall_score": number (0-100, 100=perfect),
  "risk_level": "low" | "medium" | "high",
  "personal_message": "Casual, friendly message addressing specific blood markers.",
  "main_culprits": [{{ "ingredient": "name", "impact": "how it affects markers", "severity": "low|medium|high" }}],
  "health_boosters": [{{ "ingredient": "name", "benefit": "specific benefit", "impact": "quantified impact" }}],
  "recommendations": {{ "should_avoid": boolean, "modifications": ["swap A for B"], "alternative_recipes": ["suggestion"] }},
  "blood_markers_affected": [{{ "marker": "name", "current_level": value, "predicted_impact": "change", "target_range": "range" }}]
}}
"""
    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL, temperature=0.3, response_format={"type": "json_object"},
            messages=[ {"role": "system", "content": "You are a medical nutrition expert. Return detailed JSON analysis only."}, {"role": "user", "content": prompt} ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Health analysis error: {e}")
        return { "overall_score": 45, "risk_level": "medium", "personal_message": "Could not complete AI analysis." }

# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@app.post("/debug-request")
async def debug_request(request: dict):
    print(f"Raw request received: {json.dumps(request, indent=2)}")
    return { "success": True, "message": "Request received successfully", "data": request, "data_types": {key: str(type(value)) for key, value in request.items()} }

@app.get("/test-health-response")
async def test_health_response():
    return { "success": True, "analysis": { "overall_score": 75, "risk_level": "low", "personal_message": "Hey! This recipe looks pretty good overall.", "main_culprits": [], "health_boosters": [], "recommendations": {}, "blood_markers_affected": [] }, "error": None }

@app.post("/import-recipe")
async def import_recipe(req: Request):
    body = await req.json()
    link = body.get("link", "").strip()
    if not link:
        raise HTTPException(400, "link is required")

    print(f"🔍 DEBUG: Starting import for link: {link}")
    recipe_data = {}
    source = "unknown"
    debug_info = {"attempts": [], "errors": []}

    # --- Strategy 1: yt-dlp for video sites (YouTube, Instagram, TikTok) ---
    try:
        print(f"🎬 DEBUG: Attempting yt-dlp extraction for: {link}")
        with tempfile.TemporaryDirectory() as tmpdir:
            info = run_yt_dlp(link, Path(tmpdir))
            cap, srt, speech, thumb = info["caption"], "", "", info["thumbnail_url"]
            print(f"🎬 DEBUG: yt-dlp extracted - Caption: {len(cap)} chars, Thumbnail: {thumb[:50] if thumb else 'None'}")
            
            if info["subs"]: 
                srt = srt_to_text(info["subs"])
                print(f"🎬 DEBUG: Subtitles extracted: {len(srt)} chars")
            if info["audio"]: 
                speech = transcribe(info["audio"])
                print(f"🎬 DEBUG: Audio transcribed: {len(speech)} chars")
            
            if cap or srt or speech:
                recipe_data = extract_recipe(cap, srt, speech, thumb)
                if recipe_data and recipe_data.get("steps") and recipe_data.get("ingredients"):
                    source = "video_extractor"
                    print(f"✅ DEBUG: Video extraction successful! Found {len(recipe_data.get('ingredients', []))} ingredients, {len(recipe_data.get('steps', []))} steps")
                else:
                    print(f"⚠️ DEBUG: Video extraction returned incomplete data: {recipe_data}")
                    debug_info["attempts"].append("yt-dlp: extracted data but missing ingredients/steps")
            else:
                print(f"⚠️ DEBUG: No usable content extracted from video")
                debug_info["attempts"].append("yt-dlp: no caption, subtitles, or audio")
    except Exception as e:
        error_msg = f"yt-dlp failed: {str(e)}"
        print(f"❌ DEBUG: {error_msg}")
        debug_info["errors"].append(error_msg)

    # --- Fallback to Website Extraction if Video Extraction Fails ---
    if not recipe_data or not recipe_data.get("steps") or not recipe_data.get("ingredients"):
        print(f"🌐 DEBUG: Attempting website extraction fallbacks...")
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            
            # --- Strategy 2: Instagram Fallback (for Instagram URLs when yt-dlp fails) ---
            if "instagram.com" in link.lower():
                try:
                    print(f"📱 DEBUG: Trying Instagram fallback extraction...")
                    recipe_data = await extract_instagram_fallback(link, client)
                    if recipe_data and recipe_data.get("steps") and recipe_data.get("ingredients"):
                        source = "instagram_fallback"
                        print(f"✅ DEBUG: Instagram fallback extraction successful!")
                    else:
                        print(f"⚠️ DEBUG: Instagram fallback returned incomplete data")
                        debug_info["attempts"].append("instagram_fallback: no data or incomplete")
                except Exception as e:
                    error_msg = f"Instagram fallback failed: {str(e)}"
                    print(f"❌ DEBUG: {error_msg}")
                    debug_info["errors"].append(error_msg)
            
            # --- Strategy 3: Spoonacular API (for recipe websites) ---
            if not recipe_data or not recipe_data.get("steps") or not recipe_data.get("ingredients"):
                try:
                    print(f"🥄 DEBUG: Trying Spoonacular API...")
                    recipe_data = await extract_recipe_from_spoonacular(link, client)
                    if recipe_data and recipe_data.get("steps") and recipe_data.get("ingredients"):
                        source = "spoonacular_api"
                        print(f"✅ DEBUG: Spoonacular extraction successful!")
                    else:
                        print(f"⚠️ DEBUG: Spoonacular returned incomplete data or None")
                        debug_info["attempts"].append("spoonacular: no data or incomplete")
                except Exception as e:
                    error_msg = f"Spoonacular failed: {str(e)}"
                    print(f"❌ DEBUG: {error_msg}")
                    debug_info["errors"].append(error_msg)
            
            # --- Strategy 4: Local HTML Scraping (JSON-LD or raw text) ---
            if not recipe_data or not recipe_data.get("steps") or not recipe_data.get("ingredients"):
                try:
                    print(f"🌐 DEBUG: Trying HTML scraping...")
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                    resp = await client.get(link, headers=headers)
                    resp.raise_for_status()
                    print(f"🌐 DEBUG: HTML fetched, {len(resp.text)} chars")
                    recipe_data = extract_recipe_from_html(resp.text)
                    if recipe_data and recipe_data.get("steps") and recipe_data.get("ingredients"):
                        source = "website_scrape"
                        print(f"✅ DEBUG: HTML scraping successful!")
                    else:
                        print(f"⚠️ DEBUG: HTML scraping returned incomplete data: {recipe_data}")
                        debug_info["attempts"].append("html_scrape: no data or incomplete")
                except Exception as e2:
                    error_msg = f"HTML scraping failed: {str(e2)}"
                    print(f"❌ DEBUG: {error_msg}")
                    debug_info["errors"].append(error_msg)

    # --- Final check and response ---
    if recipe_data and recipe_data.get("steps") and recipe_data.get("ingredients"):
        print(f"🎉 DEBUG: SUCCESS! Returning recipe from {source}")
        return {"success": True, "recipe": recipe_data, "source": source}
    else:
        print(f"💥 DEBUG: FINAL FAILURE - All extraction methods failed")
        print(f"💥 DEBUG: Attempts made: {debug_info['attempts']}")
        print(f"💥 DEBUG: Errors encountered: {debug_info['errors']}")
        print(f"💥 DEBUG: Final recipe_data: {recipe_data}")
        
        # Return more detailed error info for debugging
        error_detail = {
            "message": "Failed to extract a valid recipe from the provided link.",
            "link": link,
            "attempts": debug_info["attempts"],
            "errors": debug_info["errors"],
            "partial_data": recipe_data if recipe_data else None
        }
        raise HTTPException(500, detail=error_detail)

@app.post("/generate-step-images")
async def generate_step_images(req: ImageGenerationRequest):
    steps = req.instructions[:MAX_STEPS]
    if not steps:
        raise HTTPException(400, "instructions list is empty")
    comps = await parse_steps_async(steps)
    sem = asyncio.Semaphore(5)
    async def worker(i, c):
        async with sem:
            return await generate_step_image(i, c)
    res = await asyncio.gather(*[worker(i + 1, c) for i, c in enumerate(comps)], return_exceptions=True)
    good, bad = [], []
    for i, r in enumerate(res, 1):
        if isinstance(r, dict):
            r["step_text"] = steps[i - 1]
            good.append(r)
        else:
            bad.append({"step_number": i, "error": str(r)})
    return {"success": len(bad) == 0, "generated_images": good, "failed_steps": bad}

@app.post("/upload-user-info")
async def upload_user_info(kitchen_photos: List[UploadFile] = File(None), blood_test_pdf: UploadFile = File(None)):
    if not kitchen_photos and not blood_test_pdf:
        raise HTTPException(400, "No files uploaded.")
    response_data = {}
    upload_timestamp = int(time.time())
    if kitchen_photos:
        kitchen_id = f"kitchen_{uuid.uuid4()}"
        kitchen_upload_dir = USER_UPLOADS_DIR / kitchen_id
        kitchen_upload_dir.mkdir(exist_ok=True)
        for i, photo in enumerate(kitchen_photos):
            if not photo.content_type or not photo.content_type.startswith("image/"):
                raise HTTPException(400, f"File '{photo.filename}' is not a valid image.")
            file_path = kitchen_upload_dir / f"{upload_timestamp}_{i+1}.jpg"
            try:
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(await photo.read())
            except Exception as e:
                raise HTTPException(500, f"Failed to save photo '{photo.filename}': {e}")
        response_data["kitchen_id"] = kitchen_id
    if blood_test_pdf:
        if blood_test_pdf.content_type != "application/pdf":
            raise HTTPException(400, f"File '{blood_test_pdf.filename}' is not a PDF.")
        blood_test_id = f"blood_test_{uuid.uuid4()}"
        file_path = USER_UPLOADS_DIR / f"{upload_timestamp}_{blood_test_id}.pdf"
        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await blood_test_pdf.read())
        except Exception as e:
            raise HTTPException(500, f"Failed to save PDF '{blood_test_pdf.filename}': {e}")
        response_data["blood_test_id"] = blood_test_id
    return response_data

@app.post("/analyze-health-impact")
async def analyze_health_impact(request: HealthAnalysisRequest):
    try:
        if request.include_blood_test and request.blood_test_id:
            blood_data = extract_blood_test_data(request.blood_test_id)
            if not blood_data:
                raise HTTPException(404, f"Blood test data not found: {request.blood_test_id}")
            analysis_result = await analyze_recipe_health_impact(request.recipe, blood_data)
        else:
            analysis_result = await analyze_recipe_general_health(request.recipe)
        return {"success": True, "analysis": analysis_result, "error": None}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Health analysis error: {e}")
        raise HTTPException(500, f"Failed to analyze health impact: {str(e)}")

@app.get("/check-blood-test/{blood_test_id}")
async def check_blood_test(blood_test_id: str):
    try:
        pdf_files = list(USER_UPLOADS_DIR.glob(f"*{blood_test_id}*.pdf"))
        exists = len(pdf_files) > 0
        return {"success": True, "blood_test_id": blood_test_id, "exists": exists, "can_do_personalized_analysis": exists}
    except Exception as e:
        return {"success": False, "blood_test_id": blood_test_id, "exists": False, "can_do_personalized_analysis": False, "error": str(e)}

@app.get("/blood-test-summary/{blood_test_id}")
async def get_blood_test_summary(blood_test_id: str):
    try:
        blood_data = parse_blood_test_data(blood_test_id)
        risk_indicators = []
        if blood_data['cholesterol']['total'] > 200: risk_indicators.append("High Total Cholesterol")
        if blood_data['cholesterol']['ldl'] > 100: risk_indicators.append("High LDL Cholesterol")
        if blood_data['cholesterol']['hdl'] < 40: risk_indicators.append("Low HDL Cholesterol")
        if blood_data['blood_sugar']['fasting_glucose'] > 99: risk_indicators.append("Elevated Fasting Glucose")
        if blood_data['vitamins']['vitamin_d'] < 30: risk_indicators.append("Vitamin D Deficiency")
        return { "success": True, "blood_test_id": blood_test_id, "summary": { "risk_indicators": risk_indicators, "total_markers": len(risk_indicators), "key_values": { "total_cholesterol": blood_data['cholesterol']['total'], "ldl_cholesterol": blood_data['cholesterol']['ldl'], "fasting_glucose": blood_data['blood_sugar']['fasting_glucose'], "vitamin_d": blood_data['vitamins']['vitamin_d'] }, "overall_health_score": max(0, 100 - (len(risk_indicators) * 15)) } }
    except Exception as e:
        raise HTTPException(500, f"Failed to get blood test summary: {str(e)}")

def extract_blood_test_data(blood_test_id: str) -> Dict:
    pdf_files = list(USER_UPLOADS_DIR.glob(f"*{blood_test_id}*.pdf"))
    if not pdf_files:
        return { "cholesterol_total": 220.0, "ldl_cholesterol": 145.0, "hdl_cholesterol": 35.0, "triglycerides": 180.0, "glucose_fasting": 105.0, "hba1c": 5.8, "crp": 3.2, "vitamin_d": 18.0, "test_date": "2024-01-15" }
    return { "cholesterol_total": 220.0, "ldl_cholesterol": 145.0, "hdl_cholesterol": 35.0, "triglycerides": 180.0, "glucose_fasting": 105.0, "hba1c": 5.8, "crp": 3.2, "vitamin_d": 18.0, "test_date": "2024-01-15" }

async def analyze_recipe_health_impact(recipe: Recipe, blood_data: Dict) -> Dict:
    blood_summary = []
    if blood_data.get("ldl_cholesterol"): blood_summary.append(f"LDL Cholesterol: {blood_data['ldl_cholesterol']} mg/dL ({'HIGH' if blood_data['ldl_cholesterol'] > 100 else 'NORMAL'})")
    if blood_data.get("glucose_fasting"): blood_summary.append(f"Fasting Glucose: {blood_data['glucose_fasting']} mg/dL ({'HIGH' if blood_data['glucose_fasting'] > 100 else 'NORMAL'})")
    blood_context = "\n".join(blood_summary)

    ingredients_list = [
        i if isinstance(i, str) else i.get('name', str(i))
        for i in recipe.ingredients
    ]

    prompt = f"""
Analyze this recipe for someone with these blood test results. Respond in JSON.
BLOOD TEST RESULTS: {blood_context}
RECIPE: Name: {recipe.name}, Ingredients: {', '.join(ingredients_list)}
Return JSON: {{ "overall_score": <0-100>, "risk_level": "<low/medium/high>", "personal_message": "<...>", "main_culprits": [{{...}}], "health_boosters": [{{...}}], "recommendations": {{...}}, "blood_markers_affected": [{{...}}] }}
"""
    try:
        response = await aclient.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "system", "content": "You are a health analysis expert. Return strict JSON only."}, {"role": "user", "content": prompt}], temperature=0.3, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Health analysis error: {e}")
        return {"overall_score": 50, "risk_level": "medium", "personal_message": "Could not complete analysis."}

async def analyze_recipe_general_health(recipe: Recipe) -> Dict:
    ingredients_list = [
        i if isinstance(i, str) else i.get('name', str(i))
        for i in recipe.ingredients
    ]

    prompt = f"""
Analyze this recipe for general health. Respond in JSON.
RECIPE: Name: {recipe.name}, Ingredients: {', '.join(ingredients_list)}
Return JSON: {{ "overall_score": <0-100>, "risk_level": "<low/medium/high>", "personal_message": "<...>", "main_culprits": [{{...}}], "health_boosters": [{{...}}], "recommendations": {{...}}, "blood_markers_affected": [] }}
"""
    try:
        response = await aclient.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "system", "content": "You are a nutrition expert. Return strict JSON only."}, {"role": "user", "content": prompt}], temperature=0.3, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"General health analysis error: {e}")
        return {"overall_score": 70, "risk_level": "medium", "personal_message": "Could not complete general analysis."}

# ==================================================================================
# ★★★★★★★★★★★★★★★ FINAL PRODUCTION INGREDIENT IMAGE FUNCTION ★★★★★★★★★★★★★★★
# ==================================================================================
def get_default_ingredient_image() -> str:
    """Return a reliable default placeholder image. This is used when we are not confident."""
    return "https://placehold.co/500x500/F5F5F5/BDBDBD?text=Ingredient"

def ingredient_image_url(name: str) -> str:
    """
    Provides a professional, "HelloFresh-style" image URL for an ingredient.
    
    This function uses a new, more robust cleaning engine before checking for a match
    in the curated API libraries. "No image is better than a wrong image."
    """
    if not name or not isinstance(name, str):
        return get_default_ingredient_image()

    # --- Step 1: New, Robust Cleaning Engine ---
    clean_name = name.lower()
    
    # A comprehensive list of descriptive words to remove.
    descriptors_to_remove = [
        'diced', 'sliced', 'minced', 'chopped', 'to taste', 'for garnish', 'whole',
        'pitted', 'uncooked', 'cooked', 'raw', 'fresh', 'dried', 'ground', 'canned',
        'peeled', 'large', 'small', 'medium', 'freshly', 'optional', 'fillet',
        'boneless', 'skinless'
    ]
    
    # Remove quantities and common units from the beginning of the string.
    clean_name = re.sub(r"^\s*[\d/.]+\s*(kg|g|ml|l|tbsp|tsp|cup|oz|cloves|pinch)?s?\s+", '', clean_name)
    # Remove text in parentheses (e.g., " (optional)")
    clean_name = re.sub(r'\s*\([^)]*\)', '', clean_name)
    # Remove all descriptive words using word boundaries (\b) to avoid mistakes.
    descriptor_regex = r'\b(' + '|'.join(descriptors_to_remove) + r')\b'
    clean_name = re.sub(descriptor_regex, '', clean_name, flags=re.IGNORECASE)
    # Remove any leftover punctuation and clean up extra whitespace.
    clean_name = re.sub(r'[,.]', '', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    # --- Step 2: Multi-API Normalization Map ---
    # This is our curated list of high-confidence ingredients.
    multi_api_map = {
        "rice": {"spoonacular": "rice.jpg", "themealdb": "Rice"}, "basmati rice": {"spoonacular": "basmati-rice.jpg", "themealdb": "Basmati Rice"},
        "jasmine rice": {"spoonacular": "jasmine-rice.jpg", "themealdb": "Jasmine Rice"}, "chicken thigh": {"spoonacular": "chicken-thighs.png", "themealdb": "Chicken"},
        "chicken breast": {"spoonacular": "chicken-breasts.png", "themealdb": "Chicken"}, "chicken": {"spoonacular": "whole-chicken.jpg", "themealdb": "Chicken"},
        "garlic": {"spoonacular": "garlic.png", "themealdb": "Garlic"}, "onion": {"spoonacular": "onion.jpg", "themealdb": "Onion"},
        "red onion": {"spoonacular": "red-onion.png", "themealdb": "Red Onion"}, "red pepper": {"spoonacular": "red-bell-pepper.jpg", "themealdb": "Red Bell Pepper"},
        "green pepper": {"spoonacular": "green-bell-pepper.jpg", "themealdb": "Green Bell Pepper"}, "bell pepper": {"spoonacular": "bell-pepper.jpg", "themealdb": "Bell Pepper"},
        "chicken stock": {"spoonacular": "chicken-broth.png", "themealdb": "Chicken Stock"}, "honey": {"spoonacular": "honey.jpg", "themealdb": "Honey"},
        "oyster sauce": {"spoonacular": "oyster-sauce.jpg", "themealdb": "Oyster Sauce"}, "potato": {"spoonacular": "potatoes-yukon-gold.png", "themealdb": "Potatoes"},
        "carrot": {"spoonacular": "carrots.jpg", "themealdb": "Carrots"}, "tomato": {"spoonacular": "tomato.png", "themealdb": "Tomatoes"},
        "lime": {"spoonacular": "lime.jpg", "themealdb": "Lime"}, "lemon": {"spoonacular": "lemon.jpg", "themealdb": "Lemon"},
        "coriander": {"spoonacular": "cilantro.png", "themealdb": "Coriander"}, "cilantro": {"spoonacular": "cilantro.png", "themealdb": "Coriander"},
        "spring onion": {"spoonacular": "green-onions.jpg", "themealdb": "Scallions"}, "scallion": {"spoonacular": "green-onions.jpg", "themealdb": "Scallions"},
        "ginger": {"spoonacular": "ginger.png", "themealdb": "Ginger"}, "soy sauce": {"spoonacular": "soy-sauce.jpg", "themealdb": "Soy Sauce"},
        "olive oil": {"spoonacular": "olive-oil.jpg", "themealdb": "Olive Oil"}, "egg": {"spoonacular": "egg.png", "themealdb": "Egg"},
        "butter": {"spoonacular": "butter.png", "themealdb": "Butter"}, "sugar": {"spoonacular": "sugar-in-bowl.png", "themealdb": "Sugar"},
        "salt": {"spoonacular": "salt.jpg", "themealdb": "Salt"}, "black pepper": {"spoonacular": "black-pepper.png", "themealdb": "Black Pepper"},
        "paprika": {"spoonacular": "paprika.jpg", "themealdb": "Paprika"}, "cumin": {"spoonacular": "cumin.jpg", "themealdb": "Cumin"},
        "turmeric": {"spoonacular": "turmeric.jpg", "themealdb": "Turmeric"}, "basil": {"spoonacular": "basil.jpg", "themealdb": "Basil"},
        "rosemary": {"spoonacular": "rosemary.jpg", "themealdb": "Rosemary"}, "thyme": {"spoonacular": "thyme.jpg", "themealdb": "Thyme"},
        "water": {"spoonacular": "water.jpg", "themealdb": "Water"},
    }

    api_names = None
    # Find the best match from our curated list.
    for key, names in multi_api_map.items():
        if clean_name.startswith(key):
            api_names = names
            break
    
    if api_names:
        # --- Tier 1: Spoonacular (Highest Quality) ---
        if SPOONACULAR_API_KEY and SPOONACULAR_API_KEY != "YOUR_API_KEY_HERE":
            if "spoonacular" in api_names:
                return f"https://spoonacular.com/cdn/ingredients_500x500/{api_names['spoonacular']}"
        
        # --- Tier 2: TheMealDB (Great Fallback) ---
        if "themealdb" in api_names:
            formatted_name = urllib.parse.quote(api_names['themealdb'])
            return f"https://www.themealdb.com/images/ingredients/{formatted_name}.png"

    # --- Final Step: Return Default Placeholder ---
    # If no high-confidence match was found, return the safe placeholder.
    print(f"INFO: No high-confidence image match found for '{name}'. Cleaned name: '{clean_name}'. Using placeholder.")
    return get_default_ingredient_image()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    if SPOONACULAR_API_KEY and SPOONACULAR_API_KEY != "YOUR_API_KEY_HERE":
        print("✅ Spoonacular API key is configured.")
    else:
        print("⚠️ WARNING: Spoonacular API key not found. Website recipe import quality may be lower.")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)