# main.py – FastAPI backend (Updated for Render Testing)
# Reels → recipe JSON + on-demand GPT-4.1 step-image generation
# User Info → kitchen photos + blood test PDF upload handling
# deps: openai>=1.21.0 fastapi uvicorn yt-dlp pysrt python-dotenv python-multipart aiofiles

import os, json, time, tempfile, asyncio, base64, uuid, random
from pathlib import Path
from typing import List, Dict, Union, Optional
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
from pydantic import BaseModel, field_validator
import httpx
import re, urllib.parse

# --- Configuration ---
load_dotenv()
client = OpenAI()
aclient = AsyncOpenAI()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

CHAT_MODEL = "gpt-4-turbo" # Using a standard model name
IMAGE_MODEL = "dall-e-3" # Using a standard model name
MAX_STEPS = 10

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
USER_UPLOADS_DIR = Path("/tmp/user_uploads")
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

    @field_validator('createdAt', mode='before')
    @classmethod
    def parse_created_at(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, (int, float)):
            # Handle timestamp
            return datetime.fromtimestamp(v).isoformat()
        if isinstance(v, str):
            return v
        return str(v)

class HealthAnalysisRequest(BaseModel):
    recipe: Recipe
    blood_test_id: Optional[str] = None # Make optional
    include_blood_test: bool = False # Flag to control blood test analysis

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
    severity: str # Renamed from benefit/impact for consistency

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
    out = dict(audio=None, subs=None, thumb=None, caption="", thumbnail_url="")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "hi"], # Removed empty string for clarity
        "subtitlesformat": "srt",
        "writethumbnail": True,
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
    out["audio"] = next(base.parent.glob(f"{info['id']}.mp3"), None)
    out["subs"] = next(base.parent.glob(f"{info['id']}*.srt"), None)
    out["thumb"] = next(base.parent.glob(f"{info['id']}.jpg"), None) or next(base.parent.glob(f"{info['id']}.webp"), None)
    out["caption"] = (info.get("description") or "").strip()
    out["thumbnail_url"] = info.get("thumbnail", "")
    return out

def srt_to_text(path: Path) -> str:
    try:
        return " ".join(
            s.text.replace("\n", " ").strip()
            for s in pysrt.open(str(path), encoding="utf-8")
        )
    except Exception:
        return "" # Return empty string on parsing error

def transcribe(audio_path: Path) -> str:
    # OpenAI's whisper-1 is highly effective and supports large files
    model = "whisper-1"
    with audio_path.open("rb") as f:
        return client.audio.transcriptions.create(model=model, file=f).text

def gpt_json(prompt: str, temp: float) -> dict:
    rsp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temp,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user",   "content": prompt}
        ]
    )
    return json.loads(rsp.choices[0].message.content)

def extract_recipe(caption: str, srt_text: str, speech: str, thumbnail_url: str = "") -> dict:
    prompt = (
        "Based on the following video content, generate a recipe. "
        "Return a single JSON object with keys: 'title', 'description', "
        "'ingredients' (as a list of strings), and 'steps' (as a list of strings).\n\n"
        f"POST_CAPTION:\n{caption}\n\nCLOSED_CAPTIONS:\n{srt_text}\n\n"
        f"SPEECH_TRANSCRIPT:\n{speech}"
    )
    for t in (0.1, 0.5):
        try:
            data = gpt_json(prompt, t)
            if data.get("ingredients") and data.get("steps"):
                data["thumbnailUrl"] = thumbnail_url
                return data
        except Exception:
            continue
    return {
        "title": "Imported Recipe",
        "description": "Could not automatically extract recipe from Reel.",
        "ingredients": ["Please add ingredients manually."],
        "steps": ["Please add steps manually."],
        "thumbnailUrl": thumbnail_url
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
        return [{"verb": "prepare",
                 "mainIngredient": "ingredients",
                 "vessel": "vessel",
                 "visible_change": "ready"} for _ in steps]


async def generate_step_image(idx: int, comp: Dict[str, str]) -> Dict[str, str]:
    prop_choices = [
        "wooden spoon and vintage measuring cups", "ceramic ramekin of chopped fresh herbs",
        "small glass bowl of colorful spices", "tiny jug of extra-virgin olive oil",
        "marble mortar and pestle with crushed pepper", "sprig of fresh rosemary on the side",
        "chef's knife with patina finish", "linen napkin and copper spoon",
        "cast-iron skillet handle peeking in",
    ]
    chosen_props = random.choice(prop_choices)

    prompt = (
        f"High-resolution food photography, cinematic color grade. "
        f"Scene: Step {idx}. {comp['verb'].capitalize()} {comp['mainIngredient']} "
        f"in {comp['vessel']}; showing {comp['visible_change']}. "
        "Camera: 50 mm prime lens, f/2.8. Angle: top-down 90°. "
        f"Surface: rustic dark-oak board. Props: {chosen_props}. "
        "Lighting: soft window light from left. Aspect: 1:1. "
        "Negative prompts: no hands, no faces, no brand logos, no text."
    )

    rsp = await aclient.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="b64_json"
    )

    if not rsp.data or not rsp.data[0].b64_json:
        raise ValueError("Image generation failed, no b64_json data returned.")

    image_filename = f"{uuid.uuid4()}.png"
    fname = USER_UPLOADS_DIR / image_filename

    async with aiofiles.open(fname, "wb") as f:
        await f.write(base64.b64decode(rsp.data[0].b64_json))

    url = f"{BASE_URL}/images/{image_filename}"
    return {"step_number": idx, "image_url": url}

# --- Health Analysis Functions ---
def parse_blood_test_data(blood_test_id: str) -> Dict:
    """Parse blood test PDF and extract key health markers (mock implementation)."""
    # In production, this would use OCR/PDF parsing on the file found via blood_test_id
    return {
        "cholesterol": {
            "total": 220, "ldl": 140, "hdl": 45, "triglycerides": 180,
            "target_ranges": { "total": "< 200 mg/dL", "ldl": "< 100 mg/dL", "hdl": "> 40 mg/dL", "triglycerides": "< 150 mg/dL" }
        },
        "blood_sugar": {
            "fasting_glucose": 105, "hba1c": 5.8,
            "target_ranges": { "fasting_glucose": "70-99 mg/dL", "hba1c": "< 5.7%" }
        },
        "vitamins": {
            "vitamin_d": 18, "b12": 250,
            "target_ranges": { "vitamin_d": "30-100 ng/mL", "b12": "200-900 pg/mL" }
        }
    }

def extract_blood_test_data(blood_test_id: str) -> Dict:
    """Mock function to simulate fetching processed blood test data."""
    pdf_files = list(USER_UPLOADS_DIR.glob(f"*{blood_test_id}*.pdf"))
    if not pdf_files:
        # Return a structure with out-of-range values for demonstration
        return { "ldl_cholesterol": 145.0, "glucose_fasting": 105.0, "vitamin_d": 18.0 }
    # In a real app, you'd parse pdf_files[0] here.
    return { "ldl_cholesterol": 145.0, "glucose_fasting": 105.0, "vitamin_d": 18.0 }

async def analyze_recipe_health_impact(recipe: Recipe, blood_data: Dict) -> Dict:
    """Analyzes recipe impact based on user's blood data."""
    blood_summary = []
    if blood_data.get("ldl_cholesterol"): blood_summary.append(f"LDL Cholesterol: {blood_data['ldl_cholesterol']} mg/dL (HIGH)")
    if blood_data.get("glucose_fasting"): blood_summary.append(f"Fasting Glucose: {blood_data['glucose_fasting']} mg/dL (HIGH)")
    blood_context = "\n".join(blood_summary)

    # BUGFIX: The Recipe model guarantees ingredients is List[str]. No complex parsing needed.
    ingredients_list = recipe.ingredients

    prompt = f"""
Analyze this recipe for someone with these blood test results. Respond in JSON.
BLOOD TEST RESULTS: {blood_context}
RECIPE: Name: {recipe.name}, Ingredients: {', '.join(ingredients_list)}
Return JSON conforming to the HealthAnalysis Pydantic model structure provided in the system prompt.
"""
    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": "You are a health analysis expert returning JSON for a Pydantic model."}, {"role": "user", "content": prompt}],
            temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Health analysis error: {e}")
        return {"overall_score": 50, "risk_level": "medium", "personal_message": "Could not complete analysis."}

async def analyze_recipe_general_health(recipe: Recipe) -> Dict:
    """Analyzes a recipe for general health without personal data."""
    # BUGFIX: The Recipe model guarantees ingredients is List[str].
    ingredients_list = recipe.ingredients

    prompt = f"""
Analyze this recipe for general health. Respond in JSON.
RECIPE: Name: {recipe.name}, Ingredients: {', '.join(ingredients_list)}
Return JSON conforming to the HealthAnalysis Pydantic model, but with 'blood_markers_affected' as an empty list.
"""
    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": "You are a nutrition expert returning JSON for a Pydantic model."}, {"role": "user", "content": prompt}],
            temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"General health analysis error: {e}")
        return {"overall_score": 70, "risk_level": "medium", "personal_message": "Could not complete general analysis."}

def _extract_recipe_from_url_sync(link: str, tmpdir: str) -> Dict:
    """Synchronous helper for video processing to run in a thread."""
    tmp = Path(tmpdir)
    info = run_yt_dlp(link, tmp)
    cap = info["caption"]
    srt = srt_to_text(info["subs"]) if info["subs"] else ""
    speech = transcribe(info["audio"]) if info["audio"] else ""
    thumbnail_url = info["thumbnail_url"]
    recipe = extract_recipe(cap, srt, speech, thumbnail_url)
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
        # BUGFIX: Run all blocking I/O (download, file access, sync OpenAI calls)
        # in a separate thread to not block the server's event loop.
        with tempfile.TemporaryDirectory() as tmpdir:
            recipe = await asyncio.to_thread(_extract_recipe_from_url_sync, link, tmpdir)
            return {"success": True, "recipe": recipe, "source": "yt_dlp"}
    except Exception as e:
        try:
            print(f"yt_dlp failed: {e}. Falling back to basic scrape…")
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(link, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                html = resp.text

            import html as html_lib
            thumb_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            fallback_thumb = html_lib.unescape(thumb_match.group(1)) if thumb_match else ""
            desc_match = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            caption = html_lib.unescape(desc_match.group(1)) if desc_match else "Could not extract description."

            recipe = extract_recipe(caption, "", "", fallback_thumb)
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

@app.post("/upload-user-info")
async def upload_user_info(kitchen_photos: Optional[List[UploadFile]] = File(None), blood_test_pdf: Optional[UploadFile] = File(None)):
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
        file_path = USER_UPLOADS_DIR / f"{blood_test_id}.pdf" # Simplified filename
        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await blood_test_pdf.read())
        except Exception as e:
            raise HTTPException(500, f"Failed to save PDF '{blood_test_pdf.filename}': {e}")
        response_data["blood_test_id"] = blood_test_id
    return response_data

@app.post("/analyze-health-impact", response_model=HealthAnalysisResponse)
async def analyze_health_impact(request: HealthAnalysisRequest):
    try:
        if request.include_blood_test and request.blood_test_id:
            blood_data = extract_blood_test_data(request.blood_test_id)
            analysis_result = await analyze_recipe_health_impact(request.recipe, blood_data)
        else:
            analysis_result = await analyze_recipe_general_health(request.recipe)
        return HealthAnalysisResponse(success=True, analysis=analysis_result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Health analysis error: {e}")
        raise HTTPException(500, f"Failed to analyze health impact: {str(e)}")

@app.get("/check-blood-test/{blood_test_id}")
async def check_blood_test(blood_test_id: str):
    exists = any(USER_UPLOADS_DIR.glob(f"*{blood_test_id}*.pdf"))
    return {"success": True, "blood_test_id": blood_test_id, "exists": exists}

@app.get("/blood-test-summary/{blood_test_id}")
async def get_blood_test_summary(blood_test_id: str):
    try:
        # This uses the mock data function for demonstration
        blood_data = parse_blood_test_data(blood_test_id)
        risk_indicators = []
        if blood_data['cholesterol']['total'] > 200: risk_indicators.append("High Total Cholesterol")
        if blood_data['cholesterol']['ldl'] > 100: risk_indicators.append("High LDL Cholesterol")
        if blood_data['cholesterol']['hdl'] < 40: risk_indicators.append("Low HDL Cholesterol")
        if blood_data['blood_sugar']['fasting_glucose'] > 99: risk_indicators.append("Elevated Fasting Glucose")
        if blood_data['vitamins']['vitamin_d'] < 30: risk_indicators.append("Vitamin D Deficiency")

        score = max(0, 100 - (len(risk_indicators) * 15))
        summary = {
            "risk_indicators": risk_indicators,
            "key_values": {
                "total_cholesterol": blood_data['cholesterol']['total'],
                "ldl_cholesterol": blood_data['cholesterol']['ldl'],
                "fasting_glucose": blood_data['blood_sugar']['fasting_glucose'],
            },
            "overall_health_score": score
        }
        return {"success": True, "blood_test_id": blood_test_id, "summary": summary}
    except Exception as e:
        raise HTTPException(500, f"Failed to get blood test summary: {str(e)}")

# ==================================================================================
# ★★★★★★★★★★★★★★★ FINAL PRODUCTION INGREDIENT IMAGE FUNCTION ★★★★★★★★★★★★★★★
# ==================================================================================
def get_default_ingredient_image() -> str:
    """Return a reliable default placeholder image."""
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
    descriptors_to_remove = [
        'diced', 'sliced', 'minced', 'chopped', 'to taste', 'for garnish', 'whole',
        'pitted', 'uncooked', 'cooked', 'raw', 'fresh', 'dried', 'ground', 'canned',
        'peeled', 'large', 'small', 'medium', 'freshly', 'optional', 'fillet',
        'boneless', 'skinless'
    ]
    # Remove quantities and common units from the beginning.
    clean_name = re.sub(r"^\s*[\d/.]+\s*(kg|g|ml|l|tbsp|tsp|cup|oz|cloves|pinch)?s?\s+", '', clean_name)
    # Remove text in parentheses (e.g., " (optional)")
    clean_name = re.sub(r'\s*\([^)]*\)', '', clean_name)
    # Remove all descriptive words using word boundaries (\b).
    descriptor_regex = r'\b(' + '|'.join(descriptors_to_remove) + r')\b'
    clean_name = re.sub(descriptor_regex, '', clean_name, flags=re.IGNORECASE)
    # Remove any leftover punctuation and clean up extra whitespace.
    clean_name = re.sub(r'[,\.]', '', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    # --- Step 2: Multi-API Normalization Map ---
    multi_api_map = {
        "basmati rice": {"spoonacular": "basmati-rice.jpg", "themealdb": "Basmati Rice"},
        "jasmine rice": {"spoonacular": "jasmine-rice.jpg", "themealdb": "Jasmine Rice"},
        "rice": {"spoonacular": "rice.jpg", "themealdb": "Rice"},
        "chicken thigh": {"spoonacular": "chicken-thighs.png", "themealdb": "Chicken Thighs"},
        "chicken breast": {"spoonacular": "chicken-breasts.png", "themealdb": "Chicken Breast"},
        "chicken": {"spoonacular": "whole-chicken.jpg", "themealdb": "Chicken"},
        "garlic": {"spoonacular": "garlic.png", "themealdb": "Garlic"},
        "red onion": {"spoonacular": "red-onion.png", "themealdb": "Red Onion"},
        "onion": {"spoonacular": "onion.jpg", "themealdb": "Onion"},
        "red pepper": {"spoonacular": "red-bell-pepper.jpg", "themealdb": "Red Bell Pepper"},
        "green pepper": {"spoonacular": "green-bell-pepper.jpg", "themealdb": "Green Bell Pepper"},
        "bell pepper": {"spoonacular": "bell-pepper.jpg", "themealdb": "Bell Pepper"},
        "chicken stock": {"spoonacular": "chicken-broth.png", "themealdb": "Chicken Stock"},
        "honey": {"spoonacular": "honey.jpg", "themealdb": "Honey"},
        "oyster sauce": {"spoonacular": "oyster-sauce.jpg", "themealdb": "Oyster Sauce"},
        "potato": {"spoonacular": "potatoes-yukon-gold.png", "themealdb": "Potatoes"},
        "carrot": {"spoonacular": "carrots.jpg", "themealdb": "Carrots"},
        "tomato": {"spoonacular": "tomato.png", "themealdb": "Tomatoes"},
        "lime": {"spoonacular": "lime.jpg", "themealdb": "Lime"},
        "lemon": {"spoonacular": "lemon.jpg", "themealdb": "Lemon"},
        "coriander": {"spoonacular": "cilantro.png", "themealdb": "Coriander"},
        "cilantro": {"spoonacular": "cilantro.png", "themealdb": "Coriander"},
        "spring onion": {"spoonacular": "green-onions.jpg", "themealdb": "Scallions"},
        "scallion": {"spoonacular": "green-onions.jpg", "themealdb": "Scallions"},
        "ginger": {"spoonacular": "ginger.png", "themealdb": "Ginger"},
        "soy sauce": {"spoonacular": "soy-sauce.jpg", "themealdb": "Soy Sauce"},
        "olive oil": {"spoonacular": "olive-oil.jpg", "themealdb": "Olive Oil"},
        "egg": {"spoonacular": "egg.png", "themealdb": "Egg"}, "butter": {"spoonacular": "butter.png", "themealdb": "Butter"},
        "sugar": {"spoonacular": "sugar-in-bowl.png", "themealdb": "Sugar"}, "salt": {"spoonacular": "salt.jpg", "themealdb": "Salt"},
        "black pepper": {"spoonacular": "black-pepper.png", "themealdb": "Black Pepper"}, "paprika": {"spoonacular": "paprika.jpg", "themealdb": "Paprika"},
        "cumin": {"spoonacular": "cumin.jpg", "themealdb": "Cumin"}, "turmeric": {"spoonacular": "turmeric.jpg", "themealdb": "Turmeric"},
        "basil": {"spoonacular": "basil.jpg", "themealdb": "Basil"}, "rosemary": {"spoonacular": "rosemary.jpg", "themealdb": "Rosemary"},
        "thyme": {"spoonacular": "thyme.jpg", "themealdb": "Thyme"}, "water": {"spoonacular": "water.jpg", "themealdb": "Water"},
    }

    # BUGFIX: Find the best (longest) match, not just the first one.
    # This prevents "chicken breast" from matching "chicken".
    best_match_key = None
    for key in multi_api_map.keys():
        if clean_name.startswith(key):
            if best_match_key is None or len(key) > len(best_match_key):
                best_match_key = key

    api_names = multi_api_map.get(best_match_key)

    if api_names:
        # Tier 1: Spoonacular (Highest Quality)
        if SPOONACULAR_API_KEY and "spoonacular" in api_names:
            return f"https://spoonacular.com/cdn/ingredients_500x500/{api_names['spoonacular']}"

        # Tier 2: TheMealDB (Great Fallback)
        if "themealdb" in api_names:
            formatted_name = urllib.parse.quote(api_names['themealdb'])
            return f"https://www.themealdb.com/images/ingredients/{formatted_name}.png"

    # --- Final Step: Return Default Placeholder ---
    print(f"INFO: No high-confidence image match for '{name}' (cleaned: '{clean_name}'). Using placeholder.")
    return get_default_ingredient_image()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    if SPOONACULAR_API_KEY and SPOONACULAR_API_KEY != "YOUR_API_KEY_HERE":
        print("✅ Spoonacular API key is configured.")
    else:
        print("⚠️ WARNING: Spoonacular API key not found. Ingredient image quality will be lower.")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)