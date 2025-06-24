# main.py – FastAPI backend (Updated for Render Testing)
# Reels → recipe JSON + on-demand GPT-4.1 step-image generation
# User Info → kitchen photos + blood test PDF upload handling
# deps: openai>=1.21.0 fastapi uvicorn yt-dlp pysrt python-dotenv python-multipart aiofiles

import os, json, time, tempfile, asyncio, base64, uuid
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

# --- Configuration ---
load_dotenv()
client  = OpenAI()
aclient = AsyncOpenAI()

CHAT_MODEL  = "gpt-4.1"
IMAGE_MODEL = "gpt-image-1" # Kept your original model name
MAX_STEPS   = 10

# ★★★ CHANGE 1 of 3: Define Base URL from Environment Variable ★★★
# This will be your public Render URL.
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# All temporary files will go here. This is fine for short-term testing.
USER_UPLOADS_DIR = Path("/tmp/user_uploads")
# --- End Configuration ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the upload directory on startup."""
    USER_UPLOADS_DIR.mkdir(exist_ok=True)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Custom validation error handler for better debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error on {request.method} {request.url}")
    print(f"Request body: {await request.body()}")
    print(f"Validation errors: {exc.errors()}")
    return await request_validation_exception_handler(request, exc)

# ★★★ CHANGE 2 of 3: Serve all temp files from a unified directory ★★★
app.mount("/images", StaticFiles(directory=USER_UPLOADS_DIR), name="images")

# --- Pydantic Models ---
class ImageGenerationRequest(BaseModel):
    instructions: List[str]
    recipe_title: str = "Recipe"

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
        n=1,
        response_format="b64_json"  # Request b64_json to handle the file ourselves
    )
    
    # We expect b64_json, not a temporary URL from OpenAI
    if not rsp.data[0].b64_json:
        raise ValueError("Image generation failed, no b64_json data returned.")

    image_filename = f"{uuid.uuid4()}.png"
    # Save to the directory we are serving via StaticFiles
    fname = USER_UPLOADS_DIR / image_filename
    
    # Use async file writing
    async with aiofiles.open(fname, "wb") as f:
        await f.write(base64.b64decode(rsp.data[0].b64_json))
    
    # ★★★ CHANGE 3 of 3: Use the public BASE_URL for the image link ★★★
    url = f"{BASE_URL}/images/{image_filename}"

    return {"step_number": idx, "image_url": url}

# ── Health Analysis Functions ─────────────────────────────────────────
def parse_blood_test_data(blood_test_id: str) -> Dict:
    """Parse blood test PDF and extract key health markers"""
    # In production, this would use OCR/PDF parsing
    # For now, return mock data based on common blood test markers
    return {
        "cholesterol": {
            "total": 220,
            "ldl": 140,
            "hdl": 45,
            "triglycerides": 180,
            "target_ranges": {
                "total": "< 200 mg/dL",
                "ldl": "< 100 mg/dL", 
                "hdl": "> 40 mg/dL (men), > 50 mg/dL (women)",
                "triglycerides": "< 150 mg/dL"
            }
        },
        "blood_sugar": {
            "fasting_glucose": 105,
            "hba1c": 5.8,
            "target_ranges": {
                "fasting_glucose": "70-99 mg/dL",
                "hba1c": "< 5.7%"
            }
        },
        "vitamins": {
            "vitamin_d": 18,
            "b12": 250,
            "target_ranges": {
                "vitamin_d": "30-100 ng/mL",
                "b12": "200-900 pg/mL"
            }
        },
        "minerals": {
            "iron": 85,
            "calcium": 9.5,
            "target_ranges": {
                "iron": "60-170 mcg/dL",
                "calcium": "8.5-10.5 mg/dL"
            }
        }
    }

def get_recipe_data(recipe_id: str) -> Dict:
    """Get recipe data - in production this would query a database"""
    # Mock recipe data for testing
    return {
        "title": "Butter Chicken with Rice",
        "ingredients": [
            "2 lbs chicken thighs",
            "1 cup heavy cream",
            "4 tbsp butter",
            "2 tbsp ghee",
            "1 cup basmati rice",
            "2 tbsp sugar",
            "1 tsp salt",
            "Tomato sauce",
            "Garam masala",
            "Ginger garlic paste"
        ],
        "nutrition_per_serving": {
            "calories": 650,
            "saturated_fat": 25,
            "cholesterol": 120,
            "sodium": 890,
            "carbs": 45,
            "sugar": 12,
            "protein": 35
        }
    }

async def analyze_health_impact_ai(recipe_data: Dict, blood_data: Dict) -> Dict:
    """Use OpenAI to analyze health impact of recipe on user's blood markers"""
    
    # Build comprehensive analysis prompt
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
Ingredients: {', '.join(recipe_data['ingredients'])}
Nutrition per serving: {recipe_data['nutrition_per_serving']}

Analyze this recipe's impact on the user's health markers and return JSON with:
{{
  "overall_score": number (0-100, where 100 is perfectly healthy for this person),
  "risk_level": "low" | "medium" | "high",
  "personal_message": "Casual, friendly message addressing specific blood markers like 'Yo bro, I see your cholesterol is elevated at 220...'",
  "main_culprits": [
    {{
      "ingredient": "specific ingredient name",
      "impact": "how it affects their specific blood markers",
      "severity": "low" | "medium" | "high"
    }}
  ],
  "health_boosters": [
    {{
      "ingredient": "ingredient name",
      "benefit": "specific benefit for their blood markers",
      "impact": "quantified impact if possible"
    }}
  ],
  "recommendations": {{
    "should_avoid": boolean,
    "modifications": ["specific swaps like 'Replace heavy cream with coconut milk'"],
    "alternative_recipes": ["healthier recipe suggestions"]
  }},
  "blood_markers_affected": [
    {{
      "marker": "specific marker name",
      "current_level": current_value,
      "predicted_impact": "percentage or directional change",
      "target_range": "target range for this marker"
    }}
  ]
}}

Be specific about the user's actual blood values and give personalized advice. Use a friendly, conversational tone.
"""

    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a medical nutrition expert. Return detailed JSON analysis only."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Health analysis error: {e}")
        # Return fallback analysis
        return {
            "overall_score": 45,
            "risk_level": "medium",
            "personal_message": "Based on your blood work, this recipe has some concerning elements for your cholesterol levels.",
            "main_culprits": [
                {
                    "ingredient": "heavy cream",
                    "impact": "High saturated fat content will likely raise your already elevated LDL cholesterol",
                    "severity": "high"
                },
                {
                    "ingredient": "butter",
                    "impact": "Additional saturated fat that could worsen your cholesterol profile",
                    "severity": "medium"
                }
            ],
            "health_boosters": [
                {
                    "ingredient": "chicken",
                    "benefit": "Good protein source without excessive saturated fat",
                    "impact": "Supports muscle health without major cholesterol impact"
                }
            ],
            "recommendations": {
                "should_avoid": False,
                "modifications": [
                    "Replace heavy cream with coconut milk",
                    "Use olive oil instead of butter",
                    "Reduce portion size by 25%"
                ],
                "alternative_recipes": ["Grilled chicken with vegetables", "Chicken tikka with yogurt sauce"]
            },
            "blood_markers_affected": [
                {
                    "marker": "LDL Cholesterol",
                    "current_level": 140,
                    "predicted_impact": "+8-12% increase",
                    "target_range": "< 100 mg/dL"
                },
                {
                    "marker": "Total Cholesterol", 
                    "current_level": 220,
                    "predicted_impact": "+5-8% increase",
                    "target_range": "< 200 mg/dL"
                }
            ]
        }


# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@app.post("/debug-request")
async def debug_request(request: dict):
    """Debug endpoint to see what request data is being sent"""
    print(f"Raw request received: {json.dumps(request, indent=2)}")
    return {
        "success": True,
        "message": "Request received successfully",
        "data": request,
        "data_types": {key: str(type(value)) for key, value in request.items()}
    }

@app.get("/test-health-response")
async def test_health_response():
    """Return a sample health analysis response for Swift testing"""
    return {
        "success": True,
        "analysis": {
            "overall_score": 75,
            "risk_level": "low",
            "personal_message": "Hey! This recipe looks pretty good overall. Just a few minor tweaks and you're golden!",
            "main_culprits": [
                {
                    "ingredient": "salt",
                    "impact": "High sodium can raise blood pressure",
                    "severity": "medium"
                }
            ],
            "health_boosters": [
                {
                    "ingredient": "vegetables",
                    "impact": "Great source of vitamins and fiber",
                    "severity": "high"
                }
            ],
            "recommendations": {
                "should_avoid": False,
                "modifications": ["Reduce salt", "Add more vegetables"],
                "alternative_recipes": ["Healthier version with more veggies"]
            },
            "blood_markers_affected": []
        },
        "error": None
    }

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

@app.post("/upload-user-info")
async def upload_user_info(
    kitchen_photos: List[UploadFile] = File(None, description="List of kitchen photos"),
    blood_test_pdf: UploadFile = File(None, description="A single blood test report in PDF format")
):
    if not kitchen_photos and not blood_test_pdf:
        raise HTTPException(
            status_code=400,
            detail="No files were uploaded. Please provide kitchen photos and/or a blood test PDF."
        )

    response_data = {}
    upload_timestamp = int(time.time())

    if kitchen_photos:
        kitchen_id = f"kitchen_{uuid.uuid4()}"
        kitchen_upload_dir = USER_UPLOADS_DIR / kitchen_id
        kitchen_upload_dir.mkdir(exist_ok=True)
        
        for i, photo in enumerate(kitchen_photos):
            if not photo.content_type or not photo.content_type.startswith("image/"):
                raise HTTPException(400, f"File '{photo.filename}' is not a valid image.")
            
            safe_filename = f"{upload_timestamp}_{i+1}.jpg"
            file_path = kitchen_upload_dir / safe_filename
            
            try:
                async with aiofiles.open(file_path, "wb") as f:
                    content = await photo.read()
                    await f.write(content)
            except Exception as e:
                raise HTTPException(500, f"Failed to save photo '{photo.filename}': {e}")
        
        response_data["kitchen_id"] = kitchen_id

    if blood_test_pdf:
        if blood_test_pdf.content_type != "application/pdf":
            raise HTTPException(400, f"File '{blood_test_pdf.filename}' is not a PDF.")
            
        blood_test_id = f"blood_test_{uuid.uuid4()}"
        safe_filename = f"{upload_timestamp}_{blood_test_id}.pdf"
        file_path = USER_UPLOADS_DIR / safe_filename
        
        try:
            async with aiofiles.open(file_path, "wb") as f:
                content = await blood_test_pdf.read()
                await f.write(content)
        except Exception as e:
            raise HTTPException(500, f"Failed to save PDF '{blood_test_pdf.filename}': {e}")
        
        response_data["blood_test_id"] = blood_test_id

    return response_data

@app.post("/analyze-health-impact")
async def analyze_health_impact(request: HealthAnalysisRequest):
    """Analyze recipe health impact - with or without blood test results"""
    try:
        print(f"Health analysis request received:")
        print(f"Recipe: {request.recipe.name}")
        print(f"Include blood test: {request.include_blood_test}")
        print(f"Blood test ID: {request.blood_test_id}")
        print(f"Recipe created at: {request.recipe.createdAt}")
        
        # Determine analysis type
        if request.include_blood_test and request.blood_test_id:
            # Personalized analysis with blood test data
            print("Performing personalized blood test analysis...")
            
            # Extract blood test data
            blood_data = extract_blood_test_data(request.blood_test_id)
            if not blood_data:
                raise HTTPException(404, f"Blood test data not found for ID: {request.blood_test_id}")
            
            # Perform AI-powered personalized health analysis
            analysis_result = await analyze_recipe_health_impact(request.recipe, blood_data)
            
        else:
            # General health analysis without blood test data
            print("Performing general health analysis...")
            analysis_result = await analyze_recipe_general_health(request.recipe)
        
        return {
            "success": True,
            "analysis": analysis_result,
            "error": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Health analysis error: {e}")
        raise HTTPException(500, f"Failed to analyze health impact: {str(e)}")

@app.get("/check-blood-test/{blood_test_id}")
async def check_blood_test(blood_test_id: str):
    """Check if blood test data exists for the given ID"""
    try:
        # Check if blood test file exists
        pdf_files = list(USER_UPLOADS_DIR.glob(f"*{blood_test_id}*.pdf"))
        if not pdf_files:
            pdf_files = list(USER_UPLOADS_DIR.glob(f"*blood_test*.pdf"))
        
        exists = len(pdf_files) > 0
        
        return {
            "success": True,
            "blood_test_id": blood_test_id,
            "exists": exists,
            "can_do_personalized_analysis": exists
        }
        
    except Exception as e:
        return {
            "success": False,
            "blood_test_id": blood_test_id,
            "exists": False,
            "can_do_personalized_analysis": False,
            "error": str(e)
        }

@app.get("/blood-test-summary/{blood_test_id}")
async def get_blood_test_summary(blood_test_id: str):
    """Get summary of blood test results"""
    try:
        blood_data = parse_blood_test_data(blood_test_id)
        
        # Calculate risk indicators
        risk_indicators = []
        if blood_data['cholesterol']['total'] > 200:
            risk_indicators.append("High Total Cholesterol")
        if blood_data['cholesterol']['ldl'] > 100:
            risk_indicators.append("High LDL Cholesterol")
        if blood_data['cholesterol']['hdl'] < 40:
            risk_indicators.append("Low HDL Cholesterol")
        if blood_data['blood_sugar']['fasting_glucose'] > 99:
            risk_indicators.append("Elevated Fasting Glucose")
        if blood_data['vitamins']['vitamin_d'] < 30:
            risk_indicators.append("Vitamin D Deficiency")
        
        return {
            "success": True,
            "blood_test_id": blood_test_id,
            "summary": {
                "risk_indicators": risk_indicators,
                "total_markers": len(risk_indicators),
                "key_values": {
                    "total_cholesterol": blood_data['cholesterol']['total'],
                    "ldl_cholesterol": blood_data['cholesterol']['ldl'],
                    "fasting_glucose": blood_data['blood_sugar']['fasting_glucose'],
                    "vitamin_d": blood_data['vitamins']['vitamin_d']
                },
                "overall_health_score": max(0, 100 - (len(risk_indicators) * 15))
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to get blood test summary: {str(e)}")

def extract_blood_test_data(blood_test_id: str) -> Dict:
    """Extract health markers from stored blood test PDF"""
    # Find the PDF file
    pdf_files = list(USER_UPLOADS_DIR.glob(f"*{blood_test_id}*.pdf"))
    if not pdf_files:
        pdf_files = list(USER_UPLOADS_DIR.glob(f"*blood_test*.pdf"))
    
    if not pdf_files:
        # Return mock data for testing
        return {
            "cholesterol_total": 220.0,
            "ldl_cholesterol": 145.0,
            "hdl_cholesterol": 35.0,
            "triglycerides": 180.0,
            "glucose_fasting": 105.0,
            "hba1c": 5.8,
            "crp": 3.2,
            "vitamin_d": 18.0,
            "test_date": "2024-01-15"
        }
    
    # For now, return mock data - in production you'd use PDF parsing + OCR
    # Libraries like PyPDF2, pdfplumber, or Tesseract OCR would be needed
    return {
        "cholesterol_total": 220.0,
        "ldl_cholesterol": 145.0,
        "hdl_cholesterol": 35.0,
        "triglycerides": 180.0,
        "glucose_fasting": 105.0,
        "hba1c": 5.8,
        "crp": 3.2,
        "vitamin_d": 18.0,
        "test_date": "2024-01-15"
    }

async def analyze_recipe_health_impact(recipe: Recipe, blood_data: Dict) -> Dict:
    """Use AI to analyze recipe health impact based on blood test results"""
    
    # Format blood test data for the prompt
    blood_summary = []
    if blood_data.get("ldl_cholesterol"):
        status = "HIGH" if blood_data["ldl_cholesterol"] > 100 else "NORMAL"
        blood_summary.append(f"LDL Cholesterol: {blood_data['ldl_cholesterol']} mg/dL ({status})")
    
    if blood_data.get("glucose_fasting"):
        status = "HIGH" if blood_data["glucose_fasting"] > 100 else "NORMAL"
        blood_summary.append(f"Fasting Glucose: {blood_data['glucose_fasting']} mg/dL ({status})")
    
    if blood_data.get("triglycerides"):
        status = "HIGH" if blood_data["triglycerides"] > 150 else "NORMAL"
        blood_summary.append(f"Triglycerides: {blood_data['triglycerides']} mg/dL ({status})")
    
    if blood_data.get("hba1c"):
        status = "HIGH" if blood_data["hba1c"] > 5.7 else "NORMAL"
        blood_summary.append(f"HbA1c: {blood_data['hba1c']}% ({status})")
    
    blood_context = "\n".join(blood_summary)
    
    # Create the analysis prompt
    prompt = f"""
Analyze this recipe for someone with these blood test results. Respond in JSON format only.

BLOOD TEST RESULTS:
{blood_context}

RECIPE:
Name: {recipe.name}
Ingredients: {', '.join(recipe.ingredients)}
Description: {recipe.description}

Return JSON with these exact fields:
{{
    "overall_score": <number 0-100, lower = worse for health>,
    "risk_level": "<low/medium/high>",
    "personal_message": "<casual, friendly message like talking to a bro, mention specific blood markers>",
    "main_culprits": [
        {{
            "ingredient": "<ingredient name>",
            "impact": "<specific health impact>",
            "severity": "<low/medium/high>"
        }}
    ],
    "health_boosters": [
        {{
            "ingredient": "<ingredient name>",
            "impact": "<positive health benefit>",
            "severity": "<low/medium/high>"
        }}
    ],
    "recommendations": {{
        "should_avoid": <true/false>,
        "modifications": ["<modification 1>", "<modification 2>"],
        "alternative_recipes": ["<alternative 1>", "<alternative 2>"]
    }},
    "blood_markers_affected": [
        {{
            "marker": "<marker name>",
            "current_level": <current value>,
            "predicted_impact": "<percentage or description>",
            "target_range": "<healthy range>",
            "is_out_of_range": <true/false>
        }}
    ]
}}

Be specific about health impacts and use a casual, friendly tone. Focus on the most problematic ingredients first.
"""

    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a health analysis expert. Return strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Health analysis error: {e}")
        # Return fallback analysis
        return {
            "overall_score": 50,
            "risk_level": "medium",
            "personal_message": f"Hey! Based on your blood work, this recipe has some mixed impacts. Your LDL cholesterol is at {blood_data.get('ldl_cholesterol', 'unknown')} mg/dL, so watch out for high-fat ingredients!",
            "main_culprits": [
                {
                    "ingredient": "Unknown ingredient",
                    "impact": "Could affect cholesterol levels",
                    "severity": "medium"
                }
            ],
            "health_boosters": [
                {
                    "ingredient": "Vegetables",
                    "impact": "Good for overall health",
                    "severity": "low"
                }
            ],
            "recommendations": {
                "should_avoid": False,
                "modifications": ["Consider using less oil", "Add more vegetables"],
                "alternative_recipes": ["Heart-healthy alternatives"]
            },
            "blood_markers_affected": [
                {
                    "marker": "LDL Cholesterol",
                    "current_level": blood_data.get('ldl_cholesterol', 0),
                    "predicted_impact": "Moderate impact",
                    "target_range": "< 100 mg/dL",
                    "is_out_of_range": blood_data.get('ldl_cholesterol', 0) > 100
                }
            ]
        }

async def analyze_recipe_general_health(recipe: Recipe) -> Dict:
    """Analyze recipe health impact without specific blood test data"""
    
    # Create general health analysis prompt
    prompt = f"""
Analyze this recipe for general health impact. Respond in JSON format only.

RECIPE:
Name: {recipe.name}
Ingredients: {', '.join(recipe.ingredients)}
Description: {recipe.description}

Return JSON with these exact fields:
{{
    "overall_score": <number 0-100, higher = better for general health>,
    "risk_level": "<low/medium/high>",
    "personal_message": "<casual, friendly message about general health impacts>",
    "main_culprits": [
        {{
            "ingredient": "<ingredient name>",
            "impact": "<general health impact>",
            "severity": "<low/medium/high>"
        }}
    ],
    "health_boosters": [
        {{
            "ingredient": "<ingredient name>",
            "impact": "<positive health benefit>",
            "severity": "<low/medium/high>"
        }}
    ],
    "recommendations": {{
        "should_avoid": <true/false>,
        "modifications": ["<modification 1>", "<modification 2>"],
        "alternative_recipes": ["<alternative 1>", "<alternative 2>"]
    }},
    "blood_markers_affected": []
}}

Focus on general health impacts like calories, saturated fat, sodium, fiber, vitamins, etc.
Use a casual, friendly tone. Provide actionable advice for healthier cooking.
"""

    try:
        response = await aclient.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Return strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"General health analysis error: {e}")
        # Return fallback analysis
        return {
            "overall_score": 70,
            "risk_level": "medium",
            "personal_message": f"Hey! This {recipe.name} looks tasty! Let me break down the health impacts for you. Overall it's a decent choice with some room for improvement.",
            "main_culprits": [
                {
                    "ingredient": "High sodium ingredients",
                    "impact": "Could contribute to high blood pressure",
                    "severity": "medium"
                }
            ],
            "health_boosters": [
                {
                    "ingredient": "Vegetables",
                    "impact": "Great source of vitamins and fiber",
                    "severity": "low"
                }
            ],
            "recommendations": {
                "should_avoid": False,
                "modifications": ["Reduce salt", "Add more vegetables", "Use whole grains"],
                "alternative_recipes": ["Healthier version with more veggies"]
            },
            "blood_markers_affected": []
        }

if __name__ == "__main__":
    import uvicorn
    # Render uses the PORT environment variable to route traffic.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
