import os
import io
import json
import base64
import datetime
import dotenv
import uvicorn
import traceback # For detailed error logging
import requests # For calling the Apps Script Web App

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Load environment variables from .env file
dotenv.load_dotenv()

# ── OpenAI Configuration ---------------------------------------------------
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("CRITICAL ERROR: OPENAI_API_KEY not found in .env file.")
    exit("Missing OpenAI API Key.")
oa = OpenAI(api_key=OPENAI_API_KEY)
VISION_MODEL = "gpt-4o-mini" # Or "gpt-4o" for potentially higher accuracy

# --- Apps Script Web App URL ---
APPS_SCRIPT_WEB_APP_URL = os.getenv("APPS_SCRIPT_URL")
if not APPS_SCRIPT_WEB_APP_URL:
    print("CRITICAL ERROR: APPS_SCRIPT_URL not set in .env file.")
    exit("Missing Apps Script Web App URL configuration.")

# ── Pydantic Schemas -------------------------------------------------------
class Item(BaseModel):
    name: str
    qty: int
    rate: float
    total: float

class Quote(BaseModel):
    qno: str | None = None
    date: str = Field(default_factory=lambda: datetime.date.today().strftime("%d/%m/%Y"))
    address_lines: list[str] = []
    subject: str | None = None
    items: list[Item] = []
    tc_instructions: str = "" # Raw instructions from LLM

# This is used for the LLM prompt.
QUOTE_PYDANTIC_SCHEMA_FOR_LLM = Quote.model_json_schema()


# ── Terms & Conditions Logic (Python-side) -------------------------------
DEFAULT_TC = [
  "Rates including GST as applicable.",
  "Above Rates are F.O.R. our ex Guwahati Godown only; we will not be held "
  "responsible for any loss, damage or delays in transit of booked material.",
  "Goods once sold and accepted will not be taken back.",
  "100 % payment along with the order.",
  "Payment should be made in the name of Pragjyotika Enterprise payable at "
  "Guwahati, Punjab National Bank, A/C No. 3213002100002958, " # Verified A/C
  "IFSC Code PUNB0321300 by Cheque/RTGS."
]

def build_terms_string_for_document(instructions_from_llm: str) -> str:
    """
    Builds the final terms and conditions string to be inserted into the document.
    Uses OpenAI to refine DEFAULT_TC if specific instructions are provided by the LLM.
    Otherwise, formats DEFAULT_TC.
    """
    final_terms_list = []
    processed_instructions = instructions_from_llm.strip().lower() if instructions_from_llm else ""

    if not processed_instructions or processed_instructions == 'same' or processed_instructions == 'default':
        print("Using default T&C as no specific instructions or 'same'/'default' was provided.")
        final_terms_list = DEFAULT_TC
    else:
        system_prompt = (
            "You are an assistant that helps edit or generate a list of Terms & Conditions for a business quotation. "
            "You will be given a base list of T&C and a specific instruction from a user. "
            "Your goal is to return a complete, updated list of terms and conditions as a JSON array of strings. "
            "Each string in the array is a single term. "
            "If the instruction is an addition, add it to the relevant existing terms. "
            "If the instruction is a modification, modify the relevant term. "
            "If the instruction seems to be a complete replacement or a new set of terms, use that. "
            "Ensure the final list is coherent and complete. "
            "Return a JSON object with a single key 'final_terms_list' which holds this array of strings."
        )
        user_message_content = json.dumps({
            "base_default_terms": DEFAULT_TC,
            "user_instruction_for_modification": instructions_from_llm 
        }, indent=2)

        try:
            print(f"Calling OpenAI to refine T&C with instruction: '{instructions_from_llm}'")
            response = oa.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                response_format={"type": "json_object"}
            )
            refined_terms_data_str = response.choices[0].message.content
            print(f"OpenAI response for T&C refinement: {refined_terms_data_str}")
            refined_terms_data = json.loads(refined_terms_data_str)
            final_terms_list = refined_terms_data.get("final_terms_list", DEFAULT_TC) 
            if not isinstance(final_terms_list, list) or not all(isinstance(term, str) for term in final_terms_list):
                print("Warning: OpenAI T&C refinement did not return a valid list of strings. Using default.")
                final_terms_list = DEFAULT_TC
            else:
                print(f"OpenAI refined T&C list: {final_terms_list}")
        except Exception as e:
            print(f"Error calling OpenAI for T&C refinement or parsing its response: {e}. Using default T&C instead.")
            final_terms_list = DEFAULT_TC
    
    if not final_terms_list: 
        return "1. Terms and conditions apply." 
        
    return "\n".join(f"{i+1}. {term_item}" for i, term_item in enumerate(final_terms_list))

# ── FastAPI Application ----------------------------------------------------
app = FastAPI(title="PRAGJYOTIKA QUOTATION Tool", description="Upload a handwritten quotation to generate a PDF and email it.")

APPS_SCRIPT_TEMPLATE_ID_FOR_DISPLAY = "18-FHIlG4keOivkf6oiSaYUZ4lFysYCVzO-YdmLE1rv4" # From your Apps Script

HTML_FORM = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRAGJYOTIKA QUOTATION</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        .spinner {{
            border-top-color: transparent;
            border-right-color: transparent;
            animation: spin 0.6s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        /* Hide the default file input */
        #photoFile {{
            display: none;
        }}
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 to-slate-700 text-slate-50 min-h-screen flex flex-col items-center justify-center p-4 selection:bg-sky-500 selection:text-white">

    <div class="bg-slate-800 shadow-2xl rounded-xl p-6 md:p-10 w-full max-w-lg">
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-cyan-300 mb-2">PRAGJYOTIKA QUOTATION</h1>
            <p class="text-slate-400 text-sm">Upload filled quotation form to generate a PDF.</p>
            <p class="text-xs text-slate-500 mt-1">(Apps Script Template ID: {APPS_SCRIPT_TEMPLATE_ID_FOR_DISPLAY})</p>
        </div>

        <form id="uploadForm" enctype="multipart/form-data" class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Quotation Image:</label>
                
                <!-- Image Upload Area -->
                <div id="imageUploadArea" class="mt-1 flex flex-col items-center justify-center w-full h-64 px-6 pt-5 pb-6 border-2 border-slate-600 border-dashed rounded-xl cursor-pointer hover:border-sky-500 transition-colors duration-200 ease-in-out bg-slate-700/50">
                    <div id="imagePreviewContainer" class="w-full h-full flex items-center justify-center hidden">
                        <img id="imagePreview" src="#" alt="Image preview" class="max-h-full max-w-full rounded-lg shadow-md object-contain"/>
                    </div>
                    <div id="uploadPrompt" class="text-center">
                        <svg class="mx-auto h-12 w-12 text-slate-500" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                        </svg>
                        <p class="mt-2 text-sm text-slate-400">
                            <span class="font-medium text-sky-400">Click to upload</span> or drag and drop
                        </p>
                        <p class="text-xs text-slate-500">PNG, JPG, GIF (MAX. 10MB)</p>
                    </div>
                </div>
                <input id="photoFile" name="photo" type="file" class="hidden" accept="image/*" capture="environment" required>

                <!-- File Info and Retake Button -->
                <div id="fileInfoContainer" class="mt-3 text-center hidden">
                    <p class="text-sm text-slate-300">Selected: <span id="fileName" class="font-medium"></span></p>
                    <button type="button" id="changeImageButton" class="mt-2 text-xs text-sky-400 hover:text-sky-300 font-medium underline">
                        Change / Retake Image
                    </button>
                </div>
            </div>
            
            <button type="submit" id="submitButton" 
                    class="w-full flex items-center justify-center bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-600 hover:to-cyan-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-2 focus:ring-offset-slate-800 transition-all duration-150 ease-in-out text-lg">
                <span id="buttonText">Generate & Send Quotation</span>
                <svg id="loaderIcon" class="spinner w-5 h-5 ml-3 hidden" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </button>
        </form>

        <div id="responseMessageWrapper" class="mt-6 hidden">
            <h3 class="text-lg font-semibold text-slate-200 mb-2">Processing Result:</h3>
            <div id="responseMessage" class="p-4 rounded-md text-sm"></div>
        </div>
    </div>
    
    <footer class="text-center mt-10 text-xs text-slate-500">
        Powered by Pragjyotika Automation Suite &copy; {datetime.date.today().year}
    </footer>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const photoFileInput = document.getElementById('photoFile');
        const imageUploadArea = document.getElementById('imageUploadArea');
        const imagePreviewContainer = document.getElementById('imagePreviewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const uploadPrompt = document.getElementById('uploadPrompt');
        const fileInfoContainer = document.getElementById('fileInfoContainer');
        const fileNameDisplay = document.getElementById('fileName');
        const changeImageButton = document.getElementById('changeImageButton');
        
        const submitButton = document.getElementById('submitButton');
        const buttonText = document.getElementById('buttonText');
        const loaderIcon = document.getElementById('loaderIcon');
        const responseMessageWrapper = document.getElementById('responseMessageWrapper');
        const responseMessageDiv = document.getElementById('responseMessage');

        // Function to handle file display
        function handleFile(file) {{
            if (file) {{
                fileNameDisplay.textContent = file.name;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    imagePreview.src = e.target.result;
                    imagePreviewContainer.classList.remove('hidden');
                    uploadPrompt.classList.add('hidden');
                    fileInfoContainer.classList.remove('hidden');
                    imageUploadArea.classList.remove('border-dashed', 'h-64', 'items-center', 'justify-center');
                    imageUploadArea.classList.add('h-auto', 'p-2'); // Adjust height for preview
                }}
                reader.readAsDataURL(file);
            }} else {{
                imagePreview.src = "#";
                imagePreviewContainer.classList.add('hidden');
                uploadPrompt.classList.remove('hidden');
                fileInfoContainer.classList.add('hidden');
                fileNameDisplay.textContent = "";
                imageUploadArea.classList.add('border-dashed', 'h-64', 'items-center', 'justify-center');
                imageUploadArea.classList.remove('h-auto', 'p-2');
            }}
        }}

        // Trigger file input when the upload area is clicked
        imageUploadArea.addEventListener('click', () => {{
            photoFileInput.click();
        }});

        // Handle file selection via input
        photoFileInput.addEventListener('change', function(event) {{
            handleFile(event.target.files[0]);
        }});

        // Handle "Change/Retake Image" button click
        changeImageButton.addEventListener('click', function() {{
            photoFileInput.value = null; // Clear the selected file so 'change' event fires even if same file is re-selected
            photoFileInput.click(); 
        }});

        // Drag and Drop functionality
        imageUploadArea.addEventListener('dragover', (event) => {{
            event.preventDefault(); // Prevent default behavior (Prevent file from being opened)
            imageUploadArea.classList.add('border-sky-500', 'bg-slate-700');
        }});

        imageUploadArea.addEventListener('dragleave', () => {{
            imageUploadArea.classList.remove('border-sky-500', 'bg-slate-700');
        }});

        imageUploadArea.addEventListener('drop', (event) => {{
            event.preventDefault();
            imageUploadArea.classList.remove('border-sky-500', 'bg-slate-700');
            if (event.dataTransfer.files && event.dataTransfer.files[0]) {{
                photoFileInput.files = event.dataTransfer.files; // Assign dropped files to input
                handleFile(event.dataTransfer.files[0]); // Process the first dropped file
            }}
        }});


        // Form submission logic (remains largely the same)
        uploadForm.addEventListener('submit', async function(event) {{
            event.preventDefault();
            const formData = new FormData();
            const photoFile = photoFileInput.files[0];

            if (!photoFile) {{
                responseMessageWrapper.classList.remove('hidden');
                responseMessageDiv.textContent = 'Please select an image file first.';
                responseMessageDiv.className = 'p-4 rounded-md text-sm bg-red-100 text-red-700 border border-red-300';
                return;
            }}
            formData.append('photo', photoFile);

            submitButton.disabled = true;
            buttonText.textContent = 'Processing...';
            loaderIcon.classList.remove('hidden');
            responseMessageWrapper.classList.remove('hidden');
            responseMessageDiv.textContent = 'Uploading image and processing quotation... Please wait, this can take up to 90 seconds.';
            responseMessageDiv.className = 'p-4 rounded-md text-sm bg-sky-100 text-sky-700 border border-sky-300';

            try {{
                const response = await fetch('/upload', {{ method: 'POST', body: formData }});
                const result = await response.json();
                
                let messageText = 'FastAPI Call Status: ' + (response.ok ? 'Request processed by server.' : 'FastAPI Error (HTTP ' + response.status + ')') + '\\n\\n';
                
                if (result.apps_script_response) {{
                    messageText += 'Apps Script Status: ' + result.apps_script_response.status + '\\n';
                    messageText += 'Apps Script Message: ' + result.apps_script_response.message + '\\n';
                    if(result.apps_script_response.pdfName) messageText += 'PDF Name: ' + result.apps_script_response.pdfName + '\\n';
                }}
                
                if (result.error) {{
                     messageText += 'Error Type: ' + result.error + '\\n';
                     if(result.detail) messageText += 'Details: ' + result.detail + '\\n';
                }}
                
                if(result.extracted_data_sent) {{
                    messageText += '\\nData Sent to Apps Script (for debugging):\\n' + JSON.stringify(result.extracted_data_sent, null, 2);
                }} else if (result.llm_output && (!response.ok || (result.apps_script_response && result.apps_script_response.status !== 'success'))) {{
                     messageText += '\\nLLM Raw Output (if error):\\n' + result.llm_output;
                }}
                 if(result.traceback) messageText += '\\nPython Traceback (if error):\\n' + result.traceback;

                responseMessageDiv.textContent = messageText;
                let isOverallSuccess = response.ok && !result.error && (!result.apps_script_response || result.apps_script_response.status === 'success');
                if (isOverallSuccess) {{
                    responseMessageDiv.className = 'p-4 rounded-md text-sm bg-green-100 text-green-700 border border-green-300';
                }} else {{
                    responseMessageDiv.className = 'p-4 rounded-md text-sm bg-red-100 text-red-700 border border-red-300';
                }}

            }} catch (error) {{ 
                responseMessageDiv.textContent = 'Network Error or script issue: ' + error.message + '\\nCheck browser console and Python server logs for more details.'; 
                responseMessageDiv.className = 'p-4 rounded-md text-sm bg-red-100 text-red-700 border border-red-300'; 
                console.error("Fetch error:", error);
            }} finally {{
                submitButton.disabled = false;
                buttonText.textContent = 'Generate & Send Quotation';
                loaderIcon.classList.add('hidden');
            }}
        }});
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_FORM

@app.post("/upload")
async def upload(photo: UploadFile):
    img_bytes = await photo.read()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    data_url = f"data:{photo.content_type};base64,{base64_image}"

    prompt_text = (
        "Extract the handwritten values from the quotation form. "
        "The quotation number is 'qno'. "
        "The date is 'date' (try to parse it as DD/MM/YYYY or DD/MM/YY). "
        "Address lines are 'address_lines' (list of strings). "
        "The subject is 'subject'. "
        "Items are a list called 'items', where each item has 'name' (string), 'qty' (integer), "
        "'rate' (float), and 'total' (float). "
        "Any extra terms or instructions should be in 'tc_instructions' (string). "
        f"Return STRICT JSON matching this Pydantic schema (structure is key):\n{json.dumps(QUOTE_PYDANTIC_SCHEMA_FOR_LLM, indent=2)}"
    )
    
    extracted_content_str = "LLM call not made or failed early."
    payload_for_apps_script = {} 

    try:
        print("Calling OpenAI Vision to extract data from image...")
        vision_response = oa.chat.completions.create(
            model=VISION_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
            ]}]
        )
        extracted_content_str = vision_response.choices[0].message.content
        print(f"LLM Extracted Content:\n{extracted_content_str}")
        extracted_data_dict = json.loads(extracted_content_str)

        if 'date' in extracted_data_dict and isinstance(extracted_data_dict['date'], str):
            original_date_str = extracted_data_dict['date']
            parsed_successfully = False
            date_formats_to_try = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y"]
            for fmt in date_formats_to_try:
                try:
                    parsed_date = datetime.datetime.strptime(original_date_str, fmt)
                    extracted_data_dict['date'] = parsed_date.strftime("%d/%m/%Y")
                    parsed_successfully = True
                    print(f"Date '{original_date_str}' parsed with format '{fmt}' to '{extracted_data_dict['date']}'.")
                    break
                except ValueError: continue
            if not parsed_successfully:
                print(f"Warning: Date '{original_date_str}' from LLM not parsed. Sending as is: '{original_date_str}'")
                extracted_data_dict['date'] = original_date_str

        raw_tc_instructions_from_llm = extracted_data_dict.get('tc_instructions', '')
        print(f"Raw T&C instructions from LLM: '{raw_tc_instructions_from_llm}'")
        final_terms_string_for_doc = build_terms_string_for_document(raw_tc_instructions_from_llm)
        print(f"Final T&C string prepared for document:\n-----\n{final_terms_string_for_doc}\n-----")
        
        payload_for_apps_script = extracted_data_dict.copy()
        payload_for_apps_script['final_generated_terms'] = final_terms_string_for_doc

    except json.JSONDecodeError as je:
        tb_str = traceback.format_exc()
        print(f"LLM JSONDecodeError: {je}\n{tb_str}\nLLM Output which caused error:\n{extracted_content_str}")
        return JSONResponse(
            status_code=500,
            content={"error": "llm_json_decode_failed", "detail": str(je), "llm_output": extracted_content_str, "traceback": tb_str}
        )
    except Exception as e_prep: 
        tb_str = traceback.format_exc()
        print(f"Error during data preparation (LLM/Pydantic/Terms): {e_prep}\n{tb_str}")
        return JSONResponse(
            status_code=400, 
            content={"error": "data_preparation_error", "detail": str(e_prep), "llm_output": extracted_content_str, "traceback": tb_str}
        )

    try:
        print(f"Sending data to Apps Script URL: {APPS_SCRIPT_WEB_APP_URL}")
        print(f"Payload for Apps Script:\n{json.dumps(payload_for_apps_script, indent=2)}")
        headers = {"Content-Type": "application/json"}
        
        apps_script_response_obj = requests.post(
            APPS_SCRIPT_WEB_APP_URL, 
            data=json.dumps(payload_for_apps_script), 
            headers=headers, 
            timeout=90 
        )
        print(f"Apps Script HTTP Status Code: {apps_script_response_obj.status_code}")
        apps_script_response_obj.raise_for_status() 

        response_data_from_apps_script = apps_script_response_obj.json()
        print(f"Response from Apps Script (JSON Parsed):\n{json.dumps(response_data_from_apps_script, indent=2)}")

        if response_data_from_apps_script.get("status") == "success":
            return JSONResponse({
                "message": "Apps Script processed successfully.",
                "apps_script_response": response_data_from_apps_script,
                "extracted_data_sent": payload_for_apps_script
            })
        else:
            print(f"Apps Script reported an error: {response_data_from_apps_script.get('message')}")
            return JSONResponse(
                status_code=502, 
                content={
                    "error": "apps_script_processing_error",
                    "detail": response_data_from_apps_script.get("message", "Unknown error from Apps Script."),
                    "apps_script_response": response_data_from_apps_script,
                    "extracted_data_sent": payload_for_apps_script
                }
            )
    except requests.exceptions.Timeout:
        tb_str = traceback.format_exc()
        print(f"Timeout calling Apps Script:\n{tb_str}")
        return JSONResponse(
            status_code=504, 
            content={"error": "apps_script_request_timeout", "detail": "The request to Google Apps Script timed out after 90 seconds.", "traceback": tb_str, "extracted_data_sent": payload_for_apps_script}
        )
    except requests.exceptions.RequestException as re:
        tb_str = traceback.format_exc()
        error_response_content = ""
        if re.response is not None:
            error_response_content = re.response.text
            print(f"Error calling Apps Script: {re}\nApps Script HTTP Error Response Content:\n{error_response_content}\nTraceback:\n{tb_str}")
        else:
            print(f"Error calling Apps Script (no response object): {re}\nTraceback:\n{tb_str}")
        return JSONResponse(
            status_code=502, 
            content={"error": "apps_script_request_failed", "detail": str(re), "apps_script_error_content": error_response_content, "traceback": tb_str, "extracted_data_sent": payload_for_apps_script}
        )
    except Exception as e_as_call: 
        tb_str = traceback.format_exc()
        print(f"Unexpected error during Apps Script call phase: {e_as_call}\n{tb_str}")
        return JSONResponse(
            status_code=500, 
            content={"error": "unexpected_apps_script_call_error", "detail": str(e_as_call), "traceback": tb_str, "extracted_data_sent": payload_for_apps_script}
        )

if __name__ == "__main__":
    print(f"Starting Uvicorn server.")
    print(f"OpenAI API Key Loaded: {OPENAI_API_KEY is not None and len(OPENAI_API_KEY) > 0}")
    print(f"Apps Script URL Loaded: {APPS_SCRIPT_WEB_APP_URL is not None and len(APPS_SCRIPT_WEB_APP_URL) > 0}")
    print(f"The Apps Script (URL: {APPS_SCRIPT_WEB_APP_URL}) should use Template ID: {APPS_SCRIPT_TEMPLATE_ID_FOR_DISPLAY}")
    default_port = int(os.getenv("PORT", 8000))
    print(f"FastAPI app will run on http://0.0.0.0:{default_port}")
    uvicorn.run(app, host="0.0.0.0", port=default_port)
