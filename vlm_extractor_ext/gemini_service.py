import os
import json
import logging
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger("Gemini_Service")
_client = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        project_id = os.environ.get("GOOGLE_PROJECT_ID")
        location = os.environ.get("GOOGLE_PROJECT_LOCATION", "us-central1")
        _client = genai.Client(vertexai=True, project=project_id, location=location)
    return _client

def extract_with_gemini(route_model: str, image_path: str, context_text: str = "", is_retry: bool = False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing crop: {image_path}")

    client = get_client()

    # --- ADVANCED PROMPT ENGINEERING ---
    if route_model == "VLM_VISUAL":
        model_name = "gemini-2.5-pro"
        logger.info(f"🧠 Heavyweight visual extraction via {model_name}...")
        base_prompt = (
            "You are an expert data analyst and visual extraction AI. "
            "Analyze this cropped image of a complex chart, diagram, or nested table. "
            "Reconstruct the underlying data points into a structured JSON array of flat objects. "
            "STRICT RULES:\n"
            "1. Infer data points from graph axes accurately. Interpolate carefully if exact numbers aren't written.\n"
            "2. Ensure the output is a flat JSON array (e.g., [{\"Category\": \"A\", \"Value\": 10}]).\n"
            "3. Extract EVERY exact number visible. Do not round, summarize, or estimate unless visually necessary.\n"
            "4. Preserve all hierarchical relationships (e.g., if a sub-item belongs to a parent category, include the parent category in the object).\n"
            "5. Output ONLY valid JSON."
        )
    elif route_model == "VLM_TABLE":
        model_name = "gemini-2.5-flash"
        logger.info(f"⚡ High-speed tabular extraction via {model_name}...")
        base_prompt = (
            "You are an expert document OCR and table extraction AI. "
            "Analyze this cropped image of a table, form, or invoice block and convert it into a structured JSON array. "
            "STRICT RULES:\n"
            "1. The output MUST be a JSON array of objects (e.g., [{\"Column1\": \"Data\", \"Column2\": 100}]).\n"
            "2. Treat column headers as JSON keys and rows as JSON objects.\n"
            "3. If a cell is visually empty in the image, represent it as null or an empty string \"\" in the JSON. Do not shift columns.\n"
            "4. For invoices or receipts, ensure line items are kept together and totals/taxes are captured accurately.\n"
            "5. Extract EVERY number and text snippet precisely as it appears. Do not correct spelling.\n"
            "6. Output ONLY valid JSON."
        )
    else: # VLM_TEXT
        model_name = "gemini-2.5-flash"
        logger.info(f"📝 High-speed text transcription via {model_name}...")
        base_prompt = (
            "You are a highly precise OCR transcription agent. Extract the exact text from this cropped image. "
            "STRICT RULES:\n"
            "1. Do NOT format as a table, do NOT use markdown tables, and do NOT output JSON.\n"
            "2. Return the raw text exactly as it appears in the image.\n"
            "3. Maintain original line breaks and spacing.\n"
            "4. Ignore background noise, watermarks, or page borders.\n"
            "5. If the image is entirely blank or unreadable, return the word 'UNREADABLE'."
        )


    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    if context_text.strip():
        base_prompt += f"\n\nContext for abbreviations: '{context_text}'"

    if is_retry:
        base_prompt += "\n\nCRITICAL WARNING: Previous numerical validation failed. Ensure exact digit extraction."

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[pil_image, base_prompt],
            config=types.GenerateContentConfig(
                response_mime_type=mime_type,
                temperature=0.0,
            )
        )
        
        # Parse based on the requested MIME type
        if mime_type == "application/json":
            return json.loads(response.text)
        else:
            return response.text.strip()

    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {e}\nRaw: {response.text}")
        raise RuntimeError("Failed to parse Gemini output as JSON.")