import os
import json
import logging
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger("Gemini_Service")

# 🛑 Replace or set your API KEY here
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=API_KEY)

def extract_with_gemini(route_model: str, image_path: str, context_text: str = "", is_retry: bool = False) -> list | dict:
    """
    Sends a cropped image and its surrounding context to Gemini 2.5 Flash or Pro.
    Uses specialized prompts depending on the model lane.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image crop not found at path: {image_path}")
        raise FileNotFoundError(f"Missing crop: {image_path}")

    # 1. Select Model and Assign Specialized Prompts
    if route_model == "GEMINI_PRO":
        model_name = "gemini-2.5-pro"
        logger.info(f"🧠 Initiating heavyweight extraction with {model_name}...")
        base_prompt = (
            "You are an expert data extraction and visual analysis AI. "
            "Analyze this cropped image of a complex chart, diagram, or figure from a document. "
            "Reconstruct the underlying data points, axes, labels, and hierarchical relationships into a structured JSON array of objects. "
            "Rules:\n"
            "1. Infer data points from graphs/charts accurately based on visual axes.\n"
            "2. Extract EVERY number and text snippet precisely as it appears. Do not round, summarize, or hallucinate.\n"
            "3. Output ONLY valid JSON."
        )
    else:
        model_name = "gemini-2.5-flash"
        logger.info(f"⚡ Initiating high-speed extraction with {model_name}...")
        base_prompt = (
            "You are an expert document OCR and table extraction AI. "
            "Analyze this cropped image of a table, form, or list and convert it into a structured JSON array of objects. "
            "Rules:\n"
            "1. Treat column headers as JSON keys and rows as JSON objects.\n"
            "2. Extract EVERY number and text snippet precisely as it appears. Do not summarize.\n"
            "3. If the image is a math equation, return an array with a single object containing the LaTeX string, e.g., [{\"equation\": \"...\"}].\n"
            "4. Output ONLY valid JSON."
        )

    # Load the cropped image
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to load image for Gemini: {e}")
        raise

    # 2. Inject Anti-Hallucination Context
    if context_text.strip():
        base_prompt += f"\n\nUse the following surrounding text from the document to understand abbreviations, units, or context: '{context_text}'"

    if is_retry:
        base_prompt += "\n\nCRITICAL WARNING: Your previous attempt failed validation. Pay extremely close attention to the exact numbers in the image and do not hallucinate."

    logger.debug(f"Sending prompt to {model_name}. Context length: {len(context_text)} characters.")

    # 3. Execute API Call
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[pil_image, base_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0, # Zero temperature is CRITICAL for deterministic data extraction
            )
        )
        
        extracted_data = json.loads(response.text)
        logger.info(f"✅ Successfully extracted data structure: {type(extracted_data).__name__}")
        
        return extracted_data

    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {e}")
        logger.error(f"Raw output: {response.text}")
        raise RuntimeError("Failed to parse Gemini output as JSON.")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise