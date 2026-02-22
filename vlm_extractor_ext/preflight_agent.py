import os
import json
import logging
from PIL import Image
from google import genai
from google.genai import types
from vlm_extractor_ext.gemini_service import get_client

logger = logging.getLogger("PreFlight_Agent")

def analyze_page_macro_layout(image_path: str) -> dict:
    """
    Sends the entire page to Gemini Flash to analyze the macro-structure BEFORE we run OCR.
    Determines if the page is highly fragmented (like an invoice).
    """
    client = get_client()
    
    prompt = (
        "You are a master Document Layout Analyzer. Look at this entire document page. "
        "Analyze its macro-structure and return a JSON object with the following schema:\n"
        "{\n"
        "  \"document_type\": \"invoice | receipt | scientific_paper | standard_text | form\",\n"
        "  \"estimated_macro_segments\": <int>,\n"
        "  \"is_highly_fragmented\": <boolean> (true if it has many floating numbers/addresses without table borders, like an invoice),\n"
        "  \"layout_description\": \"<brief description of the page layout>\"\n"
        "}\n"
        "Output ONLY valid JSON."
    )
    
    try:
        pil_image = Image.open(image_path)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        schema = json.loads(response.text)
        logger.info(f"🚁 Pre-Flight Analysis Complete: Type=[{schema.get('document_type')}], Fragmented=[{schema.get('is_highly_fragmented')}]")
        return schema
    except Exception as e:
        logger.error(f"Pre-Flight Analysis failed: {e}")
        return {"is_highly_fragmented": False, "document_type": "unknown"}