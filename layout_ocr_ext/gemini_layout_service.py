import os
import json
import logging
import fitz
from PIL import Image
from typing import List, Dict, Any
from vlm_extractor_ext.gemini_service import get_client
from google.genai import types

logger = logging.getLogger("Gemini_Layout")

def process_with_gemini_layout(pdf_path: str, output_base_dir: str = "data_ext/outputs/gem_temp") -> List[Dict[str, Any]]:
    """
    Uses Gemini 2.5 Flash as the native layout parser. 
    It draws logical bounding boxes and extracts text simultaneously.
    """
    client = get_client()
    extracted_elements = []
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200) # High enough for Gemini to read
        temp_img_path = f"data_ext/uploads/temp_layout_p{page_num}.png"
        pix.save(temp_img_path)
        
        width, height = pix.w, pix.h
        
        prompt = f"""
        You are a Master Document Layout Analyzer. Analyze this document page.
        Image Dimensions: {width}px width by {height}px height.
        
        Divide this page into logical macro-segments (e.g., Header, Billing Info, Line Items Table, Footer, Paragraph).
        Do not over-fragment the document. Group related text (like a full address block) into a single segment.
        
        Output a JSON array of objects with this exact schema:
        [
          {{
            "type": "title" | "text" | "table" | "image",
            "bbox": [x_min, y_min, x_max, y_max],
            "base_text": "Extracted text content here (leave empty for tables or images)",
            "confidence": 1.0
          }}
        ]
        IMPORTANT: The bbox values MUST be absolute pixel coordinates based on the {width}x{height} dimensions.
        Return ONLY valid JSON.
        """
        
        try:
            pil_image = Image.open(temp_img_path)
            logger.info(f"🧠 Prompting Gemini Layout Engine for Page {page_num + 1}...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            page_elements = json.loads(response.text)
            for el in page_elements:
                el['page'] = page_num + 1
                extracted_elements.append(el)
                
        except Exception as e:
            logger.error(f"Gemini layout analysis failed on page {page_num + 1}: {e}")
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
    doc.close()
    logger.info(f"Gemini Native Layout complete. Generated {len(extracted_elements)} macro-segments.")
    return extracted_elements