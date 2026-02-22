import logging
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import List, Dict, Any
from paddleocr import PPStructure

logger = logging.getLogger("Paddle_Service")

# Initialize globally so we don't reload weights for every page
_engine = None

def process_with_paddle(pdf_path: str, output_base_dir: str = "data_ext/outputs/paddle_temp") -> List[Dict[str, Any]]:
    """
    Uses PaddleOCR's PP-Structure to analyze layout and extract text.
    Requires converting PDF pages to images first via PyMuPDF.
    """
    global _engine
    if _engine is None:
        logger.info("Initializing PaddleOCR PP-Structure Engine...")
        _engine = PPStructure(show_log=False, image_orientation=True)
        
    extracted_elements = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render PDF to image for Paddle
        pix = page.get_pixmap(dpi=150)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert RGB to BGR for cv2
        if pix.n == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
        logger.info(f"Running PP-Structure on Page {page_num + 1}...")
        result = _engine(img_data)
        
        for region in result:
            bbox = region['bbox']  # Returns standard [x_min, y_min, x_max, y_max]
            res_type = region['type'] # e.g., 'text', 'title', 'figure', 'table'
            
            text_content = ""
            confidence = 1.0
            
            # Paddle stores internal OCR text in 'res'
            if 'res' in region and isinstance(region['res'], list):
                texts = [t.get('text', '') for t in region['res']]
                confs = [t.get('confidence', 1.0) for t in region['res']]
                text_content = " ".join(texts)
                confidence = sum(confs) / len(confs) if confs else 1.0

            extracted_elements.append({
                "page": page_num + 1,
                "type": res_type.lower(),
                "bbox": bbox,
                "base_text": text_content,
                "confidence": confidence
            })
            
    doc.close()
    logger.info(f"PaddleOCR extraction complete. Found {len(extracted_elements)} segments.")
    return extracted_elements