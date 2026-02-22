import os
import fitz  # This is PyMuPDF
import logging
import uuid
from typing import List

# Setup specific logger
logger = logging.getLogger("PDF_Cropper")

def crop_pdf_element(pdf_path: str, page_num: int, bbox: List[float], output_dir: str = "data_ext/outputs/temp_crops", dpi: int = 300) -> str:
    """
    Opens a PDF, navigates to a specific page, crops the bounding box,
    and saves it as a high-res PNG for VLM processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename for this specific crop to avoid overwriting
    crop_filename = f"crop_p{page_num}_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(output_dir, crop_filename)
    
    logger.debug(f"Attempting to crop {pdf_path} (Page {page_num}) at bbox {bbox}")
    
    try:
        # 1. Open the document
        doc = fitz.open(pdf_path)
        
        # 2. Load the specific page (PyMuPDF is 0-indexed, MinerU is 1-indexed)
        # We subtract 1 to translate MinerU's page 1 to PyMuPDF's page 0.
        page = doc[page_num - 1]
        
        # 3. Define the Rectangle based on MinerU's [x0, y0, x1, y1] coordinates
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        
        # 4. Set the zoom matrix for high-resolution output (Crucial for VLM accuracy)
        # Default PDF resolution is 72 DPI. 300 DPI is ~4.16x zoom.
        zoom_factor = dpi / 72.0
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        
        # 5. Render that specific rectangle to a pixel map (image)
        pix = page.get_pixmap(matrix=matrix, clip=rect)
        
        # 6. Save to disk
        pix.save(output_path)
        logger.info(f"✂️ Successfully cropped element to {output_path} (DPI: {dpi})")
        
        doc.close()
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to crop PDF element: {e}")
        raise RuntimeError(f"PyMuPDF cropping error: {e}")