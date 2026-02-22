import logging
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter

logger = logging.getLogger("Docling_Service")

def process_with_docling(pdf_path: str, output_base_dir: str = "data_ext/outputs/docling_temp") -> List[Dict[str, Any]]:
    """
    Uses IBM Docling to extract layout and text. 
    Docling is excellent at macro-grouping elements (less granular fragmentation).
    """
    logger.info(f"Starting Docling extraction for {pdf_path}...")
    
    # Initialize Docling
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    extracted_elements = []
    
    # Iterate through Docling's structured document tree
    for item, level in result.document.iterate_items():
        # Only process items that have physical bounding box provenance on the page
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0]
            bbox = prov.bbox
            page_no = prov.page_no
            
            # Convert Docling's [left, top, right, bottom] to our standard [x0, y0, x1, y1]
            x0, y0, x1, y1 = bbox.l, bbox.t, bbox.r, bbox.b
            
            text = getattr(item, 'text', '')
            item_type = item.label  # Docling labels: 'text', 'title', 'table', 'picture'
            
            # Map Docling labels to our Router labels
            if item_type == "picture": item_type = "image"
            
            extracted_elements.append({
                "page": page_no,
                "type": item_type,
                "bbox": [x0, y0, x1, y1],
                "base_text": text,
                "confidence": 1.0  # Docling is deterministic, assuming 1.0 for layout
            })
            
    logger.info(f"Docling extraction complete. Found {len(extracted_elements)} macro-segments.")
    return extracted_elements