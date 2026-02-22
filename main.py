import os
import shutil
import logging
import argparse
import sys
import time

# ==========================================
# ENVIRONMENT SETUP (MUST HAPPEN FIRST)
# ==========================================
from settings import setup_environment
setup_environment()

from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# --- Import Custom Modules ---
from layout_ocr_ext.mineru_service import process_with_mineru
from vlm_router_ext.traffic_control import get_routing_decision
# NOTE: Corrected the spelling to match your actual filename (pdf_croper.py)
from utils_ext.pdf_croper import crop_pdf_element
from vlm_extractor_ext.gemini_service import extract_with_gemini
from validator_ext.cross_checker import validate_numbers
from assembler_ext.document_builder import assemble_final_document
from layout_ocr_ext.docling_service import process_with_docling
from layout_ocr_ext.gemini_layout_service import process_with_gemini_layout


# Setup unified logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("Main_Orchestrator")

app = FastAPI(title="Ext-OCR Hybrid Pipeline", version="1.0.0")

# Ensure required directories exist
os.makedirs("data_ext/uploads", exist_ok=True)
os.makedirs("data_ext/outputs", exist_ok=True)

def execute_extraction_pipeline(file_path: str, original_filename: str) -> str:
    """The master function that orchestrates all phases and tracks execution time."""
    logger.info(f"--- Starting Pipeline for: {original_filename} ---")
    total_start_time = time.perf_counter()
    
    # ==========================================
    # PHASE 1: Layout & Anchoring
    # ==========================================
    logger.info("Starting Phase 1: MinerU Layout & Base OCR Extraction...")
    p1_start = time.perf_counter()
    raw_elements = process_with_mineru(file_path)
    raw_elements = process_with_docling(file_path)
    raw_elements = process_with_gemini_layout(file_path)



    logger.info(f"⏱️ Phase 1 Completed in {time.perf_counter() - p1_start:.2f}s. Extracted {len(raw_elements)} elements.")
    
    processed_elements = []
    
    # ==========================================
    # PHASE 2, 3, & 4: Routing, VLM Extraction, & Validation
    # ==========================================
    logger.info("Starting Phase 2-4: Traffic Routing and Gemini VLM Processing...")
    p24_start = time.perf_counter()
    api_call_count = 0
    
    for idx, element in enumerate(raw_elements):
        logger.info(f"Processing Element {idx+1}/{len(raw_elements)} (Type: {element.get('type')})")
        try:
            # PHASE 2: Route
            route = get_routing_decision(
                element_type=element.get('type', ''), 
                confidence=element.get('confidence', 1.0), 
                base_text=element.get('base_text', '')
            )
            
            if route == "SKIP_VLM":
                element['final_content'] = element.get('base_text', '')
                processed_elements.append(element)
                continue
                
            # PHASE 3 PREP: Crop (Native support for PDFs and Images via PyMuPDF)
            image_crop_path = crop_pdf_element(
                pdf_path=file_path, 
                page_num=element.get('page', 1), 
                bbox=element.get('bbox', [0,0,0,0])
            )
            
            # PHASE 3 & 4: VLM + Validate Loop
            max_retries = 2
            attempt = 0
            is_valid = False
            final_data = None
            
            while attempt < max_retries and not is_valid:
                vlm_start = time.perf_counter()
                
                final_data = extract_with_gemini(
                    route_model=route, 
                    image_path=image_crop_path, 
                    context_text=element.get('base_text', ''),
                    is_retry=(attempt > 0)
                )
                
                api_call_count += 1
                logger.debug(f"Gemini API Call took {time.perf_counter() - vlm_start:.2f}s")
                
                is_valid = validate_numbers(
                    vlm_json_output=final_data, 
                    ground_truth_text=element.get('base_text', '')
                )
                attempt += 1
                
            # Cleanup crop
            if os.path.exists(image_crop_path):
                os.remove(image_crop_path)
                
            if not is_valid:
                element['flagged_for_human'] = True
                
            element['final_content'] = final_data
            processed_elements.append(element)

        except Exception as e:
            logger.error(f"Failed to process element '{element.get('type')}': {e}")
            element['final_content'] = element.get('base_text', "*Extraction Error*")
            element['flagged_for_human'] = True
            processed_elements.append(element)

    logger.info(f"⏱️ Phase 2-4 Completed in {time.perf_counter() - p24_start:.2f}s. API calls: {api_call_count}")

    # ==========================================
    # PHASE 5: Assembly
    # ==========================================
    logger.info("Starting Phase 5: Markdown Document Assembly...")
    md_output_path = assemble_final_document(
        elements=processed_elements, 
        original_filename=original_filename
    )
    
    # --- Final Metrics ---
    logger.info("==================================================")
    logger.info(f"🎉 PIPELINE COMPLETE IN {time.perf_counter() - total_start_time:.2f} SECONDS")
    logger.info("==================================================")
    logger.info(f"Data secured at: {md_output_path}")
    
    return md_output_path


# ==========================================
# FASTAPI ENDPOINT (Server Mode)
# ==========================================
@app.post("/api/extract/")
async def extract_document_api(file: UploadFile = File(...)):
    """Receives a document via POST request and returns the Markdown output path."""
    file_path = os.path.join("data_ext/uploads", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        md_file = execute_extraction_pipeline(file_path, file.filename)
        return JSONResponse(content={"status": "success", "markdown_file": md_file})
    except Exception as e:
        logger.error(f"API Request Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ==========================================
# CLI ENTRY POINT (Local Mode)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid VLM Extraction Pipeline")
    parser.add_argument("mode", choices=["serve", "local"], help="'serve' to boot FastAPI, 'local' to run locally.")
    parser.add_argument("--file", type=str, default="data_ext/uploads/sample_pdf.pdf", help="Target file for local mode")
    
    args = parser.parse_args()

    if args.mode == "serve":
        logger.info("Booting FastAPI Server via Uvicorn...")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
    elif args.mode == "local":
        if not os.path.exists(args.file):
            logger.error(f"Cannot find target file at {args.file}")
            sys.exit(1)
            
        execute_extraction_pipeline(args.file, os.path.basename(args.file))