import glob
import json
import logging
import os
import subprocess
from typing import Any, Dict, List

logger = logging.getLogger("MinerU_Service")


def process_with_mineru(
    pdf_path: str, output_base_dir: str = "data_ext/outputs/mineru_temp"
) -> List[Dict[str, Any]]:
    os.makedirs(output_base_dir, exist_ok=True)
    pdf_filename = os.path.basename(pdf_path)
    pdf_name_no_ext = os.path.splitext(pdf_filename)[0]
    mineru_output_dir = os.path.join(output_base_dir, pdf_name_no_ext)

    command = [
        "mineru",
        "-p",
        pdf_path,
        "-o",
        mineru_output_dir,
        "-b",
        "hybrid-auto-engine",
    ]

    logger.info("Spawning MinerU subprocess...")

    try:
        # Use Popen to stream the output in real time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Pipe errors into the standard output stream
            text=True,
            bufsize=1,  # Line buffered
        )

        # Print the stream live to the Fedora terminal
        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"  [MinerU] {line.strip()}")

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            logger.error(f"MinerU crashed with exit code {return_code}")
            raise RuntimeError("MinerU subprocess failed.")

        logger.info("MinerU subprocess completed successfully.")

    except Exception as e:
        logger.error(f"Execution wrapper failed: {e}")
        raise

    logger.info("Locating generated JSON layout file...")
    json_files = glob.glob(
        os.path.join(mineru_output_dir, "**", "*_middle.json"), recursive=True
    )

    if not json_files:
        raise FileNotFoundError("MinerU completed, but no JSON output was found.")

    target_json = json_files[0]
    logger.info(f"Parsing JSON file: {os.path.basename(target_json)}")

    return _parse_mineru_json(target_json)




def _parse_mineru_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Reads the MinerU JSON and extracts elements into the standardized format needed for Phase 2.
    """
    extracted_elements = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pdf_info = data.get('pdf_info', [])
    
    for page in pdf_info:
        page_idx = page.get('page_idx', 0)
        
        # We now know the exact key is 'para_blocks'!
        blocks = page.get('para_blocks', [])
            
        for block in blocks:
            # Robust text extraction: handles both flat text and nested span lines
            base_text = ""
            if 'text' in block:
                base_text = block['text']
            elif 'lines' in block:
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        base_text += span.get('text', '') + " "
                    base_text += "\n"
            
            # Map MinerU's output to our pipeline's expected format
            element = {
                "page": page_idx + 1,
                "type": block.get('type', 'text'),           # e.g., 'text', 'table', 'figure', 'equation'
                "bbox": block.get('bbox', [0, 0, 0, 0]),     # [x0, y0, x1, y1]
                "base_text": base_text.strip(),              # The raw OCR text (our ground truth anchor)
                "confidence": block.get('score', 1.0)        # Used by the Router in Phase 2
            }
            extracted_elements.append(element)
            
    return extracted_elements