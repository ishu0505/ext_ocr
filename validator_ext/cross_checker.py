import re
import json
import logging
from typing import Union, List, Dict

logger = logging.getLogger("Cross_Checker")

def _extract_normalized_numbers(text: str) -> set:
    pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    raw_numbers = re.findall(pattern, text)
    
    normalized_numbers = set()
    for num in raw_numbers:
        clean_num = num.replace(',', '')
        try:
            normalized_numbers.add(float(clean_num))
        except ValueError:
            continue
    return normalized_numbers

def validate_numbers(vlm_output: Union[List, Dict, str], ground_truth_text: str) -> bool:
    if not ground_truth_text.strip():
        return True

    # If the output is already a string (from VLM_TEXT), don't dump it as JSON
    if isinstance(vlm_output, str):
        vlm_string = vlm_output
    else:
        vlm_string = json.dumps(vlm_output)
    
    vlm_numbers = _extract_normalized_numbers(vlm_string)
    anchor_numbers = _extract_normalized_numbers(ground_truth_text)
    
    hallucinations = [num for num in vlm_numbers if num not in anchor_numbers]
            
    if hallucinations:
        logger.error(f"🚨 HALLUCINATION DETECTED! Gemini invented: {hallucinations}")
        return False
        
    return True