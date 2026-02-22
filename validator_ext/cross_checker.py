import re
import json
import logging
from typing import Union, List, Dict

logger = logging.getLogger("Cross_Checker")

def _extract_normalized_numbers(text: str) -> set:
    """
    Uses Regex to find all numbers in a string, ignoring commas and currency symbols.
    Converts them to floats for mathematically accurate comparison 
    (so 1,000.00 matches 1000).
    """
    # Regex breakdown: Looks for word boundaries, digits, optional commas, optional decimals
    pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    raw_numbers = re.findall(pattern, text)
    
    normalized_numbers = set()
    for num in raw_numbers:
        # Remove commas for safe float conversion
        clean_num = num.replace(',', '')
        try:
            normalized_numbers.add(float(clean_num))
        except ValueError:
            continue
            
    return normalized_numbers

def validate_numbers(vlm_json_output: Union[List, Dict], ground_truth_text: str) -> bool:
    """
    Compares the numbers extracted by Gemini against the base text from MinerU.
    Returns True if valid, False if a hallucination is detected.
    """
    # If the ground truth text is completely empty, we can't mathematically validate it.
    # We must trust the VLM in this edge case (this triggers the "Safety Net" rule we built in Phase 2).
    if not ground_truth_text.strip():
        logger.warning("Ground truth text is empty. Bypassing numerical validation.")
        return True

    # 1. Convert Gemini's JSON output back into a flat string so we can run Regex on it
    vlm_string = json.dumps(vlm_json_output)
    
    # 2. Extract normalized number sets from both sources
    vlm_numbers = _extract_normalized_numbers(vlm_string)
    anchor_numbers = _extract_normalized_numbers(ground_truth_text)
    
    logger.debug(f"VLM Numbers Found: {vlm_numbers}")
    logger.debug(f"Anchor Numbers Found: {anchor_numbers}")
    
    # 3. The Validation Check
    # We check if EVERY number Gemini found exists in the MinerU anchor text.
    hallucinations = []
    for num in vlm_numbers:
        if num not in anchor_numbers:
            hallucinations.append(num)
            
    if hallucinations:
        logger.error(f"🚨 HALLUCINATION DETECTED! Gemini invented these numbers: {hallucinations}")
        return False
        
    logger.info("✅ Numerical validation passed. Zero hallucinations detected.")
    return True