import logging

# Set up our specific logger for the routing phase
logger = logging.getLogger("Traffic_Control")

def get_routing_decision(element_type: str, confidence: float, base_text: str = "") -> str:
    """
    Acts as the traffic cop for the pipeline.
    Evaluates the element type, OCR confidence, and extracted text to determine the optimal processing lane.
    """
    # Normalize inputs just in case MinerU changes their casing in the future
    element_type = element_type.lower().strip()
    
    # Define our element categories based on MinerU's output labels
    text_elements = ['text', 'title', 'header', 'footer', 'paragraph', 'text_inline']
    flash_elements = ['table', 'equation', 'list', 'form', 'isolated_formula']
    pro_elements = ['figure', 'chart', 'image', 'complex_diagram']

    logger.debug(f"Evaluating -> Type: '{element_type}', Confidence: {confidence}, Text Length: {len(base_text)}")

    # RULE 0: The Empty Text Safety Net (Kaizen Improvement!)
    # If OCR failed to grab the text despite high confidence in the bounding box, send it to the VLM.
    if element_type in text_elements and not base_text.strip():
        logger.info(f"[FLASH LANE] ⚡ Diverting '{element_type}' to VLM. Reason: Empty base_text from OCR.")
        return "GEMINI_FLASH"

    # RULE 1: The Fast Lane (Pure text, high confidence)
    if element_type in text_elements:
        if confidence >= 0.90:
            logger.info(f"[FAST LANE]  🟢 Bypassing VLM. Reason: '{element_type}' has high confidence ({confidence:.2f}).")
            return "SKIP_VLM"
        else:
            logger.info(f"[FLASH LANE] ⚡ Diverting '{element_type}' to VLM. Reason: Low OCR confidence ({confidence:.2f}).")
            return "GEMINI_FLASH"

    # RULE 2: The Flash Lane (Lightweight VLM for structured formatting)
    if element_type in flash_elements:
        logger.info(f"[FLASH LANE] ⚡ Routing to Flash. Reason: Element is a structured '{element_type}'.")
        return "GEMINI_FLASH"

    # RULE 3: The Pro Lane (Heavyweight VLM for complex visuals)
    if element_type in pro_elements:
        logger.info(f"[PRO LANE]   🧠 Routing to Pro. Reason: Element is a complex visual '{element_type}'.")
        return "GEMINI_PRO"

    # RULE 4: The Fallback
    # If MinerU updates and introduces a brand new label we don't recognize, default safely to Flash.
    logger.warning(f"[FLASH LANE] ⚠️ Unrecognized type '{element_type}'. Safely defaulting to Flash.")
    return "GEMINI_FLASH"