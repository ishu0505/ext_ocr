import logging

logger = logging.getLogger("Traffic_Control")

def get_routing_decision(element_type: str, confidence: float, base_text: str = "") -> str:
    """Evaluates the element to determine the optimal processing lane."""
    element_type = element_type.lower().strip()
    
    text_elements = ['text', 'title', 'header', 'footer', 'paragraph', 'text_inline']
    flash_elements = ['table', 'equation', 'list', 'form', 'isolated_formula']
    pro_elements = ['figure', 'chart', 'image', 'complex_diagram']

    logger.debug(f"Evaluating -> Type: '{element_type}', Confidence: {confidence}, Text Length: {len(base_text)}")

    # 1. The Fast Lane or VLM Text Lane
    if element_type in text_elements:
        if confidence >= 0.90 and base_text.strip():
            return "SKIP_VLM"
        else:
            logger.info(f"[TEXT LANE] ⚡ Diverting '{element_type}' to VLM. Requesting raw text output.")
            return "VLM_TEXT"

    # 2. The Table Lane (Structured Data)
    if element_type in flash_elements:
        logger.info(f"[TABLE LANE] ⚡ Routing to Flash. Requesting structured JSON.")
        return "VLM_TABLE"

    # 3. The Visual Lane (Complex Reasoning)
    if element_type in pro_elements:
        logger.info(f"[VISUAL LANE] 🧠 Routing to Pro. Requesting structured JSON.")
        return "VLM_VISUAL"

    # 4. Fallback
    logger.warning(f"[FALLBACK] ⚠️ Unrecognized type '{element_type}'. Defaulting to Table extraction.")
    return "VLM_TABLE"