import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("Document_Builder")

def _json_to_markdown_table(json_data: Any) -> str:
    """
    Natively converts a list of dictionaries (or similar JSON structures) 
    into a formatted Markdown table string without needing Pandas.
    """
    if not json_data:
        return "*Empty Data*"
        
    # Normalize input: if Gemini returned {"revenue_data": [{"Q1": 100}, {"Q2": 200}]}, extract the list
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                json_data = value
                break
        else:
            json_data = [json_data] # Wrap a single flat dictionary into a list of 1 row
            
    # Fallback for completely unexpected JSON schemas
    if not isinstance(json_data, list) or not all(isinstance(row, dict) for row in json_data):
        import json
        return f"```json\n{json.dumps(json_data, indent=2)}\n```"

    if len(json_data) == 0:
        return "*Empty Table*"

    # 1. Extract all unique headers across all rows (in case some rows are missing keys)
    headers = []
    for row in json_data:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    
    # 2. Build markdown table strings
    header_row = "| " + " | ".join(str(h).replace("|", "\\|") for h in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    
    table_lines = [header_row, separator_row]
    
    for row in json_data:
        # Get the value, convert to string, remove line breaks, and escape pipe characters
        row_vals = [str(row.get(h, "")).replace("\n", " ").replace("|", "\\|") for h in headers]
        row_str = "| " + " | ".join(row_vals) + " |"
        table_lines.append(row_str)
        
    return "\n".join(table_lines)


def assemble_final_document(elements: List[Dict[str, Any]], original_filename: str, output_dir: str = "data_ext/outputs") -> str:
    """
    Stitches processed elements into a single, cohesive Markdown document.
    Injects bounding box coordinates as HTML tags for UI highlighting.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(original_filename)[0]
    md_filename = os.path.join(output_dir, f"{base_name}_final.md")
    
    # 1. Sort the elements to ensure human reading order
    # Sort hierarchy: Page Number -> Y-coordinate (Top to Bottom) -> X-coordinate (Left to Right)
    logger.info("Sorting elements to reconstruct reading order...")
    elements.sort(key=lambda e: (e.get('page', 1), e.get('bbox', [0, 0, 0, 0])[1], e.get('bbox', [0, 0, 0, 0])[0]))
    
    markdown_lines = [f"# Extracted Document: {original_filename}\n"]
    table_counter = 1

    # 2. Iterate and Assemble
    for idx, element in enumerate(elements):
        el_type = element.get('type', 'text')
        content = element.get('final_content', '')
        bbox = element.get('bbox', [0, 0, 0, 0])
        bbox_str = ",".join(str(round(coord, 1)) for coord in bbox)
        
        # Determine if human review was flagged in Phase 4
        warning_flag = " ⚠️ **[REQUIRES HUMAN REVIEW]**" if element.get('flagged_for_human') else ""
        
        # --- Handle Standard Text Elements ---
        if isinstance(content, str):
            if el_type in ['title', 'header']:
                markdown_lines.append(f"## <span data-bbox=\"{bbox_str}\">{content}</span>{warning_flag}\n")
            else:
                markdown_lines.append(f"<span data-bbox=\"{bbox_str}\">{content}</span>{warning_flag}\n")
                
        # --- Handle Structured Data (JSON/Dict/List from Gemini) ---
        elif isinstance(content, (dict, list)):
            logger.info(f"Formatting VLM structured data found at element {idx} into Markdown table...")
            
            markdown_lines.append(f"\n### Table {table_counter} (Extracted via Gemini VLM){warning_flag}")
            markdown_lines.append(f"<div data-bbox=\"{bbox_str}\">\n")
            markdown_lines.append(_json_to_markdown_table(content))
            markdown_lines.append("\n</div>\n")
            
            table_counter += 1
                
        else:
            logger.warning(f"Unknown content type for element {idx}. Skipping.")

    # 3. Write the final Markdown file to disk
    try:
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_lines))
        logger.info(f"✅ Successfully assembled final document: {md_filename}")
    except Exception as e:
        logger.error(f"Failed to write markdown file: {e}")

    return md_filename