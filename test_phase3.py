import os
import json
import logging
from PIL import Image, ImageDraw, ImageText
from ex_vlm_extractor.gemini_service import extract_with_gemini

# Setup verbose console logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)

# 🛑 INSERT YOUR GOOGLE AI STUDIO API KEY HERE FOR TESTING 🛑
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

def create_dummy_crop(filename="dummy_table.png"):
    """Generates a quick test image with some numbers."""
    img = Image.new('RGB', (300, 150), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Q1 Revenue: $45,000", fill=(0,0,0))
    d.text((10,50), "Q2 Revenue: $52,000", fill=(0,0,0))
    d.text((10,90), "Status: Exceeded Expectations", fill=(0,0,0))
    img.save(filename)
    return filename

if __name__ == "__main__":
    print("\n--- Testing Phase 3: Gemini VLM Extraction ---\n")
    
    # 1. Create a fake cropped bounding box
    test_image_path = create_dummy_crop()
    logging.info(f"Created temporary test image at {test_image_path}")
    
    # 2. Simulate context extracted from MinerU in Phase 1
    miner_u_context = "The following table shows quarterly revenue for 2026. All currency is in USD."
    
    try:
        # 3. Test the Flash Lane
        flash_result = extract_with_gemini(
            route_model="GEMINI_FLASH",
            image_path=test_image_path,
            context_text=miner_u_context
        )
        
        print("\n--- ⚡ Gemini Flash Output ---")
        print(json.dumps(flash_result, indent=2))
        
    except Exception as e:
        logging.error(f"Extraction failed. Did you insert your API key? Error: {e}")
        
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)