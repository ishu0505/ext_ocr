import logging
import os
from utils_ext.pdf_cropper import crop_pdf_element

# Setup verbose console logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)

if __name__ == "__main__":
    print("\n--- Testing Phase 3: PDF Bounding Box Cropper ---\n")
    
    # The target PDF we used in Phase 1
    test_pdf = "data_ext/uploads/sample_pdf.pdf"
    
    # The exact data from your MinerU log output!
    page_number = 1
    target_bbox = [222, 58, 372, 75]
    
    if not os.path.exists(test_pdf):
        logging.error(f"Cannot find {test_pdf}. Please make sure the file is there.")
    else:
        try:
            # Execute the crop
            output_image = crop_pdf_element(
                pdf_path=test_pdf,
                page_num=page_number,
                bbox=target_bbox,
                dpi=300 # High res for the AI
            )
            
            print(f"\n✅ Success! The cropped image is saved at: {output_image}")
            print("Go open that file in your file explorer to verify it captured the title perfectly!")
            
        except Exception as e:
            logging.error(f"Test failed: {e}")