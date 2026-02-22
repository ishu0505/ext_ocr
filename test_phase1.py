import json
import logging
import subprocess

from layout_ocr_ext.mineru_service import process_with_mineru

# Setup verbose console logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase1_Test")


def verify_gpu_environment():
    """Probes the Fedora system for the RTX 4060 and its memory state."""
    logger.info("--- Step 0: System Environment Check ---")
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        gpu_stats = result.stdout.strip()
        logger.info(f"NVIDIA GPU Detected: {gpu_stats}")
    except FileNotFoundError:
        logger.warning(
            "nvidia-smi not found. Are the NVIDIA proprietary drivers installed?"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to query GPU: {e}")


if __name__ == "__main__":
    verify_gpu_environment()

    test_pdf = "data_ext/uploads/sample_pdf.pdf"
    logger.info("--- Step 1: Initiating Pipeline ---")
    logger.info(f"Target file: {test_pdf}")

    elements = process_with_mineru(test_pdf)

    logger.info("--- Step 2: Extraction Results ---")
    if not elements:
        logger.error("Extraction returned 0 elements.")
    else:
        logger.info(f"Successfully extracted {len(elements)} structural elements.")
        logger.info("Preview of the first extracted element:")
        print(json.dumps(elements[0], indent=2))
