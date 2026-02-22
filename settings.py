import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("Settings")

def setup_environment():
    """
    Configures environment variables and Google Service Account credentials.
    Loads from a local .env file for testing, or standard env vars in production.
    """
    # Load local variables from .env if it exists
    load_dotenv()

    # 1. Process the Service Account JSON
    google_creds_json = os.environ.get("GOOGLE_CREDS_JSON")
    
    if google_creds_json:
        # Define where the temp file will live (using a safe home directory path for local testing)
        creds_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", 
            str(Path.home().joinpath(".google_creds_ext_ocr.json"))
        )
        
        # If the file doesn't exist yet, write the JSON string into it
        if not os.path.exists(creds_path):
            try:
                creds_data = json.loads(google_creds_json)
                os.makedirs(os.path.dirname(creds_path), exist_ok=True)
                with open(creds_path, "w") as f:
                    json.dump(creds_data, f)
                logger.info(f"🔐 Google ADC credentials written to {creds_path}")
            except Exception as e:
                logger.error(f"Failed to parse or write GOOGLE_CREDS_JSON: {e}")
        
        # Explicitly set the environment variable so the google-genai SDK finds it
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    else:
        logger.warning("GOOGLE_CREDS_JSON not found. Relying on existing ADC config if present.")

    # 2. Set Project Defaults
    if "GOOGLE_PROJECT_ID" not in os.environ:
        os.environ["GOOGLE_PROJECT_ID"] = "gfsa-422417" 
        
    if "GOOGLE_PROJECT_LOCATION" not in os.environ:
        os.environ["GOOGLE_PROJECT_LOCATION"] = "us-central1"