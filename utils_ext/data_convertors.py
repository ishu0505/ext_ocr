import pandas as pd
import logging
from typing import Union, List, Dict
import os

logger = logging.getLogger("Data_Converter")

def json_to_dataframe(json_data: Union[List, Dict]) -> pd.DataFrame:
    """
    Safely converts a JSON array (or nested dictionary) from Gemini into a Pandas DataFrame.
    """
    try:
        # If Gemini returned a list of dictionaries (standard table format)
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
            
        # If Gemini returned a single dictionary with a nested list (e.g., {"revenue_data": [...]})
        elif isinstance(json_data, dict):
            # Try to find the first list inside the dictionary to use as rows
            for key, value in json_data.items():
                if isinstance(value, list):
                    df = pd.DataFrame(value)
                    logger.info(f"Unpacked nested list from key: '{key}'")
                    break
            else:
                # Fallback: treat the single dictionary as one row
                df = pd.DataFrame([json_data])
        else:
            raise ValueError(f"Unsupported JSON format for DataFrame conversion: {type(json_data)}")
            
        return df
        
    except Exception as e:
        logger.error(f"Failed to convert JSON to DataFrame: {e}")
        # Return an empty DataFrame so the pipeline doesn't completely crash
        return pd.DataFrame()

def save_dataframe_to_csv(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """
    Saves the Pandas DataFrame to a CSV file on disk.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping CSV export.")
        return ""
        
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    try:
        # index=False prevents pandas from writing row numbers (0, 1, 2...) into the file
        df.to_csv(file_path, index=False)
        logger.info(f"💾 Saved structured data to CSV: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save CSV to {file_path}: {e}")
        return ""