import glob
import json

# Find the exact JSON file we just generated
json_files = glob.glob(
    "data_ext/outputs/mineru_temp/sample_pdf/sample_pdf/hybrid_auto/*_middle.json"
)

if not json_files:
    print("Could not find the middle.json file.")
else:
    file_path = json_files[0]
    print(f"Reading: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n--- Top Level Keys ---")
    print(list(data.keys()))

    # If 'pdf_info' exists, let's look inside the first item
    if "pdf_info" in data and len(data["pdf_info"]) > 0:
        print("\n--- Keys inside pdf_info[0] ---")
        print(list(data["pdf_info"][0].keys()))
