from pathlib import Path
import os
import json

SCRIPT_DIR = Path(__file__).resolve().parent
# project root (adjust depending on where script lives)
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" 
CLEAN_AUDIO_PATH = PROJECT_ROOT / "data" / "clean_data"
JSON_PATH = SCRIPT_DIR / "audio_paths.json"
print(JSON_PATH)
json_dict = {}
class_dir = os.listdir(CLEAN_AUDIO_PATH)
for folder in class_dir:
    if not folder.startswith("."):
        json_dict[folder] = []
        class_file = os.listdir(CLEAN_AUDIO_PATH / folder)
        for file in class_file:
            json_dict[folder].append(str(CLEAN_AUDIO_PATH / folder / file))

with open(JSON_PATH, "w") as f:
    json.dump(json_dict, f, indent=4)
