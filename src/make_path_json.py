from pathlib import Path
import os
import json
def make_path_json(clean_data_path=None,json_path=None):
    SCRIPT_DIR = Path(__file__).resolve().parent
    # project root (adjust depending on where script lives)
    PROJECT_ROOT = SCRIPT_DIR.parent
    CLEAN_AUDIO_PATH = Path(clean_data_path)
    JSON_PATH = Path(json_path)
    print(JSON_PATH)
    json_dict = {}
    class_dir = os.listdir(CLEAN_AUDIO_PATH)
    for folder in class_dir:
        if folder.startswith("."):
            continue
        json_dict[folder] = []
        class_file = os.listdir(CLEAN_AUDIO_PATH / folder)
        for file in class_file:
            json_dict[folder].append(str(CLEAN_AUDIO_PATH / folder / file))

    with open(JSON_PATH, "w") as f:
        json.dump(json_dict, f, indent=4)
