from pathlib import Path
import os
import re
import shutil
import librosa
import soundfile as sf
import numpy as np
# path to this script
SCRIPT_DIR = Path(__file__).resolve().parent
# project root (adjust depending on where script lives)
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" 
RAW_AUDIO_PATH = PROJECT_ROOT / "data" / "raw"
CLEAN_AUDIO_PATH = PROJECT_ROOT / "data" / "clean_data"
try:
    shutil.rmtree(CLEAN_AUDIO_PATH)
except:
    pass

os.mkdir(CLEAN_AUDIO_PATH)
data_path = os.listdir(RAW_AUDIO_PATH)
for folder in data_path:
    os.mkdir(CLEAN_AUDIO_PATH / folder)
    if not folder.startswith("."):
        class_path = os.listdir(PROJECT_ROOT / "data" / "raw" / folder)
        for file in class_path:
            if not file.startswith("."):
                file_name = re.split(r"[.]",file)[0]
                input_file = str(RAW_AUDIO_PATH / folder / file)
                output_file = str(CLEAN_AUDIO_PATH / folder / (file_name + ".wav"))
                # print(output_file)
                # convert the audio
                audio, sr = librosa.load(input_file, sr=16000, mono=True)
                # Make sure dtype is float32
                audio = audio.astype(np.float32)
                # Save as WAV
                sf.write(output_file, audio, 16000, subtype="FLOAT")
                print(file_name + ".wav fixed")
# print(a)
