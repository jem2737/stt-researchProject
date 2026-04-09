from pathlib import Path
import os
import shutil
import librosa
import soundfile as sf
import numpy as np
import make_path_json
# path to this script
def convert_audio(raw_data_path=None,clean_data_path=None):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    # DATA_PATH = PROJECT_ROOT / "data"
    # project root (adjust depending on where script lives)
    RAW_AUDIO_PATH = Path(raw_data_path)
    CLEAN_AUDIO_PATH = Path(clean_data_path)
    if CLEAN_AUDIO_PATH.exists():
        shutil.rmtree(CLEAN_AUDIO_PATH)

    CLEAN_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    data_path = os.listdir(RAW_AUDIO_PATH)
    for folder in data_path:
        if folder.startswith("."):
            continue
        (CLEAN_AUDIO_PATH / folder).mkdir(exist_ok=True)
        class_path = os.listdir(RAW_AUDIO_PATH / folder)
        for file in class_path:
            if file.startswith("."):
                continue
            file_name = Path(file).stem
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

