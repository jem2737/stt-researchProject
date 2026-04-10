from pathlib import Path
from scipy.io import wavfile
import json
import re
from transformers import AutoProcessor, Wav2Vec2Model
import convert_audio
import torch
import make_path_json

model_name = "facebook/wav2vec2-base"
processor = AutoProcessor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()

def embed(raw_data_path = None, clean_data_path=None, clean_data_json = None, prepare_data = True, speaker_test = False):
    # path to this script
    SCRIPT_DIR = Path(__file__).resolve().parent
    # project root (adjust depending on where script lives)
    PROJECT_ROOT = SCRIPT_DIR.parent
    Dataset = []
    # json file
    if clean_data_json is None:
        if speaker_test:
            json_path = PROJECT_ROOT / "data" / "audio_paths_speaker.json"
        else:
            json_path = PROJECT_ROOT / "data" / "audio_paths.json"
    else:
        json_path = Path(clean_data_json)

    if raw_data_path is None:
        if speaker_test:
            RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_speaker"
        else:
            RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
    else:
        RAW_DATA_PATH = Path(raw_data_path)
    if clean_data_path is None:
        if speaker_test:
            CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean_data_speaker"
        else:
            CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean_data"
    else:
        CLEAN_DATA_PATH = Path(clean_data_path)
    if prepare_data:
        convert_audio.convert_audio(RAW_DATA_PATH,CLEAN_DATA_PATH)
        make_path_json.make_path_json(CLEAN_DATA_PATH,json_path)
    with open(json_path,'r') as f:
        audio_paths = json.load(f)
    # reading in the data & creating a dataset
    for key in audio_paths.keys():
        for path in audio_paths[key]:
            label = lambda text: re.split(r"[/]",text)[len(re.split(r"[/]",text))-1]
            samplerate, data = wavfile.read(Path(path))
            Dataset.append({"class":key,
                                "label":label(path),
                                "path": str(Path(path)),
                                "samplerate": samplerate,
                                "data":data})
    embedding_data = []
    data_class = []
    for d_set in Dataset:
        sr = d_set["samplerate"]
        data = d_set["data"]
        inputs = processor(
            data,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False
        )
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state      # [batch, time, hidden]
        embedding = last_hidden_state.mean(dim=1).squeeze(0)
        embedding_data.append(embedding.numpy())
        data_class.append(d_set["class"])

    return [embedding_data, data_class]
# embed(speaker_test=True)
