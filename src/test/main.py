from pathlib import Path
from scipy.io import wavfile
import json
import re
import torch
from transformers import AutoProcessor, Wav2Vec2Model
from scipy.signal import resample_poly

model_name = "facebook/wav2vec2-base"
processor = AutoProcessor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()

# path to this script
SCRIPT_DIR = Path(__file__).resolve().parent
# project root (adjust depending on where script lives)
PROJECT_ROOT = SCRIPT_DIR.parent
Dataset = []
# json file
json_path = PROJECT_ROOT.parent / "utils" / "audio_paths.json"
with open(json_path,'r') as f:
    audio_paths = json.load(f)
# reading in the data & creating a dataset
for key in audio_paths.keys():
    for path in audio_paths[key]:
        label = lambda text: re.split(r"[/]",text)[len(re.split(r"[/]",text))-1]
        samplerate, data = wavfile.read(PROJECT_ROOT.parent / path)
        Dataset.append({"class":key,
                            "label":label(path),
                             "path": str(PROJECT_ROOT.parent / path),
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
    embedding = last_hidden_state.mean(dim=1)
    embedding_data.append(embedding.numpy())
    data_class.append(d_set["class"])

print(embedding_data)
print(data_class)





