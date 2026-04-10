import pyaudio
from pathlib import Path
import os
import numpy as np
from collections import deque
from transformers import AutoProcessor, Wav2Vec2Model
import torch
import joblib
import tkinter as tk
import threading
# root = tk.Tk()
# root.title("Audio Status")
# root.geometry("300x200")
# label = tk.Label(root, text="WAITING", font=("Arial", 24), width=20, height=10)
# label.pack(fill="both", expand=True)
# COLOR_MAP = {
#     "distressed": "red",
#     "stressed": "orange",
#     "normal": "green",
#     "crowd": "gray",
#     "silence": "gray",
#     }
# def update_status(prediction):
#     color = COLOR_MAP.get(prediction.lower(), "gray")
#     label.config(text=prediction.upper(), bg=color)
def listen(spk_classifier = None, type_classifier = None):
    
    CHUNK = 16000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if spk_classifier is None:
        CLASSIFIER_MODEL = "logreg_classifier.joblib"
        clf_spk = joblib.load(PROJECT_ROOT / "classifier" / "raw_speaker" / CLASSIFIER_MODEL)
    else:
        CLASSIFIER_MODEL = spk_classifier
        clf_spk = joblib.load(CLASSIFIER_MODEL)
        
    if type_classifier is None:
        CLASSIFIER_MODEL = "logreg_classifier.joblib"
        clf_type = joblib.load(PROJECT_ROOT / "classifier" / "raw" / CLASSIFIER_MODEL)
    else:
        CLASSIFIER_MODEL = type_classifier
        clf_type = joblib.load(CLASSIFIER_MODEL)
    

    model_name = "facebook/wav2vec2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    audio_data = deque(maxlen=80000)
    print("listening")
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_data.extend(samples)

            if len(audio_data) == 80000:
                window = np.array(audio_data, dtype=np.float32)
                inputs = processor(
                    window,
                    sampling_rate=RATE,
                    return_tensors="pt",
                    padding=False
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state      # [batch, time, hidden]
                embedding = last_hidden_state.mean(dim=1).squeeze(0)
                spk_prediction = clf_spk.predict(embedding.reshape(1, -1))
                if spk_prediction[0] == "speaker":
                    class_prediction = clf_type.predict(embedding.reshape(1, -1))
                    print(class_prediction)
                else:
                    print("no speaker detected")
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()
listen(type_classifier="/Users/jamesmcdonald/Documents/stt-researchProject/classifier/raw/svm_linear_classifier.joblib")


# threading.Thread(target=listen, daemon=True).start()
# root.mainloop()