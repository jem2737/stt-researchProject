import pyaudio
from pathlib import Path
import os
import numpy as np
from collections import deque
from transformers import AutoProcessor, Wav2Vec2Model
import torch
import joblib
def listen(classifier_path = None):
    CHUNK = 16000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if classifier_path is None:
        CLASSIFIER_MODEL = "logreg_classifier.joblib"
        clf = joblib.load(PROJECT_ROOT / "classifier" / CLASSIFIER_MODEL)
    else:
        CLASSIFIER_MODEL = Path(classifier_path)
        clf = joblib.load(CLASSIFIER_MODEL)
    

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
                prediction = clf.predict(embedding.reshape(1, -1))
                print(prediction)
                print(window.shape)   # should be (80000,)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()