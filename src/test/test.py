import wave
import sys
import json 

from vosk import Model, KaldiRecognizer, SetLogLevel

# You can set log level to -1 to disable debug messages
SetLogLevel(0)
 
wf = wave.open("src/test/test.wav")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)

model = Model("src/models/vosk-model-en-us-0.22-lgraph")

# You can also init model by name or with a folder path
# model = Model(model_name="vosk-model-en-us-0.21")
# model = Model("models/en")

rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print(result["text"])
        #print(rec.Result())
    else:
        result = json.loads(rec.PartialResult())
        if not result["partial"] == "":
            print(result["partial"])
        #print(rec.PartialResult())

#print(rec.FinalResult())
