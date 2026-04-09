from pathlib import Path
from scipy.io import wavfile
import json
import re
import os
import torch
from transformers import AutoProcessor, Wav2Vec2Model
import joblib as jb
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import sklearn.pipeline as pipe
import sklearn.preprocessing as prep
import shutil
import convert_audio
import make_path_json
###################################
#   these are the classification  #
#   models I am going to use      #
###################################
import sklearn.svm as svm
import sklearn.linear_model as lin
import sklearn.neighbors as nb
import sklearn.ensemble as em
import sklearn.neural_network as nn

model_name = "facebook/wav2vec2-base"
processor = AutoProcessor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()

class_models = {
    "logreg": lin.LogisticRegression(max_iter=2000),
    "svm_linear": svm.SVC(kernel="linear"),
    "svm_rbf": svm.SVC(kernel="rbf"),
    "knn": nb.KNeighborsClassifier(n_neighbors=3),
    "rf": em.RandomForestClassifier(n_estimators=200, random_state=42),
    "mlp": nn.MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, solver="lbfgs", random_state=42),
}

def train(clean_data_json = None, classifier_path = None, raw_data_path = None, clean_data_path=None, prepare_data = True):
    
    # path to this script
    SCRIPT_DIR = Path(__file__).resolve().parent
    # project root (adjust depending on where script lives)
    PROJECT_ROOT = SCRIPT_DIR.parent
    Dataset = []
    # json file
    if clean_data_json is None:
        json_path = PROJECT_ROOT / "data" / "audio_paths.json"
    else:
        json_path = Path(clean_data_json)
    if classifier_path is None:
        CLASSIFIER_PATH = PROJECT_ROOT / "classifier"
    else:
        CLASSIFIER_PATH = Path(classifier_path)
    if raw_data_path is None:
        RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
    else:
        RAW_DATA_PATH = Path(raw_data_path)
    if clean_data_path is None:
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

    emb_train, emb_test, class_train, class_test = ms.train_test_split(
        embedding_data, data_class,
        test_size=0.2,
        random_state=42,
        stratify=data_class
    )
    print("split")
    
    try:
        shutil.rmtree(CLASSIFIER_PATH)
    except:
        pass
    os.mkdir(CLASSIFIER_PATH)
    for key in class_models:
        print("############################")
        print('')
        print("training " + key)
        print('')
        print("############################")
        clf = pipe.make_pipeline(
            prep.StandardScaler(),
            class_models[key]
        )

        clf.fit(emb_train, class_train)

        y_pred = clf.predict(emb_test)

        print("Accuracy:", metrics.accuracy_score(class_test, y_pred))
        print(metrics.classification_report(class_test, y_pred))

        jb.dump(clf, CLASSIFIER_PATH / (key + "_classifier.joblib"))
    # print(embedding_data)
    # print(data_class)
train()
    






