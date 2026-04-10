from pathlib import Path
import os
from transformers import AutoProcessor, Wav2Vec2Model
import joblib as jb
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import sklearn.pipeline as pipe
import sklearn.preprocessing as prep
import shutil
###################################
#   these are the classification  #
#   models I am going to use      #
###################################
import sklearn.svm as svm
import sklearn.linear_model as lin
import sklearn.neighbors as nb
import sklearn.ensemble as em
import sklearn.neural_network as nn
import embed
from datetime import datetime

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

def write_results(log_file, model_name, classifier ,y_true, y_pred):
    with open(log_file, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model: {classifier}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Accuracy: {metrics.accuracy_score(y_true, y_pred)}\n")
        f.write(metrics.classification_report(y_true, y_pred))
        f.write("\n")

def train(raw_data_path = None, clean_data_path=None, classifier_dir_path = None, clean_data_json = None, prepare_data = True):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    LOG_DIR = SCRIPT_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    if clean_data_json is None:
        json_path = PROJECT_ROOT / "data" / "audio_paths.json"
    else:
        json_path = Path(clean_data_json)
    if classifier_dir_path is None:
        CLASSIFIER_DIR_PATH = PROJECT_ROOT / "classifier"
    else:
        CLASSIFIER_DIR_PATH = Path(classifier_dir_path)
    if raw_data_path is None:
        RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
    else:
        RAW_DATA_PATH = Path(raw_data_path)
    if clean_data_path is None:
        CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean_data"
    else:
        CLEAN_DATA_PATH = Path(clean_data_path)
    Datasets = {}
    Datasets['raw'] = embed.embed(raw_data_path,clean_data_path,clean_data_json,prepare_data,speaker_test=False)
    Datasets['raw_speaker'] = embed.embed(raw_data_path,clean_data_path,clean_data_json,prepare_data,speaker_test=True)
    # print(Datasets['raw'][1])
    # print(Datasets['raw_speaker'][1])
    # print(Datasets.keys())
    try:
        shutil.rmtree(CLASSIFIER_DIR_PATH)
    except:
        pass
    os.mkdir(CLASSIFIER_DIR_PATH)
    for d_key in Datasets.keys():
        embedding_data = Datasets[d_key][0]
        data_class = Datasets[d_key][1]
        emb_train, emb_test, class_train, class_test = ms.train_test_split(
            embedding_data, data_class,
            test_size=0.2,
            random_state=42,
            stratify=data_class
        )    
        os.mkdir(CLASSIFIER_DIR_PATH / d_key)
        for model_key in class_models:
            clf = pipe.make_pipeline(
                prep.StandardScaler(),
                class_models[model_key]
            )

            clf.fit(emb_train, class_train)

            y_pred = clf.predict(emb_test)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if d_key == "raw":
                log_file = LOG_DIR / f"emotion_training_{timestamp}.txt"
            if d_key == "raw_speaker":
                log_file = LOG_DIR / f"speaker_training_{timestamp}.txt"
            write_results(log_file, model_name, model_key, class_test, y_pred)
            jb.dump(clf, CLASSIFIER_DIR_PATH/ d_key / (model_key + "_classifier.joblib"))
# train()

    






