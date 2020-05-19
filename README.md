# Emotion Detection and Recognition from Text data

<p align="center">
<img src="https://devblogs.microsoft.com/cse/wp-content/uploads/sites/55/2015/11/Figure_6_emoticons_on_scale.png"  />
</p>

## Project Structure

```

├── README.md          <- README file.
├── api                <- APIs to interact with the inference model.
│   ├── example.py
|
├── data
│   ├── example.csv       <- raw data from third party sources.
|
├── docs               <- Project related analysis and other documents
│
├── models             <- Trained and serialized models/artifacts
|   |── v1
|       |── artifact.h5
|   |── v2
|       |── artifact.h5
│
├── notebooks          <- Data analysis Jupyter notebooks
│
├── requirements.txt   <- Pip generated requirements file for the project.
│
├── emotion_detection     <- Source code for use in this project.
│   ├── __init__.py
│   │
│   ├── config         <- Contains the config files.
│   │   └── config.py
|   |
│   ├── data           <- Scripts to download data and store on root data path.
│   │   └── make_dataset.py
|   |
│   ├── dispatcher     <- Collection of various ML models ready to dispatch.
│   │   └── dispatcher.py
│   │
│   ├── features       <- Scripts to process the data.
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train, test, and build model
│   │   │
│   │   ├── test_model.py
│   │   └── train_model.py
│   │   └── build_model.py
|   |
│   ├── utils          <- Collection of various utility functions.
|   |   └── example.py
|   |
│   ├── run_app.py      <- script to run the flask web app
│   ├── run.py          <- script to run the model training
│   ├── simple_inference.py     <- script to test the model on cli

```

## Getting Started

### Requirements

```
pip install -r requirements.txt
```

## Config File

Config file at `emotion_detection/config/config.py` contains all the necessary configurations. Please make sure to check that before preoceeding.

### Example:

```
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

DATASET_NAME = "ISEAR_dataset.csv"

DATASET_URL = <dataset-url>

MODEL_PATH = os.path.join(BASE_DIR, "models")

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")

```

### Dispatcher

All the available ML models should be listed in the `emotion_detection/dispatcher/dispatcher.py` file. This will be used as the `model-name` while training and testing.

Example:

```
MODELS = {
    "randomforest": ensemble.RandomForestClassifier,
    "naive_bayes": MultinomialNB,
    "xgboost": XGBClassifier,
    "logistic": LogisticRegression,
    "sgd_classifier": SGDClassifier,
    "svm_svc": SVC,
}
```

### Model parameters

Hyperparameters for the listed models are to be stored in the `emotion_detection/config/model_params.py` file with the same name as the listed models in dispatcher.

Example:

```
"bert_classifier": {},
"xgboost": {},
"randomforest": {},
"naive_bayes": {"alpha": 0.1},
```

### Download the dataset

The following command will download the dataset from the URL given in `src/config/config.py` file .

```
python -m emotion_detection.data.make_dataset
```

### Run

```
python run.py --model-name <model-name> --vocab-size <vocab-size> --train-size <train-size>
```

Example:

```
python run.py --model-name naiv_bayes --vocab-size 7000 --train-size 0.7
```

## Flask Web App

```
python run_app.py
```

## Try running modules seperately

### Train the model

The following command will train the model by first pre-processing the dataset from the `feature_generator.py` and train on the configured ML model.

```
python -m emotion_detection.models.train_model
```

### Test the model

```
python -m emotion_detection.models.test_model

```

## To-do List

- [x] Download dataset
- [x] Pre-process data
- [x] Train model
- [x] Test model
- [x] Main Pipeline
- [x] Flas app
