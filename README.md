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
│   │   └── example.py

```

## Getting Started

### Requirements

```
pip install -r requirements.txt
```

### Download the dataset

The following command will download the dataset from the URL given in `src/config/config.py` file .

```
python -m emotion_detection.data.make_dataset
```

### Run

```
python -m emotion_detection.main
```

OR

```
./run.sh
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
- [ ] Main Pipeline
- [ ] Build model - artifact building
