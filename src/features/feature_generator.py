import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import text
from sklearn.preprocessing import MultiLabelBinarizer
from .text_processors import KerasTextPreprocessor, TFIDFProcessor
from src.config import config
import pickle


def process_data(VOCAB_SIZE, processor_engine="keras", train_size=0.8):

    """ Process the text data with respect to the processor_engine

    This function will process the text data from the given processor and store them
    in the checkpoints directory.

    Parameters
    -----------


    Returns
    --------
    

    """

    df = pd.read_csv(
        os.path.join(config.DATA_PATH, config.DATASET_NAME),
        names=["#", "emotions", "texts"],
        header=None,
    )
    data = df.dropna()
    data = shuffle(data, random_state=0)

    _train_size = int(len(data) * train_size)

    print(f"[INFO] Splitting data of size {len(data)} by {train_size} train size..")

    train_data = data["texts"].values[:_train_size]
    test_data = data["texts"].values[_train_size:]

    print(f"[INFO] Train size:{train_data.shape} & Test size: {test_data.shape}")

    if processor_engine == "keras":

        processor = KerasTextPreprocessor(VOCAB_SIZE)

        ## Keras does not have stop words remover
        ## so passing dataframe instead values to process
        processor.create_tokenizer(data["texts"])

        body_train = processor.transform_text(train_data)
        body_test = processor.transform_text(test_data)

    elif processor_engine == "tfidf":
        processor = TFIDFProcessor(feature_size=VOCAB_SIZE)

        processor.create_vocab(data["texts"].values)

        body_train = processor.transform_text(train_data)
        body_test = processor.transform_text(test_data)

    else:
        raise "[ERROR] No processor engine!!"

    labels_split = [labels.split(",") for labels in data["emotions"].values]

    label_encoder = MultiLabelBinarizer()
    label_encoded = label_encoder.fit_transform(labels_split)

    train_labels = label_encoded[:_train_size]
    test_labels = label_encoded[_train_size:]

    print("[INFO] Saving processor..")
    with open(
        os.path.join(
            config.CHECKPOINT_PATH, f"{processor_engine}-data_processor_state.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(processor, f)

    print("[INFO] Saved data processor..")
    with open(os.path.join(config.CHECKPOINT_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print("[INFO] Saved label encoder..")

    return body_train, train_labels, body_test, test_labels


if __name__ == "__main__":
    process_data(5000, processor_engine="tfidf")
