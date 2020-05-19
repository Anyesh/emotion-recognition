import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# from tensorflow.keras.preprocessing import text
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from .text_processors import TFIDFProcessor
from sklearn.model_selection import train_test_split
from emotion_detection.utils.tokenizer import nltk_tokenizer_df
import pickle


def process_data(
    dataset_path, dataset_name, vocab_size, train_size, processor_engine="SKLEARN",
):

    """ Process the text data with respect to the processor_engine

    This function will process the text data from the given processor and store them
    in the checkpoints directory.

    Parameters
    -----------
     dataset_path
     dataset_name
     vocab_size
     train_size
     processor_engine


    Returns
    --------
    train_data
    y_train
    test_data
    y_test
    processor
    label_encoder
    

    """

    df = pd.read_csv(
        os.path.join(dataset_path, dataset_name),
        names=["#", "emotions", "texts"],
        header=None,
    )

    df["texts"] = df["texts"].apply(nltk_tokenizer_df)
    data = df.dropna()
    # data = shuffle(data, random_state=train_size, 32)
    # = int(len(data) * train_size)

    print(f"[INFO] Splitting data of size {len(data)} by {train_size} train size..")

    # train_data = data["texts"].values[:_train_size]
    # test_data = data["texts"].values[_train_size:]
    labels = data["emotions"].values
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data["texts"],
        label_encoded,
        train_size=train_size,
        shuffle=True,
        random_state=32,
    )

    print(f"[INFO] Train size:{X_train.shape} & Test size: {X_test.shape}")

    # if processor_engine == "KERAS":

    #     processor = KerasTextPreprocessor(vocab_size)

    #     processor.create_tokenizer(data["texts"].values)

    #     train_data = processor.transform_text(X_train)
    #     test_data = processor.transform_text(X_test)

    if processor_engine == "SKLEARN":
        processor = TFIDFProcessor(feature_size=vocab_size)

        processor.create_vocab(data["texts"].values)

        train_data = processor.transform_text(X_train)
        test_data = processor.transform_text(X_test)

    else:
        raise "[ERROR] No processor engine!!"

    # train_labels = label_encoded[:_train_size]
    # test_labels = label_encoded[_train_size:]
    return train_data, y_train, test_data, y_test, processor, label_encoder


if __name__ == "__main__":
    process_data(
        dataset_path=config.DATA_PATH,
        dataset_name=config.DATASET_NAME,
        vocab_size=400,
        processor_engine="KERAS",
    )
