from src.dispatcher import dispatcher
from src.features import feature_generator
from src.config import config
import numpy as np
import os
import pickle


VOCAB_SIZE = config.VOCAB_SIZE
MODEL_NAME = "naive_bayes"
model = dispatcher.MODELS[MODEL_NAME]


if os.path.isfile(os.path.join(config.CHECKPOINT_PATH, "frozen_data.npy")):
    print("[INFO] Loading saved data..")
    train_data, train_label, test_data, test_label = np.load(
        os.path.join(config.CHECKPOINT_PATH, "frozen_data.npy"), allow_pickle=True
    )
else:
    print("[INFO] No saved data found.. Processing data..")
    train_data, train_label, test_data, test_label = feature_generator.process_data(
        VOCAB_SIZE, processor_engine="keras"
    )
    np.save(
        os.path.join(config.CHECKPOINT_PATH, "frozen_data"),
        [train_data, train_label, test_data, test_label],
    )

print("[INFO] Processing data complete!!")


# print("[INFO] Building a model..")
# clf = model._build()
print("[INFO] Training model..")
clf = model.fit(train_data, train_label)
# clf.fit(train_data, train_label, epochs=10, batch_size=16, validation_split=0.1)
# clf.evaluate(test_data, test_label, batch_size=16)

print("[INFO] Saving model..")
# clf.save(os.path.join(config.CHECKPOINT_PATH, "model_v1.h5"))

with open(
    os.path.join(config.CHECKPOINT_PATH, f"{MODEL_NAME}-model_v1.pkl"), "wb",
) as f:
    pickle.dump(clf, f)
