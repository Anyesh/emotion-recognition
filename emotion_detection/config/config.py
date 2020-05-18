import os


## Dir Config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")

## Dataset Config
DATASET_NAME = "ISEAR_dataset.csv"

DATASET_URL = "https://www.floydhub.com/api/v1/resources/qM4BHN3pNjkfkvYjMMtjU4/ISEAR.csv?content=true&rename=isearcsv"


VOCAB_SIZE = MAX_LEN = 512

## BERT config
# if os.environ.get("USE_BERT"):

# import transformers

# TRAIN_BATCH_SIZE = 3
# VALID_BATCH_SIZE = 3
# EPOCHS = 10
# BERT_PATH = os.path.join(DATA_PATH, "external", "bert-base-uncased")
# MODEL_PATH = os.path.join(CHECKPOINT_PATH, "model.bin")
# TRAINING_FILE = os.path.join(DATA_PATH, DATASET_NAME)
# TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
# HIDDEN_SIZE = 768
# NUM_LABELS = 7
