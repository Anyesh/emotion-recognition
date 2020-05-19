import argparse


def train_size_limit(x):
    """ Limit train size
    """

    x = float(x)
    if x < 0.6:
        raise argparse.ArgumentTypeError("Minimum train size 0.6")
    return x


parser = argparse.ArgumentParser(
    description="Arguments to train and test the emotion detection."
)

parser.add_argument(
    "--model-name",
    action="store",
    type=str,
    required=True,
    help="Model name to train on (Example: naive_bayes, xgboost.. Look at the model availabe at dispatcher.)",
)
parser.add_argument(
    "--vocab-size",
    action="store",
    type=int,
    required=True,
    help="Vocab or feature size of the data. Exaple(400, 512, 1000...)",
)
parser.add_argument(
    "--train-size",
    action="store",
    type=train_size_limit,
    help="Train size to split. Example(0.7, 0.8..)",
)


args = parser.parse_args()


if __name__ == "__main__":

    from emotion_detection.main import train_pipeline

    train_pipeline(args.model_name, args.vocab_size, args.train_size)
