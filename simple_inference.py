import argparse


parser = argparse.ArgumentParser(description="Arguments to test the emotion detection.")

parser.add_argument(
    "--model-name",
    action="store",
    type=str,
    required=True,
    help="Model name to test on (Example: naive_bayes, xgboost.. Look at the model availabe at dispatcher.)",
)


args = parser.parse_args()


if __name__ == "__main__":

    from emotion_detection.main import test_pipeline

    input_text = input("Enter your text: ")

    output, _, _ = test_pipeline(args.model_name, input_text)
    print(f"{input_text} - predicted {output[0]}.")
