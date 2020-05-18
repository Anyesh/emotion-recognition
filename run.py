import argparse

parser = argparse.ArgumentParser(
    description="Arguments to train and test the emotion detection."
)

parser.add_argument(
    "--engine",
    action="store",
    type=str,
    required=True,
    help="Processor engine (Example: BERT, SKLEARN).",
)
parser.add_argument(
    "--input-size",
    action="store",
    type=int,
    required=True,
    help="Input size to pass on models.",
)
parser.add_argument(
    "--output-size",
    action="store",
    type=int,
    required=True,
    help="Output size to pass on models.",
)


args = parser.parse_args()
print(args)

if __name__ == "__main__":

    import emotion_detection.main as main

    main(processor_engine, model_name, input_shape, output_shape)
