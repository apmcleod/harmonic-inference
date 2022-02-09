"""Create an h5 data file containing scheduled sampling from one model."""

import argparse
import sys
from pathlib import Path

from harmonic_inference.models.joint_model import MODEL_CLASSES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a scheduled sampling dataset, from a given model and already "
            "created h5 data files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODEL_CLASSES.keys(),
        help="The type of model to use to generate the scheduled sampling inputs.",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help=(
            "The directory to load the model checkpoint from, within a subdirectory of the "
            "model's name (e.g., CCM checkpoints will be loaded from `--checkpoint`/ccm)."
        ),
    )

    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help=(
            "Specify a version number to load the model from. If given, "
            "the model will be loaded from `--checkpoint`/`--model`/version_`--model-version`"
        ),
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("h5_data"),
        help="The directory containing the non-scheduling sampled h5 data.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("h5_data"),
        help="The directory to save the created scheduling sampled h5 data into.",
    )

    parser.add_argument(
        "-l", "--log", type=str, default=sys.stderr, help="The log file to print messages to."
    )

    ARGS = parser.parse_args()
