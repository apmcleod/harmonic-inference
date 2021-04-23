import argparse
import itertools
import os
from pathlib import Path

from harmonic_inference.models.joint_model import MODEL_CLASSES, add_joint_model_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a list of commands for a grid search. "
            "Most arguments are simply passed through to the commands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv files.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests on the actual test set, rather than the validation set.",
    )

    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help=(
            "The --input data comes from the funtional-harmony repository, as MusicXML "
            "files and labels CSV files."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="output",
        help="The directory in which to write the grid search results to (in subdirectories).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="The directory containing checkpoints for each type of model.",
    )

    for model in MODEL_CLASSES.keys():
        if model == "icm":
            continue

        DEFAULT_PATH = os.path.join(
            "`--checkpoint`", model, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        )
        parser.add_argument(
            f"--{model}",
            type=str,
            default=DEFAULT_PATH,
            help=f"A checkpoint file to load the {model} from.",
        )

        parser.add_argument(
            f"--{model}-version",
            type=int,
            default=None,
            help=(
                f"Specify a version number to load the model from. If given, --{model} is ignored"
                f" and the {model} will be loaded from "
                + DEFAULT_PATH.replace("_*", f"_`--{model}-version`")
            ),
        )

    parser.add_argument(
        "--icm-json",
        type=str,
        default=os.path.join("`--checkpoint`", "icm", "initial_chord_prior.json"),
        help="The json file to load the icm from.",
    )

    parser.add_argument(
        "-h5",
        "--h5_dir",
        default=Path("h5_data"),
        type=Path,
        help=(
            "The directory that holds the h5 data containing file_ids to test on, and the piece "
            "pkl files."
        ),
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="The seed used when generating the h5_data.",
    )

    parser.add_argument(
        "--threads",
        default=None,
        type=int,
        help="The number of pytorch cpu threads to create.",
    )

    add_joint_model_args(parser, grid_search=True)

    ARGS = parser.parse_args()

    # Create arguments that won't change during the grid search
    static_args = []

    if ARGS.threads is not None:
        static_args.append("--threads")
        static_args.append(str(ARGS.threads))
    del ARGS.threads

    if ARGS.seed != 0:
        static_args.append("--seed")
        static_args.append(str(ARGS.seed))
    del ARGS.seed

    if ARGS.h5_dir != Path("h5_data"):
        static_args.append("-h5")
        static_args.append(str(ARGS.h5_dir))
    del ARGS.h5_dir

    if ARGS.icm_json != os.path.join("`--checkpoint`", "icm", "initial_chord_prior.json"):
        static_args.append("--icm-json")
        static_args.append(ARGS.icm_json)
    del ARGS.icm_json

    if ARGS.checkpoint != "checkpoints":
        static_args.append("--checkpoint")
        static_args.append(ARGS.checkpoint)
    del ARGS.checkpoint

    if ARGS.xml:
        static_args.append("-x")
    del ARGS.xml

    if ARGS.test:
        static_args.append("--test")
    del ARGS.test

    if ARGS.input != Path("corpus_data"):
        static_args.append("--input")
        static_args.append(str(ARGS.input))
    del ARGS.input

    for model_name in MODEL_CLASSES.keys():
        if model_name == "icm":
            continue

        DEFAULT_PATH = os.path.join(
            "`--checkpoint`", model_name, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        )
        checkpoint_arg = getattr(ARGS, model_name)
        version_arg = getattr(ARGS, f"{model_name}_version")

        if version_arg is not None:
            static_args.append(f"--{model_name}-version")
            static_args.append(str(version_arg))

        elif checkpoint_arg != DEFAULT_PATH:
            static_args.append(f"--{model_name}")
            static_args.append(checkpoint_arg)

        delattr(ARGS, model_name)
        delattr(ARGS, f"{model_name}_version")

    argument_names = []
    argument_names_abbr = []
    argument_values = []
    for arg_name, arg_val in vars(ARGS).items():
        if arg_name.startswith("output"):
            continue

        argument_names.append(arg_name)
        argument_names_abbr.append("".join(name[0] for name in arg_name.split("_")))
        argument_values.append(arg_val if isinstance(arg_val, list) else [arg_val])

    for values in itertools.product(*argument_values):

        DIR_NAME = "_".join(
            ["-".join([name, str(val)]) for name, val in zip(argument_names_abbr, values)]
        )

        this_args = [arg for arg in static_args]
        this_args.append("--output")
        this_args.append(str(ARGS.output / DIR_NAME))

        this_args.append("--log")
        this_args.append(str(ARGS.output / DIR_NAME / "test.log"))

        for name, val in zip(argument_names, values):
            this_args.append(f"--{name.replace('_', '-')}")
            this_args.append(str(val))

        print(f"python test.py {' '.join(this_args)}")
