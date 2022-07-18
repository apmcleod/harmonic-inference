"""Script to annotate an input score with the harmonic inference model.
This does not evaluate the annotation based on a ground truth."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

import torch
from tqdm import tqdm

import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import (
    MODEL_CLASSES,
    HarmonicInferenceModel,
    add_joint_model_args,
    from_args,
)
from harmonic_inference.utils.data_utils import load_models_from_argparse, load_pieces

SPLITS = ["train", "valid", "test"]


def set_default_args(ARGS: argparse.Namespace):
    """
    Check if --defaults or --fh-defaults was given in the arguments, and if so,
    set the corresponding hyperparameters appropriately.

    Parameters
    ----------
    ARGS : argparse.Namespace
        The parsed command line arguments, which will be edited in place.
    """
    if ARGS.defaults:
        if ARGS.csm_version == 0:
            ARGS.min_chord_change_prob = 0.25
            ARGS.max_no_chord_change_prob = 0.45
            ARGS.max_chord_length = 8
            ARGS.min_key_change_prob = 0.1
            ARGS.max_no_key_change_prob = 0.75
            ARGS.beam_size = 50
            ARGS.max_chord_branching_factor = 5
            ARGS.target_chord_branch_prob = 0.75
            ARGS.max_key_branching_factor = 2
            ARGS.target_key_branch_prob = 0.5
            ARGS.hash_length = 5
            ARGS.ksm_exponent = 100

        elif ARGS.csm_version == 1:
            ARGS.min_chord_change_prob = 0.25
            ARGS.max_no_chord_change_prob = 0.5
            ARGS.max_chord_length = 8
            ARGS.min_key_change_prob = 0.05
            ARGS.max_no_key_change_prob = 0.75
            ARGS.beam_size = 50
            ARGS.max_chord_branching_factor = 5
            ARGS.target_chord_branch_prob = 0.75
            ARGS.max_key_branching_factor = 2
            ARGS.target_key_branch_prob = 0.5
            ARGS.hash_length = 5
            ARGS.ksm_exponent = 50

        elif ARGS.csm_version == 2 or ARGS.csm_version is None:
            ARGS.min_chord_change_prob = 0.25
            ARGS.max_no_chord_change_prob = 0.45
            ARGS.max_chord_length = 8
            ARGS.min_key_change_prob = 0.05
            ARGS.max_no_key_change_prob = 0.75
            ARGS.beam_size = 50
            ARGS.max_chord_branching_factor = 5
            ARGS.target_chord_branch_prob = 0.5
            ARGS.max_key_branching_factor = 2
            ARGS.target_key_branch_prob = 0.5
            ARGS.hash_length = 5
            ARGS.ksm_exponent = 50

        else:
            logging.error("--defaults is only valid with --csm-version 0, 1, or 2.")
            sys.exit(1)

    if ARGS.fh_defaults:
        if ARGS.csm_version == 0:
            ARGS.min_chord_change_prob = 0.3
            ARGS.max_no_chord_change_prob = 0.4
            ARGS.max_chord_length = 8
            ARGS.min_key_change_prob = 0.1
            ARGS.max_no_key_change_prob = 0.7
            ARGS.beam_size = 50
            ARGS.max_chord_branching_factor = 5
            ARGS.target_chord_branch_prob = 0.75
            ARGS.max_key_branching_factor = 2
            ARGS.target_key_branch_prob = 0.5
            ARGS.hash_length = 5
            ARGS.ksm_exponent = 30

        elif ARGS.csm_version == 1:
            ARGS.min_chord_change_prob = 0.3
            ARGS.max_no_chord_change_prob = 0.4
            ARGS.max_chord_length = 8
            ARGS.min_key_change_prob = 0.1
            ARGS.max_no_key_change_prob = 0.7
            ARGS.beam_size = 50
            ARGS.max_chord_branching_factor = 5
            ARGS.target_chord_branch_prob = 0.75
            ARGS.max_key_branching_factor = 2
            ARGS.target_key_branch_prob = 0.5
            ARGS.hash_length = 5
            ARGS.ksm_exponent = 30

        elif ARGS.csm_version == 2 or ARGS.csm_version is None:
            ARGS.min_chord_change_prob = 0.3
            ARGS.max_no_chord_change_prob = 0.4
            ARGS.max_chord_length = 8
            ARGS.min_key_change_prob = 0.1
            ARGS.max_no_key_change_prob = 0.7
            ARGS.beam_size = 50
            ARGS.max_chord_branching_factor = 5
            ARGS.target_chord_branch_prob = 0.75
            ARGS.max_key_branching_factor = 2
            ARGS.target_key_branch_prob = 0.5
            ARGS.hash_length = 5
            ARGS.ksm_exponent = 30

        else:
            logging.error("--fh-defaults is only valid with --csm-version 0, 1, or 2.")
            sys.exit(1)


def annotate(
    model: HarmonicInferenceModel,
    pieces: List[Piece],
    output_tsv_dir: Union[Path, str] = None,
):
    """
    Get estimated chords and keys on the given pieces using the given model.

    Parameters
    ----------
    model : HarmonicInferenceModel
        The model to use to estimate chords and keys.
    pieces : List[Piece]
        The input pieces to estimate chords and keys from.
    output_tsv_dir : Union[Path, str]
        A directory to output TSV labels into. Each piece's output labels will go into
        a sub-directory according to its name field. If None, label TSVs are not generated.
    """
    if output_tsv_dir is not None:
        output_tsv_dir = Path(output_tsv_dir)

    for piece in tqdm(pieces, desc="Getting harmony for pieces"):
        if piece.name is not None:
            logging.info("Running piece %s", piece.name)

        state = model.get_harmony(piece)

        if state is None:
            logging.info("Returned None")

        else:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                eu.log_state(state, piece, model.CHORD_OUTPUT_TYPE, model.KEY_OUTPUT_TYPE)

            annotation_df = eu.get_annotation_df(
                state,
                piece,
                model.CHORD_OUTPUT_TYPE,
                model.KEY_OUTPUT_TYPE,
                model.chord_classifier.use_inversions,
                model.chord_classifier.reduction,
                use_chord_pitches=True,
            )

            if piece.name is not None and output_tsv_dir is not None:
                piece_name = Path(piece.name.split(" ")[-1])
                output_tsv_path = output_tsv_dir / (piece_name.stem + ".tsv")

                try:
                    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
                    annotation_df.to_csv(output_tsv_path, sep="\t")
                    logging.info("Labels TSV written out to %s", output_tsv_path)
                except Exception:
                    logging.exception("Error writing to csv %s", output_tsv_path)
                    logging.debug(annotation_df)

            else:
                logging.debug(annotation_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate functional harmony labels given some data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv of MusicXML files.",
    )

    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Only evaluate the given file_id (from files.tsv).",
    )

    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help="The --input data refers to MusicXML files.",
    )

    parser.add_argument(
        "--defaults",
        action="store_true",
        help=(
            "Use the default hyperparameter settings from the internal-corpus-trained system, "
            "according to the chosen `--csm-version` (0, 1, or 2)."
        ),
    )

    parser.add_argument(
        "--fh-defaults",
        action="store_true",
        help=(
            "Use the default hyperparameter settings from the FH-corpus-trained system, "
            "according to the chosen `--csm-version` (0, 1, or 2)."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="outputs",
        help="The directory to write label tsvs and annotated MuseScore3 scores to.",
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
        "-l",
        "--log",
        type=str,
        default=sys.stderr,
        help=(
            "The log file to print messages to. If a file is given, it will be interpreted "
            "relative to `--output`."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logging information.",
    )

    parser.add_argument(
        "--threads",
        default=None,
        type=int,
        help="The number of pytorch cpu threads to create.",
    )

    add_joint_model_args(parser)

    ARGS = parser.parse_args()

    if ARGS.threads is not None:
        torch.set_num_threads(ARGS.threads)

    if ARGS.log is not sys.stderr:
        log_path = ARGS.output / ARGS.log
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.output / ARGS.log,
        level=logging.DEBUG if ARGS.verbose else logging.INFO,
        filemode="w",
    )

    # Load models
    models = load_models_from_argparse(ARGS)

    # Load pieces
    pieces = load_pieces(xml=ARGS.xml, input_path=ARGS.input, specific_id=ARGS.id)

    annotate(
        from_args(models, ARGS),
        pieces,
        output_tsv_dir=ARGS.output,
    )
