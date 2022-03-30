"""This script is used to quickly grid search over cpm hyperparameters."""

import argparse
import logging
import os
import sys
from bisect import bisect_left
from fractions import Fraction
from pathlib import Path
from typing import List, Union

import h5py
import pandas as pd
import torch
from tqdm import tqdm

import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.data_types import ChordType, KeyMode, PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import (
    MODEL_CLASSES,
    HarmonicInferenceModel,
    add_joint_model_args,
    from_args,
)
from harmonic_inference.utils.beam_search_utils import State
from harmonic_inference.utils.data_utils import load_models_from_argparse, load_pieces
from harmonic_inference.utils.harmonic_utils import get_chord_one_hot_index, get_key_one_hot_index

SPLITS = ["train", "valid", "test"]


def load_state_from_results_tsv(
    results_tsv_path: Union[str, Path], piece: Piece, model: HarmonicInferenceModel
) -> State:
    """
    Load a State (as if it's the results of a beam search) from a results_tsv.

    Parameters
    ----------
    results_tsv_path : Union[str, Path]
        The output results_tsv from which to load the State.
    piece : Piece
        The piece which corresponds to this results_tsv.
    model : HarmonicInferenceModel
        The model, used to ensure the correct label -> chord/key id mapping.

    Returns
    -------
    state : State
        The State which would've produced the given results_tsv.
    """
    results_df = pd.read_csv(results_tsv_path, sep="\t", index_col=0)
    results_df["new_state"] = results_df["est_chord"] != results_df["est_chord"].shift(1)

    chord_ids = []
    key_ids = []
    change_indexes = []

    duration = 0
    starting_positions = [0] + list(piece.get_duration_cache().cumsum())

    for _, row in results_df.iterrows():
        if row["new_state"]:
            chord_ids.append(
                get_chord_one_hot_index(
                    ChordType[row["est_chord_type"].split(".")[1]],
                    row["est_root"],
                    model.CHORD_OUTPUT_TYPE,
                    inversion=row["est_inversion"],
                    use_inversion=model.chord_classifier.use_inversions,
                    reduction=model.chord_classifier.reduction,
                )
            )
            key_ids.append(
                get_key_one_hot_index(
                    KeyMode[row["est_mode"].split(".")[1]],
                    row["est_tonic"],
                    model.KEY_OUTPUT_TYPE,
                )
            )
            change_indexes.append(bisect_left(starting_positions, duration))

        duration += Fraction(row["duration"])

    change_indexes.append(len(piece.get_duration_cache()))

    state = State(key=key_ids[0])
    for chord_id, key_id, change_id in zip(chord_ids, key_ids, change_indexes[1:]):
        state = State(chord=chord_id, key=key_id, change_index=change_id, prev_state=state)

    return state


def evaluate(
    model: HarmonicInferenceModel,
    pieces: List[Piece],
    output_tsv_dir: Union[Path, str],
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
    for piece in tqdm(pieces, desc="Loading harmony for pieces"):
        if piece.name is not None:
            logging.info("Evaluating piece %s", piece.name)

        piece_name = Path(piece.name.split(" ")[-1])
        output_tsv_path = output_tsv_dir / piece_name
        results_tsv_path = output_tsv_path.parent / (output_tsv_path.name[:-4] + "_results.tsv")

        state = load_state_from_results_tsv(results_tsv_path, piece, model)
        model.load_piece(piece)
        model.cpm_post_processing(state)

        # Create results dfs
        results_df = eu.get_results_df(
            piece,
            state,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            PitchType.TPC,
            PitchType.TPC,
        )

        # Perform evaluations
        chord_acc_with_pitches = eu.evaluate_features(results_df, ["chord", "pitches"])
        chord_acc_full = eu.evaluate_features(results_df, ["chord"])
        chord_acc_no_inv = eu.evaluate_features(results_df, ["chord_type", "root"])
        chord_acc_triad = eu.evaluate_features(results_df, ["triad", "root", "inversion"])
        chord_acc_triad_no_inv = eu.evaluate_features(results_df, ["triad", "root"])
        chord_acc_root_only = eu.evaluate_features(results_df, ["root"])

        pitch_acc_default = eu.evaluate_features(
            results_df,
            ["pitches"],
            filter=(
                results_df["gt_is_default"]
                & (results_df["gt_root"] == results_df["est_root"])
                & (results_df["gt_chord_type"] == results_df["est_chord_type"])
            ),
        )
        pitch_acc_non_default = eu.evaluate_features(
            results_df,
            ["pitches"],
            filter=(
                ~results_df["gt_is_default"]
                & (results_df["gt_root"] == results_df["est_root"])
                & (results_df["gt_chord_type"] == results_df["est_chord_type"])
            ),
        )

        logging.info("Chord accuracy = %s", chord_acc_full)
        logging.info("Chord accuracy with pitches = %s", chord_acc_with_pitches)
        logging.info("Chord accuracy, no inversions = %s", chord_acc_no_inv)
        logging.info("Chord accuracy, triads = %s", chord_acc_triad)
        logging.info("Chord accuracy, triad, no inversions = %s", chord_acc_triad_no_inv)
        logging.info("Chord accuracy, root only = %s", chord_acc_root_only)

        logging.info(
            "Chord pitch accuracy on correct chord root+type, GT default only = %s",
            pitch_acc_default,
        )
        logging.info(
            "Chord pitch accuracy on correct chord root+type, GT non-default only = %s",
            pitch_acc_non_default,
        )

        key_acc_full = eu.evaluate_features(results_df, ["key"])
        key_acc_tonic = eu.evaluate_features(results_df, ["tonic"])

        logging.info("Key accuracy = %s", key_acc_full)
        logging.info("Key accuracy, tonic only = %s", key_acc_tonic)

        full_acc_with_pitches = eu.evaluate_features(results_df, ["chord", "key", "pitches"])
        full_acc = eu.evaluate_features(results_df, ["chord", "key"])
        logging.info("Full accuracy = %s", full_acc)
        logging.info("Full accuracy with pitches = %s", full_acc_with_pitches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cpm hyperparameters from a given output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="outputs",
        help="The directory to read in outputs from.",
        required=True,
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
        help="Outputs are from the actual test set, rather than the validation set.",
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
        "-h5",
        "--h5_dir",
        default=Path("h5_data"),
        type=Path,
        help=(
            "The directory that holds the h5 data containing file_ids, and the piece " "pkl files."
        ),
    )

    parser.add_argument(
        "--no_h5",
        action="store_true",
        help="Do not read file_ids and piece_dicts from the `--h5_dir` directory.",
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

    add_joint_model_args(parser, cpm_only=True)

    ARGS = parser.parse_args()

    if ARGS.threads is not None:
        torch.set_num_threads(ARGS.threads)

    if ARGS.log is not sys.stderr:
        log_path = ARGS.output / ARGS.log
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.output / ARGS.log,
        level=logging.INFO,
        filemode="w",
    )

    # Load models
    models = load_models_from_argparse(ARGS)

    data_type = "test" if ARGS.test else "valid"

    # Load data for ctm to get file_ids
    h5_path = Path(ARGS.h5_dir / f"ChordTransitionDataset_{data_type}_seed_{ARGS.seed}.h5")
    if not ARGS.no_h5 and h5_path.exists():
        with h5py.File(h5_path, "r") as h5_file:
            if "file_ids" not in h5_file:
                logging.error("file_ids not found in %s. Re-create with create_h5_data.py", h5_path)
                sys.exit(1)

            file_ids = list(h5_file["file_ids"])
    else:
        file_ids = None

    # Load pieces
    pieces = load_pieces(
        xml=ARGS.xml,
        input_path=ARGS.input,
        piece_dicts_path=None
        if ARGS.no_h5
        else Path(ARGS.h5_dir / f"pieces_{data_type}_seed_{ARGS.seed}.pkl"),
        file_ids=file_ids,
    )

    evaluate(from_args(models, ARGS, cpm_only=True), pieces, ARGS.output)
