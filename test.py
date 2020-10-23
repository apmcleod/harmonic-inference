import argparse
import logging
import os
import pickle
import sys
from glob import glob
from pathlib import Path
from typing import List

import h5py
from tqdm import tqdm

import harmonic_inference.models.initial_chord_models as icm
import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.piece import Piece, ScorePiece
from harmonic_inference.models.joint_model import (
    MODEL_CLASSES,
    HarmonicInferenceModel,
    add_joint_model_args,
    from_args,
)

SPLITS = ["train", "valid", "test"]


def evaluate(model: HarmonicInferenceModel, pieces: List[Piece]):
    states = []
    for piece in tqdm(pieces, desc="Getting harmony for pieces"):
        state = model.get_harmony(piece)
        print(eu.evaluate(piece, state))

    print(states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a harmonic inference model on some data.",
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
            f"--{model}-checkpoint",
            type=str,
            default=DEFAULT_PATH,
            help=f"The checkpoint file to load the {model} from.",
        )

    parser.add_argument(
        "--icm-json",
        type=str,
        default=os.path.join("checkpoints", "icm", "initial_chord_prior.json"),
        help="The json file to load the icm from.",
    )

    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=sys.stderr,
        help="The log file to print messages to.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logging information.",
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

    add_joint_model_args(parser)

    ARGS = parser.parse_args()

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.log,
        level=logging.DEBUG if ARGS.verbose else logging.INFO,
        filemode="w",
    )

    # Load models
    models = {}
    for model_name, model_class in MODEL_CLASSES.items():
        if model_name == "icm":
            continue

        DEFAULT_PATH = os.path.join(
            "`--checkpoint`", model_name, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        )
        checkpoint_arg = getattr(ARGS, f"{model_name}_checkpoint")

        if checkpoint_arg == DEFAULT_PATH:
            checkpoint_arg = checkpoint_arg.replace("`--checkpoint`", ARGS.checkpoint)

        possible_checkpoints = list(glob(checkpoint_arg))
        if len(possible_checkpoints) == 0:
            logging.error(f"No checkpoints found for {model_name} in {checkpoint_arg}")
            sys.exit(2)

        if len(possible_checkpoints) == 1:
            checkpoint = possible_checkpoints[0]
            logging.info(f"Loading checkpoint {checkpoint} for {model_name}.")

        else:
            checkpoint = possible_checkpoints[-1]
            logging.info(f"Multiple checkpoints found for {model_name}. Loading {checkpoint}.")

        models[model_name] = model_class.load_from_checkpoint(checkpoint)
        models[model_name].freeze()

    # Load icm json differently
    models["icm"] = icm.SimpleInitialChordModel(ARGS.icm_json)

    # Load validation data for ctm
    h5_path = Path(ARGS.h5_dir / "ChordTransitionDataset_valid_seed_0.h5")
    with h5py.File(h5_path, "r") as h5_file:
        if "file_ids" not in h5_file:
            logging.error(f"file_ids not found in {h5_path}. Re-create with create_h5_data.py")
            sys.exit(1)

        file_ids = list(h5_file["file_ids"])

    # Load pieces
    files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

    # Load from pkl if available
    pkl_path = Path(ARGS.h5_dir / "pieces_valid_seed_0.pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as pkl_file:
            piece_dicts = pickle.load(pkl_file)
        pieces = [
            ScorePiece(None, None, measures_df.loc[file_id], piece_dict=piece_dict)
            for file_id, piece_dict in zip(file_ids, piece_dicts)
        ]

    # Generate from dfs
    else:
        pieces = []
        for file_id in tqdm(file_ids, desc="Loading Pieces"):
            pieces.append(
                ScorePiece(notes_df.loc[file_id], chords_df.loc[file_id], measures_df.loc[file_id])
            )

    evaluate(from_args(models, ARGS), pieces)
