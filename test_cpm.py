"""Test the CPM by taking as input the GT pieces and estimating pitches."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.data_types import TRIAD_REDUCTION
from harmonic_inference.data.datasets import ChordPitchesDataset
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.chord_pitches_models import ChordPitchesModel, decode_cpm_outputs
from harmonic_inference.models.joint_model import (
    CPM_CHORD_TONE_THRESHOLD_DEFAULT,
    CPM_NON_CHORD_TONE_ADD_THRESHOLD_DEFAULT,
    CPM_NON_CHORD_TONE_REPLACE_THRESHOLD_DEFAULT,
    add_joint_model_args,
)
from harmonic_inference.utils.data_utils import load_models_from_argparse, load_pieces


def evaluate_cpm(
    cpm: ChordPitchesModel,
    pieces: List[Piece],
    output_tsv_dir: Union[str, Path] = None,
    cpm_chord_tone_threshold: float = CPM_CHORD_TONE_THRESHOLD_DEFAULT,
    cpm_non_chord_tone_add_threshold: float = CPM_NON_CHORD_TONE_ADD_THRESHOLD_DEFAULT,
    cpm_non_chord_tone_replace_threshold: float = CPM_NON_CHORD_TONE_REPLACE_THRESHOLD_DEFAULT,
) -> None:
    """
    _summary_

    Parameters
    ----------
    cpm : ChordPitchesModel
        The CPM to evaluate.
    pieces : List[Piece]
        A List of pieces to run the CPM on.
    output_tsv_dir : Union[str, Path]
        The directory to write out label tsvs to.
    cpm_chord_tone_threshold : float
        The threshold above which a default chord tone must reach in the CPM output
        in order to be considered present in a given chord.
    cpm_non_chord_tone_add_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to be an added tone in a given chord.
    cpm_non_chord_tone_replace_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to replace a chord tone in a given chord.
    """
    for piece in tqdm(pieces):
        dataset = ChordPitchesDataset([piece], **cpm.get_dataset_kwargs())
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        outputs = []
        for batch in dl:
            outputs.extend(cpm.get_output(batch).numpy())

        chord_pitches = decode_cpm_outputs(
            np.vstack(outputs),
            np.vstack([chord.get_chord_pitches_target_vector() for chord in piece.get_chords()]),
            np.vstack(
                [
                    chord.get_chord_pitches_target_vector(reduction=TRIAD_REDUCTION)
                    for chord in piece.get_chords()
                ]
            ),
            [TRIAD_REDUCTION[chord.chord_type] for chord in piece.get_chords()],
            cpm_chord_tone_threshold,
            cpm_non_chord_tone_add_threshold,
            cpm_non_chord_tone_replace_threshold,
            cpm.INPUT_PITCH,
        )

        # TODO: Post-process and evaluate
        print(chord_pitches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a cpm on some data.",
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
        "-o",
        "--output",
        type=Path,
        default="outputs",
        help="The directory to write label tsvs to.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="The directory containing checkpoints for each type of model.",
    )

    DEFAULT_PATH = os.path.join(
        "`--checkpoint`", "cpm", "lightning_logs", "version_*", "checkpoints", "*.ckpt"
    )
    parser.add_argument(
        "--cpm",
        type=str,
        default=DEFAULT_PATH,
        help="A checkpoint file to load the cpm from.",
    )

    parser.add_argument(
        "--cpm-version",
        type=int,
        default=None,
        help=(
            "Specify a version number to load the model from. If given, --cpm is ignored"
            " and the cpm will be loaded from " + DEFAULT_PATH.replace("_*", "_`--cpm-version`")
        ),
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

    add_joint_model_args(parser, cpm_only=True)

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
    cpm = load_models_from_argparse(ARGS, model_type="cpm")["cpm"]

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
        input_path=ARGS.input,
        piece_dicts_path=Path(ARGS.h5_dir / f"pieces_{data_type}_seed_{ARGS.seed}.pkl"),
        file_ids=file_ids,
    )

    evaluate_cpm(
        cpm,
        pieces,
        output_tsv_dir=ARGS.output,
        cpm_chord_tone_threshold=ARGS.cpm_chord_tone_threshold,
        cpm_non_chord_tone_add_threshold=ARGS.cpm_non_chord_tone_add_threshold,
        cpm_non_chord_tone_replace_threshold=ARGS.cpm_non_chord_tone_replace_threshold,
    )
