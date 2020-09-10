from pathlib import Path
import logging
import argparse
import sys

import numpy as np

from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
import harmonic_inference.data.datasets as ds
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.key_transition_models as ktm
import harmonic_inference.models.key_sequence_models as ksm
from harmonic_inference.data.data_types import PitchType, PieceType

SPLITS = ["train", "valid", "test"]

MODEL_CLASSES = {
    'ccm': ccm,
    'ctm': ctm,
    'csm': csm,
    'ktm': ktm,
    'ksm': ksm,
}

def evaluate(piece, ccm_model):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a harmonic inference model on some data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help='The directory containing the raw corpus_data tsv files.',
    )

    parser.add_argument(
        "--ccm-checkpoint",
        type=str,
        required=True,
        help="The checkpoint file to load the CCM from."
    )

    # parser.add_argument(
    #     "--ctm-checkpoint",
    #     type=str,
    #     required=True,
    #     help="The checkpoint file to load the CTM from."
    # )

    # parser.add_argument(
    #     "--csm-checkpoint",
    #     type=str,
    #     required=True,
    #     help="The checkpoint file to load the CSM from."
    # )

    # parser.add_argument(
    #     "--ktm-checkpoint",
    #     type=str,
    #     required=True,
    #     help="The checkpoint file to load the KTM from."
    # )

    # parser.add_argument(
    #     "--ksm-checkpoint",
    #     type=str,
    #     required=True,
    #     help="The checkpoint file to load the KSM from."
    # )

    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=sys.stderr,
        help="The log file to print messages to."
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="The random seed to use to create the data."
    )

    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help=f"The proportions for splits {SPLITS}. These values will be normalized.",
    )

    ARGS = parser.parse_args()

    # Validate and normalize splits
    if any([split_prop < 0  for split_prop in ARGS.splits]):
        print("--split values cannot be negative.", file=sys.stderr)
        sys.exit(1)
    sum_splits = sum(ARGS.splits)
    if sum_splits == 0:
        print("At least one --split value must be positive.", file=sys.stderr)
        sys.exit(1)
    ARGS.splits = [split / sum_splits for split in ARGS.splits]

    if ARGS.log is not sys.stderr:
        logging.basicConfig(filename=ARGS.log, level=logging.INFO, filemode='w')
    files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

    if ARGS.seed is None:
        ARGS.seed = np.random.randint(0, 2 ** 32)

    split_ids, split_pieces = ds.get_split_file_ids_and_pieces(
        files_df,
        measures_df,
        chords_df,
        notes_df,
        splits=ARGS.splits,
        seed=ARGS.seed,
    )

    # Use validation data
    ids = split_ids[1]
    pieces = split_pieces[1]

    ccm_model = ccm.load_from_checkpoint(ARGS.ccm_checkpoint)
    # ctm_model = ctm.load_from_checkpoint(ARGS.ctm_checkpoint)
    # csm_model = csm.load_from_checkpoint(ARGS.csm_checkpoint)
    # ktm_model = ktm.load_from_checkpoint(ARGS.ktm_checkpoint)
    # ksm_model = ksm.load_from_checkpoint(ARGS.ksm_checkpoint)

    # trainer = pl.Trainer(
    #     default_root_dir=ARGS.checkpoint,
    #     profiler=ARGS.profile,
    #     early_stop_callback=True
    # )
    # trainer.fit(model, dl_train, dl_valid)

    for piece in pieces:
        ds.ChordTransitionDataset([pieces])
        evaluate(piece, ccm_model)
