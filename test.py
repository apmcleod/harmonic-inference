from typing import Dict, Iterable
from pathlib import Path
import logging
import argparse
import sys
import os
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import h5py

import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.key_transition_models as ktm
import harmonic_inference.models.key_sequence_models as ksm
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.piece import Piece, ScorePiece
import harmonic_inference.data.datasets as ds

SPLITS = ["train", "valid", "test"]

MODEL_CLASSES = {
    'ccm': ccm.SimpleChordClassifier,
    'ctm': ctm.SimpleChordTransitionModel,
    # 'csm': csm.SimpleChordSequenceModel,
    # 'ktm': ktm.SimpleKeyTransitionModel,
    # 'ksm': ksm.SimpleKeySequenceModel,
}

def evaluate(models: Dict, pieces: Iterable[Piece]):
    ctm_dataset = ds.ChordTransitionDataset(pieces)
    ctm_loader = DataLoader(
        ctm_dataset,
        batch_size=ds.ChordTransitionDataset.valid_batch_size,
        shuffle=False,
    )

    outputs = []
    lengths = []
    for batch in ctm_loader:
        output, length = models['ctm'].get_output(batch)
        outputs.extend(output.numpy())
        lengths.extend(length.numpy())
    outputs = np.vstack(outputs)
    lengths = np.array(lengths)


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
        "--checkpoint",
        type=str,
        default='checkpoints',
        help='The directory containing checkpoints for each type of model.',
    )

    for model in MODEL_CLASSES.keys():
        DEFAULT_PATH = os.path.join(
            '`--checkpoint`', model, 'lightning_logs', 'version_*', 'checkpoints', '*.ckpt'
        )
        parser.add_argument(
            f"--{model}-checkpoint",
            type=str,
            default=DEFAULT_PATH,
            help=f"The checkpoint file to load the {model} from."
        )

    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=sys.stderr,
        help="The log file to print messages to."
    )

    parser.add_argument(
        "-h5",
        "--h5_dir",
        default=Path("h5_data"),
        type=Path,
        help="The directory that holds the h5 data containing file_ids to test on.",
    )

    ARGS = parser.parse_args()

    if ARGS.log is not sys.stderr:
        logging.basicConfig(filename=ARGS.log, level=logging.INFO, filemode='w')

    # Load models
    models = {}
    for model_name, model_class in MODEL_CLASSES.items():
        DEFAULT_PATH = os.path.join(
            '`--checkpoint`', model_name, 'lightning_logs', 'version_*', 'checkpoints', '*.ckpt'
        )
        checkpoint_arg = getattr(ARGS, f'{model_name}_checkpoint')

        if checkpoint_arg == DEFAULT_PATH:
            checkpoint_arg = checkpoint_arg.replace("`--checkpoint`", ARGS.checkpoint)

        possible_checkpoints = list(glob(checkpoint_arg))
        if len(possible_checkpoints) == 0:
            logging.error(f'No checkpoints found for {model_name} in {checkpoint_arg}')
            sys.exit(2)

        if len(possible_checkpoints) == 1:
            checkpoint = possible_checkpoints[0]
            logging.info(f"Loading checkpoint {checkpoint} for {model_name}.")

        else:
            checkpoint = possible_checkpoints[-1]
            logging.info(f"Multiple checkpoints found for {model_name}. Loading {checkpoint}.")

        models[model_name] = model_class.load_from_checkpoint(checkpoint)
        models[model_name].freeze()

    # Load validation data for ctm
    h5_path = Path(ARGS.h5_dir / 'ChordTransitionDataset_valid_seed_0.h5')
    with h5py.File(h5_path, 'r') as h5_file:
        if 'file_ids' not in h5_file:
            logging.error(f'file_ids not found in {h5_path}. Re-create with create_h5_data.py')
            sys.exit(1)

        file_ids = list(h5_file['file_ids'])

    # Load pieces
    _, measures_df, _, _ = load_clean_corpus_dfs(ARGS.input)
    with open(Path(ARGS.h5_dir / 'pieces_valid_seed_0.pkl'), 'rb') as pkl_file:
        piece_dicts = pickle.load(pkl_file)

    pieces = []
    for file_id, piece_dict in tqdm(zip(file_ids, piece_dicts), desc='Loading Pieces'):
        pieces.append(
            ScorePiece(None, None, measures_df.loc[file_id], piece_dict=piece_dict)
        )

    evaluate(models, pieces)
