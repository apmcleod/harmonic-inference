import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import h5py
import harmonic_inference.data.datasets as ds
import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.initial_chord_models as icm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
import pytorch_lightning as pl
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.data_types import PieceType, PitchType
from harmonic_inference.data.piece import ScorePiece
from harmonic_inference.models.joint_model import MODEL_CLASSES
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train any of the submodules for harmonic inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODEL_CLASSES.keys(),
        help="The type of model to train.",
        required=True,
    )

    DEFAULT_CHECKPOINT_PATH = os.path.join("checkpoints", "`model`")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="The directory to save model checkpoints into.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv files. Used only for -m icm",
    )

    parser.add_argument(
        "-h5",
        "--h5_dir",
        default=Path("h5_data"),
        type=Path,
        help="The directory that holds the h5 data to train on.",
    )

    parser.add_argument(
        "-w",
        "--workers",
        default=4,
        type=int,
        help="The number of workers per DataLoader.",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run a profiler during training. The results are saved to "
        f"{os.path.join('`--checkpoint`', 'profile.log')}",
    )

    parser.add_argument(
        "-lr",
        default=0.001,
        type=float,
        help="The learning rate to use.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="The seed used when generating the h5_data.",
    )

    ARGS = parser.parse_args()

    if ARGS.checkpoint == DEFAULT_CHECKPOINT_PATH:
        ARGS.checkpoint = os.path.join("checkpoints", ARGS.model)
        os.makedirs(ARGS.checkpoint, exist_ok=True)

    if ARGS.model == "ccm":
        model = ccm.SimpleChordClassifier(
            PieceType.SCORE,
            PitchType.TPC,
            use_inversions=True,
            learning_rate=ARGS.lr,
        )
        dataset = ds.ChordClassificationDataset
    elif ARGS.model == "ctm":
        model = ctm.SimpleChordTransitionModel(PieceType.SCORE, learning_rate=ARGS.lr)
        dataset = ds.ChordTransitionDataset
    elif ARGS.model == "csm":
        model = csm.SimpleChordSequenceModel(PitchType.TPC, learning_rate=ARGS.lr)
        dataset = ds.ChordSequenceDataset
    elif ARGS.model == "ktm":
        model = ktm.SimpleKeyTransitionModel(PitchType.TPC, learning_rate=ARGS.lr)
        dataset = ds.KeyTransitionDataset
    elif ARGS.model == "ksm":
        model = ksm.SimpleKeySequenceModel(PitchType.TPC, PitchType.TPC, learning_rate=ARGS.lr)
        dataset = ds.KeySequenceDataset
    elif ARGS.model == "icm":
        # Load training data for ctm, just to get file_ids
        h5_path = Path(ARGS.h5_dir / f"ChordTransitionDataset_train_seed_{ARGS.seed}.h5")
        with h5py.File(h5_path, "r") as h5_file:
            if "file_ids" not in h5_file:
                logging.error(f"file_ids not found in {h5_path}. Re-create with create_h5_data.py")
                sys.exit(1)

            file_ids = list(h5_file["file_ids"])

        # Load pieces
        files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

        # Load from pkl if available
        pkl_path = Path(ARGS.h5_dir / f"pieces_train_seed_{ARGS.seed}.pkl")
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
                    ScorePiece(
                        notes_df.loc[file_id], chords_df.loc[file_id], measures_df.loc[file_id]
                    )
                )

        chords = [piece.get_chords()[0] for piece in pieces]
        icm.SimpleInitialChordModel.train(
            chords,
            os.path.join(ARGS.checkpoint, "initial_chord_prior.json"),
            use_inversions=True,
            add_n_smoothing=1.0,
        )
        sys.exit(0)

    else:
        logging.error(f"Invalid model: {ARGS.model}")
        sys.exit(1)

    h5_path_train = Path(ARGS.h5_dir / f"{dataset.__name__}_train_seed_{ARGS.seed}.h5")
    h5_path_valid = Path(ARGS.h5_dir / f"{dataset.__name__}_valid_seed_{ARGS.seed}.h5")

    dataset_train = ds.h5_to_dataset(h5_path_train, dataset, transform=torch.from_numpy)
    dataset_valid = ds.h5_to_dataset(h5_path_valid, dataset, transform=torch.from_numpy)

    dl_train = DataLoader(
        dataset_train,
        batch_size=dataset.train_batch_size,
        shuffle=True,
        num_workers=ARGS.workers,
    )
    dl_valid = DataLoader(
        dataset_valid,
        batch_size=dataset.valid_batch_size,
        shuffle=False,
        num_workers=ARGS.workers,
    )

    if ARGS.profile:
        ARGS.profile = AdvancedProfiler(
            output_filename=os.path.join(ARGS.checkpoint, "profile.log")
        )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
    )

    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir=ARGS.checkpoint,
        profiler=ARGS.profile,
        callbacks=[early_stopping_callback, lr_logger],
    )
    trainer.fit(model, dl_train, dl_valid)
