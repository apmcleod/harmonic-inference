from pathlib import Path
import argparse
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler

import harmonic_inference.data.datasets as ds
import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.key_transition_models as ktm
import harmonic_inference.models.key_sequence_models as ksm
from harmonic_inference.data.data_types import PitchType, PieceType


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train any of the submodules for harmonic inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=['ccm', 'ctm', 'csm', 'ktm', 'ksm'],
        help="The type of model to train.",
        required=True,
    )

    DEFAULT_CHECKPOINT_PATH = os.path.join('checkpoints', '`model`')
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help='The directory to save model checkpoints into.',
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

    ARGS = parser.parse_args()

    if ARGS.checkpoint == DEFAULT_CHECKPOINT_PATH:
        ARGS.checkpoint = os.path.join('checkpoints', ARGS.model)
        os.makedirs(ARGS.checkpoint, exist_ok=True)

    if ARGS.model == 'ccm':
        model = ccm.SimpleChordClassifier(
            PitchType.TPC,
            PitchType.TPC,
            use_inversions=True,
            learning_rate=ARGS.lr,
        )
        dataset = ds.ChordClassificationDataset
    elif ARGS.model == 'ctm':
        model = ctm.SimpleChordTransitionModel(PieceType.SCORE, learning_rate=ARGS.lr)
        dataset = ds.ChordTransitionDataset
    elif ARGS.model == 'csm':
        model = csm.SimpleChordSequenceModel(PitchType.TPC, learning_rate=ARGS.lr)
        dataset = ds.ChordSequenceDataset
    elif ARGS.model == 'ktm':
        model = ktm.SimpleKeyTransitionModel(PitchType.TPC, learning_rate=ARGS.lr)
        dataset = ds.KeyTransitionDataset
    elif ARGS.model == 'ksm':
        model = ksm.SimpleKeySequenceModel(PitchType.TPC, PitchType.TPC, learning_rate=ARGS.lr)
        dataset = ds.KeySequenceDataset

    h5_path_train = Path(ARGS.h5_dir / f'{dataset.__name__}_train_seed_0.h5')
    h5_path_valid = Path(ARGS.h5_dir / f'{dataset.__name__}_valid_seed_0.h5')

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
            output_filename=os.path.join(ARGS.checkpoint, 'profile.log')
        )

    trainer = pl.Trainer(
        default_root_dir=ARGS.checkpoint,
        profiler=ARGS.profile,
        early_stop_callback=True
    )
    trainer.fit(model, dl_train, dl_valid)
