from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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

    parser.add_argument(
        "-h5",
        "--h5_dir",
        default=Path("h5_data"),
        type=Path,
        help="The directory that holds the h5 data to train on.",
    )

    ARGS = parser.parse_args()

    if ARGS.model == 'ccm':
        model = ccm.SimpleChordClassifier(PitchType.TPC, PitchType.TPC, use_inversions=True)
        dataset = ds.ChordClassificationDataset
    elif ARGS.model == 'ctm':
        model = ctm.SimpleChordTransitionModel(PieceType.SCORE)
        dataset = ds.ChordTransitionDataset
    elif ARGS.model == 'csm':
        model = csm.SimpleChordSequenceModel(PitchType.TPC)
        dataset = ds.ChordSequenceDataset
    elif ARGS.model == 'ktm':
        model = ktm.SimpleKeyTransitionModel(PitchType.TPC)
        dataset = ds.KeyTransitionDataset
    elif ARGS.model == 'ksm':
        model = ksm.SimpleKeySequenceModel(PitchType.TPC, PitchType.TPC)
        dataset = ds.KeySequenceDataset

    h5_path = Path(ARGS.h5_dir / f'{dataset.__name__}_train_seed_0.h5')
    h5_path_valid = Path(ARGS.h5_dir / f'{dataset.__name__}_valid_seed_0.h5')

    dataset_train = ds.h5_to_dataset(h5_path, dataset, transform=torch.from_numpy)
    dataset_valid = ds.h5_to_dataset(h5_path_valid, dataset, transform=torch.from_numpy)

    dl_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dl_valid = DataLoader(dataset_valid, batch_size=128, shuffle=False)

    trainer = pl.Trainer()
    trainer.fit(model, dl_train, dl_valid)
