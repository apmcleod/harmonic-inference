from pathlib import Path

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

MODEL = csm.SimpleChordSequenceModel(PitchType.TPC)
DATASET = ds.ChordSequenceDataset
H5_DIR = Path('h5_data')

h5_path = Path(H5_DIR / f'{DATASET.__name__}_train_seed_0.h5')
h5_path_valid = Path(H5_DIR / f'{DATASET.__name__}_valid_seed_0.h5')

dataset = ds.h5_to_dataset(h5_path, DATASET, transform=torch.from_numpy)
dataset_valid = ds.h5_to_dataset(h5_path_valid, DATASET, transform=torch.from_numpy)

dl = DataLoader(dataset, batch_size=128, shuffle=True)
dl_valid = DataLoader(dataset_valid, batch_size=128, shuffle=False)

trainer = pl.Trainer()
trainer.fit(MODEL, dl, dl_valid)
