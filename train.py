import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import harmonic_inference.data.datasets as ds
import harmonic_inference.models.chord_classifier_models as ccm
from harmonic_inference.data.data_types import PitchType

h5_path = 'h5_data/ChordClassificationDataset_train_seed_0.h5'
h5_path_valid = 'h5_data/ChordClassificationDataset_valid_seed_0.h5'

ccds = ds.h5_to_dataset(h5_path, ds.ChordClassificationDataset, transform=torch.from_numpy)
ccds_valid = ds.h5_to_dataset(h5_path_valid, ds.ChordClassificationDataset, transform=torch.from_numpy)

dl = DataLoader(ccds, batch_size=128, shuffle=True)
dl_valid = DataLoader(ccds_valid, batch_size=128, shuffle=False)

model = ccm.SimpleChordClassifier(PitchType.TPC, PitchType.TPC, True)

trainer = pl.Trainer()
trainer.fit(model, dl, dl_valid)
