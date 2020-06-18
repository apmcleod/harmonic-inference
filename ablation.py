import pandas as pd
import numpy as np
import harmonic_inference_data as hid

from corpus_reading import read_dump
import corpus_utils

chords_tsv = 'data/chord_list.tsv'
notes_tsv = 'data/note_list.tsv'
measures_tsv = 'data/measure_list.tsv'
files_tsv = 'data/file_list.tsv'

chords_df = read_dump(chords_tsv)
notes_df = read_dump(notes_tsv, index_col=[0,1,2])
measures_df = read_dump(measures_tsv)
files_df = read_dump(files_tsv, index_col=0)

# Bugfixes
measures_df.loc[(685, 487), 'next'][0] = 488

train_dataset, valid_dataset, test_dataset = hid.get_train_valid_test_splits(
    chords_df=chords_df, notes_df=notes_df, measures_df=measures_df, files_df=files_df,
    seed=0, h5_directory='data', h5_prefix='811split', make_dfs=True
)

mask_names = ['no_rhythm', 'no_levels', 'no_chord-relative_rhythm', 'no_lowest', 'no_octave']
masks = [np.ones(30) for _ in range(len(mask_names))]
masks[0][-6:-1] = 0
masks[1][-6:-4] = 0
masks[2][-4:-1] = 0
masks[3][-1] = 0
masks[4][1+12:1+12+127//12+1] = 0

import torch
import harmonic_inference_models as him
import harmonic_utils
import model_trainer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

for mask, mask_name in zip(masks, mask_names):
    print(mask_name)
    mask = torch.tensor(mask)
    model = him.MusicScoreModel(len(train_dataset[0]['notes'][0]), len(harmonic_utils.CHORD_TYPES) * 12, dropout=0.2, input_mask=mask)
    
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    criterion = CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    schedule_var = 'valid_loss'

    trainer = model_trainer.ModelTrainer(model, train_dataset=train_dataset, valid_dataset=valid_dataset,
                                         test_dataset=test_dataset, seed=0, num_epochs=100, early_stopping=20,
                                         optimizer=optimizer, scheduler=scheduler, schedule_var=schedule_var,
                                         criterion=criterion,
                                         log_every=1, 
                                         save_every=10, save_dir=mask_name, save_prefix='checkpoint',
                                         resume=None, log_file=open(mask_name + '.log', 'w'))
    
    trainer.train()
