import pandas as pd
import numpy as np
import argparse
import os
import sys

import harmonic_inference_data as hid
import harmonic_inference_models as him
import harmonic_utils as hu
from corpus_reading import read_dump
import model_trainer
import eval_utils as eu

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

def get_masks_and_names(include_none=True):
    """
    Get input masks and input mask names for ablation studies.
    
    Parameters
    ----------
    include_none : boolean
        True to include a 'no_ablation' mask of all 1's in the list. False otherwise.
    
    Returns
    -------
    masks : list
        A list of length-30 binary masks for the note input vectors.
        
    mask_names : list
        A list of the name of each binary mask in masks.
    """
    mask_names = ['no_rhythm', 'no_levels', 'no_chord-relative_rhythm', 'no_lowest', 'no_octave']
    if include_none:
        mask_names.append('no_ablation')
    
    masks = [np.ones(30) for _ in range(len(mask_names))]
    masks[0][-6:-1] = 0
    masks[1][-6:-4] = 0
    masks[2][-4:-1] = 0
    masks[3][-1] = 0
    masks[4][1 + 12:1 + 12 + 127 // 12 + 1] = 0
    
    return masks, mask_names



def load_all_ablated_dfs(directory=None, prefix=None, include_none=True):
    """
    Load all ablated dataframes into a list of dfs.
    
    Parameters
    ----------
    directory : string
        If given, the directory all of the csv files are found in.
        
    prefix : string
        If given, a prefix to the file name of each csv (followed by '_').
        
    include_none : boolean
        True to include the 'no_ablation' mask results. False otherwise.
    
    Returns
    -------
    dfs : list(pd.DataFrame)
        The DataFrames, loaded in the order of masks returned by get_masks_and_names.
        If a csv does not exist, that value is None.
    """
    _, mask_names = get_masks_and_names(include_none=include_none)
    
    dfs = []
    for mask_name in mask_names:
        if prefix is not None:
            mask_name = prefix + '_' + mask_name
        try:
            if directory is None:
                dfs.append(eu.load_eval_df(mask_name + '.csv'))
            else:
                dfs.append(eu.load_eval_df(os.path.join(directory, mask_name + '.csv')))
        except:
            print(f"Error loading eval df from {mask_name}.csv", file=sys.stderr)
            dfs.append(None)
            
    return dfs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate ablation results.')
    
    parser.add_argument('--dir', help='The directory to save checkpoints and results into',
                        default='.')
    parser.add_argument('--eval', action='store_true', help='Create tsvs of the results.')
    parser.add_argument('--split', choices=['test', 'valid'], default='test',
                        help='Which split to calculate results for (with --eval)')
    parser.add_argument('--rel', choices=['local', 'global'], help='Use data with pitches '
                        'and chord roots relative to a local or global key.')
    
    args = parser.parse_args()
    
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
    
    transpose_global = args.rel is not None and args.rel == 'global'
    transpose_local = args.rel is not None and args.rel == 'local'
    
    h5_prefix = '811split'
    if args.rel is not None:
        h5_prefix = args.rel + 'key_811split'

    train_dataset, valid_dataset, test_dataset = hid.get_train_valid_test_splits(
        chords_df=chords_df, notes_df=notes_df, measures_df=measures_df, files_df=files_df,
        seed=0, h5_directory='data', h5_prefix=h5_prefix, make_dfs=True,
        transpose_global=transpose_global, transpose_local=transpose_local
    )

    masks, mask_names = get_masks_and_names(include_none=True)

    for mask, mask_name in zip(masks, mask_names):
        print(mask_name)
        if args.rel is not None:
            mask_name = args.rel + '_' + mask_name
        if mask is not None:
            mask = torch.tensor(mask)
        model = him.MusicScoreModel(len(train_dataset[0]['notes'][0]), len(hu.CHORD_TYPES) * 12, dropout=0.2, input_mask=mask)

        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
        criterion = CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        schedule_var = 'valid_loss'
        resume = None
        if args.eval:
            resume = os.path.join(args.dir, mask_name, 'best.pth.tar')

        trainer = model_trainer.ModelTrainer(model, train_dataset=train_dataset, valid_dataset=valid_dataset,
                                             test_dataset=test_dataset, seed=0, num_epochs=100, early_stopping=20,
                                             optimizer=optimizer, scheduler=scheduler, schedule_var=schedule_var,
                                             criterion=criterion,
                                             log_every=1, 
                                             save_every=10, save_dir=os.path.join(args.dir, mask_name), save_prefix='checkpoint',
                                             resume=resume, log_file_name=os.path.join(args.dir, mask_name + '.log'))

        if args.eval:
            loss, acc, outputs, labels = trainer.evaluate(valid=args.split == 'valid')
            eval_df = eu.get_eval_df(labels, outputs, test_dataset if args.split == 'test' else valid_dataset)
            eu.write_eval_df(eval_df, os.path.join(args.dir, mask_name + '.csv'))
        else:
            trainer.train()
