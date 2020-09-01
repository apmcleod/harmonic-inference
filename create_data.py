from pathlib import Path
import logging

from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
import harmonic_inference.data.datasets as ds

logging.basicConfig(filename='create_data.log', level=logging.INFO, filemode='w')
files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs('corpus_data')

dataset_classes = [
    ds.ChordTransitionDataset,
    ds.ChordClassificationDataset,
    ds.ChordSequenceDataset,
    ds.KeyTransitionDataset,
    ds.KeySequenceDataset,
]
seed = 0

dataset_splits = ds.get_dataset_splits(
    files_df,
    measures_df,
    chords_df,
    notes_df,
    dataset_classes,
    splits=[0.8, 0.1, 0.1],
    seed=seed,
)

for i1, data_type in enumerate(dataset_classes):
    for i2, split in enumerate(['train', 'valid', 'test']):
        if dataset_splits[i1][i2] is not None:
            h5_path = Path('h5_data', f'{data_type.__name__}_{split}_seed_{seed}.h5')
            dataset_splits[i1][i2].to_h5(Path(h5_path))
