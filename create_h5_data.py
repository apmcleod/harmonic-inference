import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

import harmonic_inference.data.datasets as ds
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs

DATASET_CLASSES = [
    ds.ChordTransitionDataset,
    ds.ChordClassificationDataset,
    ds.ChordSequenceDataset,
    ds.KeyTransitionDataset,
    ds.KeySequenceDataset,
]

SPLITS = ["train", "valid", "test"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the h5 dataset files for each type of dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("h5_data"),
        help="The directory to save the created h5 data into.",
    )

    parser.add_argument(
        "-l", "--log", type=str, default=sys.stderr, help="The log file to print messages to."
    )

    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="The random seed to use to create the data."
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
    if any([split_prop < 0 for split_prop in ARGS.splits]):
        print("--split values cannot be negative.", file=sys.stderr)
        sys.exit(1)
    sum_splits = sum(ARGS.splits)
    if sum_splits == 0:
        print("At least one --split value must be positive.", file=sys.stderr)
        sys.exit(1)
    ARGS.splits = [split / sum_splits for split in ARGS.splits]

    if ARGS.log is not sys.stderr:
        logging.basicConfig(filename=ARGS.log, level=logging.INFO, filemode="w")
    files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

    if ARGS.seed is None:
        ARGS.seed = np.random.randint(0, 2 ** 32)
        logging.info(f"Using seed {ARGS.seed}")

    dataset_splits, split_ids, split_pieces = ds.get_dataset_splits(
        files_df[:10],
        measures_df,
        chords_df,
        notes_df,
        DATASET_CLASSES,
        splits=ARGS.splits,
        seed=ARGS.seed,
    )

    for split, pieces in zip(SPLITS, split_pieces):
        if len(pieces) > 0:
            pickle_path = ARGS.output / f"pieces_{split}_seed_{ARGS.seed}.pkl"
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump([piece.to_dict() for piece in pieces], pickle_file)

    for i1, data_type in enumerate(DATASET_CLASSES):
        for i2, split in enumerate(SPLITS):
            if dataset_splits[i1][i2] is not None:
                h5_path = ARGS.output / f"{data_type.__name__}_{split}_seed_{ARGS.seed}.h5"
                dataset_splits[i1][i2].to_h5(h5_path, file_ids=split_ids[i2])
                dataset_splits[i1][i2] = None
