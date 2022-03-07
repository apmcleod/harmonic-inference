"""Create blocks of h5 data to use for training, validation, and testing."""
import argparse
import logging
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np

import harmonic_inference.data.datasets as ds
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs

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
        help="The directory containing the raw corpus files.",
    )

    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help=(
            "The --input data comes from the funtional-harmony repository, as MusicXML "
            "files and labels CSV files."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("h5_data"),
        help="The directory to save the created h5 data into.",
    )

    parser.add_argument(
        "-c",
        "--corpus",
        type=str,
        nargs="*",
        help="Only include pieces whose corpus name contains any of these strings",
    )

    parser.add_argument(
        "-l", "--log", type=str, default=sys.stderr, help="The log file to print messages to."
    )

    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="The random seed to use to create the data."
    )

    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help=f"The proportions for splits {SPLITS}. These values will be normalized.",
    )

    parser.add_argument(
        "--changes",
        action="store_true",
        help="Do not merge otherwise identical chords whose changes (chord pitches) differ.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="True to parse only the first 5 pieces, as a test of data creation.",
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
        os.makedirs(Path(ARGS.log).parent, exist_ok=True)
        logging.basicConfig(filename=ARGS.log, level=logging.INFO, filemode="w")

    xmls_and_csvs = None
    dfs = None
    if ARGS.xml:
        xmls = []
        csvs = []

        for file_path in sorted(glob(os.path.join(ARGS.input, "**", "*.mxl"), recursive=True)):
            music_xml_path = Path(file_path)
            label_csv_path = (
                music_xml_path.parent.parent / "chords" / Path(str(music_xml_path.stem) + ".csv")
            )

            if music_xml_path.exists() and label_csv_path.exists():
                xmls.append(music_xml_path)
                csvs.append(label_csv_path)

        xmls_and_csvs = {"xmls": xmls, "csvs": csvs}

    else:
        files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

        dfs = {
            "files": files_df,
            "measures": measures_df,
            "chords": chords_df,
            "notes": notes_df,
        }

        if ARGS.debug:
            dfs["files"] = dfs["files"].iloc[:5]

    if ARGS.seed is None:
        ARGS.seed = np.random.randint(0, 2 ** 32)
        logging.info("Using seed %s", ARGS.seed)

    if ARGS.corpus:
        if ARGS.xml:
            logging.info("--corpus not implemented for MusicXML files yet. Ignoring.")
        else:
            logging.info("Only using pieces whose corpus contains the string(s): %s", ARGS.corpus)
            dfs["files"] = dfs["files"][
                dfs["files"]["corpus_name"].str.contains("|".join(ARGS.corpus))
            ]

    dataset_splits, split_ids, split_pieces = ds.get_dataset_splits(
        ds.DATASETS.values(),
        data_dfs=dfs,
        xml_and_csv_paths=xmls_and_csvs,
        splits=ARGS.splits,
        seed=ARGS.seed,
        changes=ARGS.changes,
    )

    os.makedirs(Path(ARGS.output), exist_ok=True)

    for split, pieces in zip(SPLITS, split_pieces):
        if len(pieces) > 0:
            pickle_path = ARGS.output / f"pieces_{split}_seed_{ARGS.seed}.pkl"
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump([piece.to_dict() for piece in pieces], pickle_file)

    for i1, data_type in enumerate(ds.DATASETS.values()):
        for i2, split in enumerate(SPLITS):
            if dataset_splits[i1][i2] is not None:
                h5_path = ARGS.output / f"{data_type.__name__}_{split}_seed_{ARGS.seed}.h5"
                dataset_splits[i1][i2].to_h5(h5_path, file_ids=split_ids[i2])
                dataset_splits[i1][i2] = None
