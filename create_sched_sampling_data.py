"""Create an h5 data file containing scheduled sampling from one model."""

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.datasets import DATASETS, h5_to_dataset
from harmonic_inference.models.joint_model import MODEL_CLASSES
from harmonic_inference.utils.data_utils import load_models_from_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a scheduled sampling dataset, from a given model and already "
            "created h5 data files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODEL_CLASSES.keys(),
        help="The type of model to use to generate the scheduled sampling inputs.",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help=(
            "The directory to load the model checkpoint from, within a subdirectory of the "
            "model's name (e.g., CCM checkpoints will be loaded from `--checkpoint`/ccm)."
        ),
    )

    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help=(
            "Specify a version number to load the model from. If given, "
            "the model will be loaded from `--checkpoint`/`--model`/version_`--model-version`"
        ),
    )

    DEFAULT_PATH = os.path.join(
        "`--checkpoint`", "model", "lightning_logs", "version_*", "checkpoints", "*.ckpt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_PATH,
        help="A checkpoint file to load the model from.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("h5_data"),
        help="The directory containing the non-scheduling sampled h5 data.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("h5_data"),
        help="The directory to save the created scheduling sampled h5 data into.",
    )

    parser.add_argument(
        "-w",
        "--workers",
        default=4,
        type=int,
        help="The number of workers per DataLoader.",
    )

    parser.add_argument(
        "--threads",
        default=None,
        type=int,
        help="The number of pytorch cpu threads to create.",
    )

    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="The random seed to used to create the data."
    )

    parser.add_argument(
        "-l", "--log", type=str, default=sys.stderr, help="The log file to print messages to."
    )

    ARGS = parser.parse_args()

    if ARGS.log is not sys.stderr:
        log_path = ARGS.output / ARGS.log
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.output / ARGS.log,
        level=logging.INFO,
        filemode="w",
    )

    if ARGS.threads is not None:
        torch.set_num_threads(ARGS.threads)

    model = load_models_from_argparse(ARGS, model_type=ARGS.model)[ARGS.model]
    model.transposition_range = (0, 0)
    dataset_class = DATASETS[ARGS.model]

    # Create scheduling sampled train, valid, and test (we may not use all of them)
    for split in ["train", "valid", "test"]:
        logging.info(f"Creating {split} data.")

        # Load data
        input_h5_path = Path(ARGS.input / f"{dataset_class.__name__}_{split}_seed_{ARGS.seed}.h5")
        with h5py.File(input_h5_path, "r") as h5_file:
            file_ids = list(h5_file["file_ids"])

        # Create dataset
        dataset = h5_to_dataset(
            input_h5_path,
            dataset_class,
            transform=torch.from_numpy,
            dataset_kwargs=model.get_dataset_kwargs(),
        )
        dl = DataLoader(
            dataset,
            batch_size=dataset_class.valid_batch_size,
            shuffle=False,
            num_workers=ARGS.workers,
        )

        # Generate outputs
        outputs = [
            model.get_output(batch) for batch in tqdm(dl, desc="Generating data from batches")
        ]

        # Save outputs
        output_h5_path = Path(ARGS.output / f"{ARGS.model}_{split}_sched_samp_seed_{ARGS.seed}.h5")
        if not output_h5_path.parent.exists():
            output_h5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_h5_path, "w") as h5_file:
            h5_file.create_dataset("outputs", data=np.vstack(outputs), compression="gzip")
            h5_file.create_dataset("file_ids", data=np.array(file_ids), compression="gzip")
            model.add_settings_to_h5_file(h5_file)
