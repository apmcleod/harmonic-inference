"""Script to train any of the different models for harmonic inference."""
import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler
from torch.utils.data import DataLoader

import harmonic_inference.data.datasets as ds
import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.initial_chord_models as icm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
from harmonic_inference.data.data_types import PieceType, PitchType
from harmonic_inference.models.joint_model import MODEL_CLASSES
from harmonic_inference.utils.data_utils import load_kwargs_from_json, load_pieces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train any of the submodules for harmonic inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODEL_CLASSES.keys(),
        help="The type of model to train.",
        required=True,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "The device number for the GPU to train on. "
            "If not given, the model will be trained on CPU."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help=(
            "The directory to save model checkpoints into, within a subdirectory of the model's "
            "name (e.g., CSM checkpoints will be saved into `--checkpoint`/csm)."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="A path to a checkpoint file from which to resume training.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv files. Used only for -m icm",
    )

    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help=(
            "The --input data comes from the funtional-harmony repository, as MusicXML "
            "files and labels CSV files. Only important for --model icm. Other models load "
            "h5 data directly."
        ),
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
        "--threads",
        default=None,
        type=int,
        help="The number of pytorch cpu threads to create.",
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

    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="The seed used when generating the h5_data.",
    )

    parser.add_argument(
        "--model-kwargs",
        default=None,
        type=Path,
        help="A json file containing kwargs to be passed to the model initializer function.",
    )

    ARGS = parser.parse_args()

    ARGS.checkpoint = os.path.join(ARGS.checkpoint, ARGS.model)

    os.makedirs(ARGS.checkpoint, exist_ok=True)

    kwargs = load_kwargs_from_json(ARGS.model_kwargs)

    if ARGS.model == "ccm":
        model = ccm.SimpleChordClassifier(
            PieceType.SCORE,
            PitchType.TPC,
            PitchType.TPC,
            learning_rate=ARGS.lr,
            **kwargs,
        )

        if "input_mask" in kwargs:
            kwargs["input_mask"] = ds.transform_input_mask_to_binary(
                kwargs["input_mask"],
                model.input_dim,
            )
            model = ccm.SimpleChordClassifier(
                PieceType.SCORE,
                PitchType.TPC,
                PitchType.TPC,
                learning_rate=ARGS.lr,
                **kwargs,
            )

        dataset = ds.ChordClassificationDataset

    elif ARGS.model == "ctm":
        model = ctm.SimpleChordTransitionModel(
            PieceType.SCORE,
            PitchType.TPC,
            learning_rate=ARGS.lr,
            **kwargs,
        )

        if "input_mask" in kwargs:
            kwargs["input_mask"] = ds.transform_input_mask_to_binary(
                kwargs["input_mask"],
                model.input_dim,
            )
            model = ctm.SimpleChordTransitionModel(
                PieceType.SCORE,
                PitchType.TPC,
                learning_rate=ARGS.lr,
                **kwargs,
            )

        dataset = ds.ChordTransitionDataset

    elif ARGS.model == "csm":
        model = csm.SimpleChordSequenceModel(
            PitchType.TPC,
            PitchType.TPC,
            PitchType.TPC,
            learning_rate=ARGS.lr,
            **kwargs,
        )

        if "input_mask" in kwargs:
            kwargs["input_mask"] = ds.transform_input_mask_to_binary(
                kwargs["input_mask"],
                model.input_dim,
            )
            model = csm.SimpleChordSequenceModel(
                PitchType.TPC,
                PitchType.TPC,
                PitchType.TPC,
                learning_rate=ARGS.lr,
                **kwargs,
            )

        dataset = ds.ChordSequenceDataset

    elif ARGS.model == "ktm":
        model = ktm.SimpleKeyTransitionModel(
            PitchType.TPC,
            PitchType.TPC,
            learning_rate=ARGS.lr,
            **kwargs,
        )

        if "input_mask" in kwargs:
            kwargs["input_mask"] = ds.transform_input_mask_to_binary(
                kwargs["input_mask"],
                model.input_dim,
            )
            model = ktm.SimpleKeyTransitionModel(
                PitchType.TPC,
                PitchType.TPC,
                learning_rate=ARGS.lr,
                **kwargs,
            )

        dataset = ds.KeyTransitionDataset

    elif ARGS.model == "ksm":
        model = ksm.SimpleKeySequenceModel(
            PitchType.TPC,
            PitchType.TPC,
            PitchType.TPC,
            learning_rate=ARGS.lr,
            **kwargs,
        )

        if "input_mask" in kwargs:
            kwargs["input_mask"] = ds.transform_input_mask_to_binary(
                kwargs["input_mask"],
                model.input_dim,
            )
            model = ksm.SimpleKeySequenceModel(
                PitchType.TPC,
                PitchType.TPC,
                PitchType.TPC,
                learning_rate=ARGS.lr,
                **kwargs,
            )

        dataset = ds.KeySequenceDataset

    elif ARGS.model == "icm":
        # Load training data for ctm, just to get file_ids
        h5_path = Path(ARGS.h5_dir / f"ChordTransitionDataset_train_seed_{ARGS.seed}.h5")
        with h5py.File(h5_path, "r") as h5_file:
            if "file_ids" not in h5_file:
                logging.error("file_ids not found in %s. Re-create with create_h5_data.py", h5_path)
                sys.exit(1)

            file_ids = list(h5_file["file_ids"])

        pieces = load_pieces(
            xml=ARGS.xml,
            input_path=ARGS.input,
            piece_dicts_path=Path(ARGS.h5_dir / f"pieces_train_seed_{ARGS.seed}.pkl"),
            file_ids=file_ids,
        )

        chords = [piece.get_chords()[0] for piece in pieces]
        icm.train_icm(
            chords,
            os.path.join(ARGS.checkpoint, "initial_chord_prior.json"),
            add_n_smoothing=1.0,
        )
        sys.exit(0)

    else:
        logging.error("Invalid model: %s", ARGS.model)
        sys.exit(1)

    if ARGS.threads is not None:
        torch.set_num_threads(ARGS.threads)

    h5_path_train = Path(ARGS.h5_dir / f"{dataset.__name__}_train_seed_{ARGS.seed}.h5")
    h5_path_valid = Path(ARGS.h5_dir / f"{dataset.__name__}_valid_seed_{ARGS.seed}.h5")

    dataset_train = ds.h5_to_dataset(
        h5_path_train,
        dataset,
        transform=torch.from_numpy,
        dataset_kwargs=model.get_dataset_kwargs(),
    )
    dataset_valid = ds.h5_to_dataset(
        h5_path_valid,
        dataset,
        transform=torch.from_numpy,
        dataset_kwargs=model.get_dataset_kwargs(),
    )

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
            output_filename=os.path.join(ARGS.checkpoint, "profile.log")
        )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
    )

    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir=ARGS.checkpoint,
        profiler=ARGS.profile,
        callbacks=[early_stopping_callback, lr_logger],
        gpus=[ARGS.gpu] if ARGS.gpu is not None else None,
        resume_from_checkpoint=ARGS.resume,
    )
    trainer.fit(model, dl_train, dl_valid)
