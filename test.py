import argparse
import logging
import os
import pickle
import sys
from glob import glob
from pathlib import Path
from typing import List, Union

from tqdm import tqdm

import h5py
import harmonic_inference.models.initial_chord_models as icm
import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.data_types import NO_REDUCTION, TRIAD_REDUCTION
from harmonic_inference.data.piece import Piece, ScorePiece
from harmonic_inference.models.joint_model import (
    MODEL_CLASSES,
    HarmonicInferenceModel,
    add_joint_model_args,
    from_args,
)

SPLITS = ["train", "valid", "test"]


def evaluate(
    model: HarmonicInferenceModel,
    pieces: List[Piece],
    output_tsv_dir: Union[Path, str] = None,
    annotations_base_dir: Union[Path, str] = None,
):
    if output_tsv_dir is not None:
        output_tsv_dir = Path(output_tsv_dir)

    if annotations_base_dir is not None:
        annotations_base_dir = Path(annotations_base_dir)

    for piece in tqdm(pieces, desc="Getting harmony for pieces"):
        if piece.name is not None:
            logging.info(f"Running piece {piece.name}")

        state = model.get_harmony(piece)

        if state is None:
            logging.info("Returned None")
        else:
            chord_acc_full = eu.evaluate_chords(
                piece,
                state,
                model.CHORD_OUTPUT_TYPE,
                use_inversion=True,
                reduction=NO_REDUCTION,
            )
            chord_acc_no_inv = eu.evaluate_chords(
                piece,
                state,
                model.CHORD_OUTPUT_TYPE,
                use_inversion=False,
                reduction=NO_REDUCTION,
            )
            chord_acc_triad = eu.evaluate_chords(
                piece,
                state,
                model.CHORD_OUTPUT_TYPE,
                use_inversion=True,
                reduction=TRIAD_REDUCTION,
            )
            chord_acc_triad_no_inv = eu.evaluate_chords(
                piece,
                state,
                model.CHORD_OUTPUT_TYPE,
                use_inversion=False,
                reduction=TRIAD_REDUCTION,
            )

            logging.info("Chord accuracy = %s", chord_acc_full)
            logging.info("Chord accuracy, no inversions = %s", chord_acc_no_inv)
            logging.info("Chord accuracy, triads = %s", chord_acc_triad)
            logging.info("Chord accuracy, triad, no inversions = %s", chord_acc_triad_no_inv)

            key_acc_full = eu.evaluate_keys(piece, state, model.KEY_OUTPUT_TYPE, tonic_only=False)
            key_acc_tonic = eu.evaluate_keys(piece, state, model.KEY_OUTPUT_TYPE, tonic_only=True)

            logging.info("Key accuracy = %s", key_acc_full)
            logging.info("Key accuracy, tonic only = %s", key_acc_tonic)

            full_acc = eu.evaluate_chords_and_keys_jointly(
                piece,
                state,
                root_type=model.CHORD_OUTPUT_TYPE,
                tonic_type=model.KEY_OUTPUT_TYPE,
                use_inversion=True,
                chord_reduction=NO_REDUCTION,
                tonic_only=False,
            )
            logging.info("Full accuracy = %s", full_acc)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                eu.log_state(state, piece, model.CHORD_OUTPUT_TYPE, model.KEY_OUTPUT_TYPE)

            labels_df = eu.get_label_df(
                state,
                piece,
                model.CHORD_OUTPUT_TYPE,
                model.KEY_OUTPUT_TYPE,
            )

            if piece.name is not None and output_tsv_dir is not None:
                piece_name = Path(piece.name.split(" ")[-1])
                output_tsv_path = output_tsv_dir / piece_name

                try:
                    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
                    labels_df.to_csv(output_tsv_path, sep="\t")
                    logging.info("Labels TSV written out to %s", output_tsv_path)
                except Exception:
                    logging.exception("Error writing to csv %s", output_tsv_path)
                    logging.debug(labels_df)
                else:
                    if annotations_base_dir is not None:
                        log_level = logging.getLogger().getEffectiveLevel()
                        log_file = logging.getLogger().handlers[-1].baseFilename

                        try:
                            eu.write_labels_to_score(
                                output_tsv_dir / piece_name.parent,
                                annotations_base_dir / piece_name.parent,
                                piece_name.stem,
                            )
                            logging.basicConfig(filename=log_file, level=log_level)
                            logging.info(
                                "Writing score out to %s",
                                output_tsv_dir / piece_name.parent,
                            )
                        except Exception:
                            logging.basicConfig(filename=log_file, level=log_level)
                            logging.exception(
                                "Error writing score out to %s",
                                output_tsv_dir / piece_name.parent,
                            )

            else:
                logging.debug(labels_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a harmonic inference model on some data.",
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
        default="output",
        help="The directory to write label tsvs and annotated MuseScore3 scores to.",
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("../corpora/annotations"),
        help=(
            "A directory containing corpora annotation tsvs and MuseScore3 scores, which "
            "will be used to write out labels onto new MuseScore3 score files in the "
            "--output directory."
        ),
    )

    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Only evaluate the given file_id (which must be from the validation set).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="The directory containing checkpoints for each type of model.",
    )

    for model in MODEL_CLASSES.keys():
        if model == "icm":
            continue

        DEFAULT_PATH = os.path.join(
            "`--checkpoint`", model, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        )
        parser.add_argument(
            f"--{model}-checkpoint",
            type=str,
            default=DEFAULT_PATH,
            help=f"The checkpoint file to load the {model} from.",
        )

    parser.add_argument(
        "--icm-json",
        type=str,
        default=os.path.join("checkpoints", "icm", "initial_chord_prior.json"),
        help="The json file to load the icm from.",
    )

    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=sys.stderr,
        help="The log file to print messages to.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logging information.",
    )

    parser.add_argument(
        "-h5",
        "--h5_dir",
        default=Path("h5_data"),
        type=Path,
        help=(
            "The directory that holds the h5 data containing file_ids to test on, and the piece "
            "pkl files."
        ),
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="The seed used when generating the h5_data.",
    )

    add_joint_model_args(parser)

    ARGS = parser.parse_args()

    if ARGS.log is not sys.stderr:
        log_path = Path(ARGS.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.log,
        level=logging.DEBUG if ARGS.verbose else logging.INFO,
        filemode="w",
    )

    # Load models
    models = {}
    for model_name, model_class in MODEL_CLASSES.items():
        if model_name == "icm":
            continue

        DEFAULT_PATH = os.path.join(
            "`--checkpoint`", model_name, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        )
        checkpoint_arg = getattr(ARGS, f"{model_name}_checkpoint")

        if checkpoint_arg == DEFAULT_PATH:
            checkpoint_arg = checkpoint_arg.replace("`--checkpoint`", ARGS.checkpoint)

        possible_checkpoints = sorted(glob(checkpoint_arg))
        if len(possible_checkpoints) == 0:
            logging.error(f"No checkpoints found for {model_name} in {checkpoint_arg}")
            sys.exit(2)

        if len(possible_checkpoints) == 1:
            checkpoint = possible_checkpoints[0]
            logging.info(f"Loading checkpoint {checkpoint} for {model_name}.")

        else:
            checkpoint = possible_checkpoints[-1]
            logging.info(f"Multiple checkpoints found for {model_name}. Loading {checkpoint}.")

        models[model_name] = model_class.load_from_checkpoint(checkpoint)
        models[model_name].freeze()

    # Load icm json differently
    logging.info(f"Loading checkpoint {ARGS.icm_json} for icm.")
    models["icm"] = icm.SimpleInitialChordModel(ARGS.icm_json)

    # Load validation data for ctm
    h5_path = Path(ARGS.h5_dir / f"ChordTransitionDataset_valid_seed_{ARGS.seed}.h5")
    with h5py.File(h5_path, "r") as h5_file:
        if "file_ids" not in h5_file:
            logging.error(f"file_ids not found in {h5_path}. Re-create with create_h5_data.py")
            sys.exit(1)

        file_ids = list(h5_file["file_ids"])

    # Load pieces
    files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

    # Load from pkl if available
    pkl_path = Path(ARGS.h5_dir / f"pieces_valid_seed_{ARGS.seed}.pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as pkl_file:
            piece_dicts = pickle.load(pkl_file)
    else:
        piece_dicts = [None] * len(file_ids)

    # Run only on a specific file
    if ARGS.id is not None:
        if ARGS.id not in file_ids:
            raise ValueError("Given id not in the validation set.")

        index = file_ids.index(ARGS.id)
        file_ids = [file_ids[index]]
        piece_dicts = [piece_dicts[index]]

    pieces = [
        ScorePiece(
            notes_df.loc[file_id],
            chords_df.loc[file_id],
            measures_df.loc[file_id],
            piece_dict=piece_dict,
            name=(
                f"{file_id}: {files_df.loc[file_id, 'corpus_name']}/"
                f"{files_df.loc[file_id, 'file_name']}"
            ),
        )
        for file_id, piece_dict in tqdm(
            zip(file_ids, piece_dicts),
            total=len(file_ids),
            desc="Loading pieces",
        )
    ]

    evaluate(
        from_args(models, ARGS),
        pieces,
        output_tsv_dir=ARGS.output,
        annotations_base_dir=ARGS.annotations,
    )
