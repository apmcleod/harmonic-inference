"""Script to test (evaluate) a joint model for harmonic inference."""
import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Union

import torch
from tqdm import tqdm

import h5py
import harmonic_inference.models.initial_chord_models as icm
import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.data_types import NO_REDUCTION, TRIAD_REDUCTION, PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import (
    MODEL_CLASSES,
    HarmonicInferenceModel,
    add_joint_model_args,
    from_args,
)
from harmonic_inference.utils.data_utils import load_pieces

SPLITS = ["train", "valid", "test"]


def write_tsvs_to_scores(
    output_tsv_dir: Union[Path, str],
    annotations_base_dir: Union[Path, str],
):
    """
    Write the labels TSVs from the given output directory onto annotated scores
    (from the annotations_base_dir) in the output directory.

    Parameters
    ----------
    output_tsv_dir : Union[Path, str]
        The path to TSV files containing labels to write onto annotated scores.
        The directory should contain sub-directories for each composer (aligned
        with sub-dirs in the annotations base directory), and a single TSV for each
        output.
    annotations_base_dir : Union[Path, str]
        The path to annotations and MuseScore3 scores, whose sub-directories and file names
        are aligned with those in the output TSV directory.
    """
    output_tsv_dir = Path(output_tsv_dir)
    annotations_base_dir = Path(annotations_base_dir)

    output_paths = sorted(glob(str(output_tsv_dir / "**" / "*.tsv"), recursive=True))
    for piece_name in tqdm(output_paths, desc="Writing labesl to scores"):
        piece_name = Path(piece_name).relative_to(output_tsv_dir)
        try:
            eu.write_labels_to_score(
                output_tsv_dir / piece_name.parent,
                annotations_base_dir / piece_name.parent,
                piece_name.stem,
            )
        except Exception:
            logging.exception("Error writing score out to %s", output_tsv_dir / piece_name.parent)


def evaluate(
    model: HarmonicInferenceModel,
    pieces: List[Piece],
    output_tsv_dir: Union[Path, str] = None,
    annotations_base_dir: Union[Path, str] = None,
):
    """
    Get estimated chords and keys on the given pieces using the given model.

    Parameters
    ----------
    model : HarmonicInferenceModel
        The model to use to estimate chords and keys.
    pieces : List[Piece]
        The input pieces to estimate chords and keys from.
    output_tsv_dir : Union[Path, str]
        A directory to output TSV labels into. Each piece's output labels will go into
        a sub-directory according to its name field. If None, label TSVs are not generated.
    annotations_base_dir : Union[Path, str]
        A directory containing annotated scores, which the estimated labels can be written
        onto and then saved into the output_tsv directory.
    """
    if output_tsv_dir is not None:
        output_tsv_dir = Path(output_tsv_dir)

    if annotations_base_dir is not None:
        annotations_base_dir = Path(annotations_base_dir)

    for piece in tqdm(pieces, desc="Getting harmony for pieces"):
        if piece.name is not None:
            logging.info("Running piece %s", piece.name)

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

            # Write MIDI outputs for SPS chord-eval testing
            results_df = eu.get_results_df(
                piece,
                state,
                model.CHORD_OUTPUT_TYPE,
                model.KEY_OUTPUT_TYPE,
                PitchType.MIDI,
                PitchType.MIDI,
            )

            if piece.name is not None and output_tsv_dir is not None:
                piece_name = Path(piece.name.split(" ")[-1])
                output_tsv_path = output_tsv_dir / piece_name

                try:
                    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
                    results_tsv_path = output_tsv_path.parent / (
                        output_tsv_path.name[:-4] + "_results.tsv"
                    )
                    results_df.to_csv(results_tsv_path, sep="\t")
                    logging.info("Results TSV written out to %s", results_tsv_path)
                except Exception:
                    logging.exception("Error writing to csv %s", results_tsv_path)
                    logging.debug(results_df)

                try:
                    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
                    labels_df.to_csv(output_tsv_path, sep="\t")
                    logging.info("Labels TSV written out to %s", output_tsv_path)
                except Exception:
                    logging.exception("Error writing to csv %s", output_tsv_path)
                    logging.debug(labels_df)
                else:
                    if annotations_base_dir is not None:
                        try:
                            eu.write_labels_to_score(
                                output_tsv_dir / piece_name.parent,
                                annotations_base_dir / piece_name.parent,
                                piece_name.stem,
                            )
                            logging.info(
                                "Writing score out to %s",
                                output_tsv_dir / piece_name.parent,
                            )
                        except Exception:
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
        "--test",
        action="store_true",
        help="Run tests on the actual test set, rather than the validation set.",
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
        "--scores",
        action="store_true",
        help="Write the output label TSVs onto annotated scores in the output directory.",
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
        default=None,
        help=(
            "A directory containing corpora annotation tsvs and MuseScore3 scores, which "
            "will be used to write out labels onto new MuseScore3 score files in the "
            "--output directory."
        ),
    )

    parser.add_argument(
        "--average",
        type=Path,
        default=False,
        help="Average the accuracies from the given log file.",
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

    parser.add_argument(
        "--threads",
        default=None,
        type=int,
        help="The number of pytorch cpu threads to create.",
    )

    add_joint_model_args(parser)

    ARGS = parser.parse_args()

    if ARGS.threads is not None:
        torch.set_num_threads(ARGS.threads)

    if ARGS.log is not sys.stderr:
        log_path = Path(ARGS.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.log,
        level=logging.DEBUG if ARGS.verbose else logging.INFO,
        filemode="w",
    )

    if ARGS.average:
        for key, average in eu.average_results(ARGS.average).items():
            print(f"Average {key} = {average}")
        sys.exit(0)

    if ARGS.scores:
        if ARGS.annotations is None:
            raise ValueError("--annotations must be given with --scores option.")
        if ARGS.output is None:
            raise ValueError("--output must be given with --scores option.")
        write_tsvs_to_scores(ARGS.output, ARGS.annotations)
        sys.exit(0)

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
            logging.error("No checkpoints found for %s in %s.", model_name, checkpoint_arg)
            sys.exit(2)

        if len(possible_checkpoints) == 1:
            checkpoint = possible_checkpoints[0]
            logging.info("Loading checkpoint %s for %s.", checkpoint, model_name)

        else:
            checkpoint = possible_checkpoints[-1]
            logging.info("Multiple checkpoints found for %s. Loading %s.", model_name, checkpoint)

        models[model_name] = model_class.load_from_checkpoint(checkpoint)
        models[model_name].freeze()

    # Load icm json differently
    logging.info("Loading checkpoint %s for icm.", ARGS.icm_json)
    models["icm"] = icm.SimpleInitialChordModel(ARGS.icm_json)

    data_type = "test" if ARGS.test else "valid"

    # Load data for ctm
    h5_path = Path(ARGS.h5_dir / f"ChordTransitionDataset_{data_type}_seed_{ARGS.seed}.h5")
    with h5py.File(h5_path, "r") as h5_file:
        if "file_ids" not in h5_file:
            logging.error("file_ids not found in %s. Re-create with create_h5_data.py", h5_path)
            sys.exit(1)

        file_ids = list(h5_file["file_ids"])

    # Load pieces
    pieces = load_pieces(
        xml=ARGS.xml,
        input_path=ARGS.input,
        piece_dicts_path=Path(ARGS.h5_dir / f"pieces_{data_type}_seed_{ARGS.seed}.pkl"),
        file_ids=file_ids,
        specific_id=ARGS.id,
    )

    evaluate(
        from_args(models, ARGS),
        pieces,
        output_tsv_dir=ARGS.output,
        annotations_base_dir=ARGS.annotations,
    )
