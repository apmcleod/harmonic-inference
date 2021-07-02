"""Script to annotate an input score with the harmonic inference model.
This does not evaluate the annotation based on a ground truth."""
import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Union

import torch
from tqdm import tqdm

import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.data_types import PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import (
    MODEL_CLASSES,
    HarmonicInferenceModel,
    add_joint_model_args,
    from_args,
)
from harmonic_inference.utils.data_utils import load_models_from_argparse, load_pieces

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


def annotate(
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
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                eu.log_state(state, piece, model.CHORD_OUTPUT_TYPE, model.KEY_OUTPUT_TYPE)

            labels_df = eu.get_label_df(
                state,
                piece,
                model.CHORD_OUTPUT_TYPE,
                model.KEY_OUTPUT_TYPE,
            )

            # Write outputs tsv
            results_df = eu.get_results_df(
                piece,
                state,
                model.CHORD_OUTPUT_TYPE,
                model.KEY_OUTPUT_TYPE,
                PitchType.TPC,
                PitchType.TPC,
            )

            # Write MIDI outputs for SPS chord-eval testing
            results_midi_df = eu.get_results_df(
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
                    results_tsv_path = output_tsv_path.parent / (
                        output_tsv_path.name[:-4] + "_results_midi.tsv"
                    )
                    results_midi_df.to_csv(results_tsv_path, sep="\t")
                    logging.info("MIDI results TSV written out to %s", results_tsv_path)
                except Exception:
                    logging.exception("Error writing to csv %s", results_tsv_path)
                    logging.debug(results_midi_df)

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
        description="Generate functional harmony labels given some data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv of MusicXML files.",
    )

    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help="The --input data refers to MusicXML files.",
    )

    parser.add_argument(
        "--scores",
        action="store_true",
        help=(
            "Write the output label TSVs onto annotated scores in the output directory. "
            "Only works for DCML-format corpus_data tsv input and MuseScore3 scores."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="outputs",
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
            f"--{model}",
            type=str,
            default=DEFAULT_PATH,
            help=f"A checkpoint file to load the {model} from.",
        )

        parser.add_argument(
            f"--{model}-version",
            type=int,
            default=None,
            help=(
                f"Specify a version number to load the model from. If given, --{model} is ignored"
                f" and the {model} will be loaded from "
                + DEFAULT_PATH.replace("_*", f"_`--{model}-version`")
            ),
        )

    parser.add_argument(
        "--icm-json",
        type=str,
        default=os.path.join("`--checkpoint`", "icm", "initial_chord_prior.json"),
        help="The json file to load the icm from.",
    )

    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=sys.stderr,
        help=(
            "The log file to print messages to. If a file is given, it will be interpreted "
            "relative to `--output`."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logging information.",
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
        log_path = ARGS.output / ARGS.log
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=None if ARGS.log is sys.stderr else ARGS.output / ARGS.log,
        level=logging.DEBUG if ARGS.verbose else logging.INFO,
        filemode="w",
    )

    if ARGS.scores:
        if ARGS.annotations is None:
            raise ValueError("--annotations must be given with --scores option.")
        if ARGS.output is None:
            raise ValueError("--output must be given with --scores option.")
        write_tsvs_to_scores(ARGS.output, ARGS.annotations)
        sys.exit(0)

    # Load models
    models = load_models_from_argparse(ARGS)

    # Load pieces
    pieces = load_pieces(xml=ARGS.xml, input_path=ARGS.input)

    annotate(
        from_args(models, ARGS),
        pieces,
        output_tsv_dir=ARGS.output,
        annotations_base_dir=ARGS.annotations,
    )
