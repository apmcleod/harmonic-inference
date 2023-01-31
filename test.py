"""Script to test (evaluate) a joint model for harmonic inference."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Union

import h5py
import ms3
import numpy as np
import torch
from tqdm import tqdm

import harmonic_inference.utils.eval_utils as eu
from annotate import set_default_args
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
    suffix: str = "_inferred",
    ask_for_input: bool = True,
    replace: bool = True,
    staff: Optional[Literal[1, 2, 3, 4]] = None,
    voice: Literal[1, 2, 3, 4] = None,
    harmony_layer: Optional[Literal[0, 1, 2]] = None,
    check_for_clashes: bool = True,
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

    suffix : str
        Suffix to be added to the newly created scores. Defaults to _inferred.

    ask_for_input : bool
        What to do if more than one TSV or MuseScore file is detected for a particular fname.
        By default, the user is asked for input.
        Pass False to prevent that and pick the files with the shortest relative paths instead.

    replace : bool
        By default, any existing labels are removed from the scores. Pass False to leave them in,
        which may lead to clashes.

    staff : Optional[int]
        If you pass a staff ID, the labels will be attached to that staff where 1 is the upper
        staff.
        By default, the staves indicated in the 'staff' column of the labels are used, or, if
        such a column is not present, labels will be inserted under the lowest staff -1.

    voice : Optional[Literal[1, 2, 3, 4]]
        If you pass the ID of a notational layer (where 1 is the upper voice, blue in MuseScore),
        the labels will be attached to that one.
        By default, the notational layers indicated in the 'voice' column of the labels are used,
        or, if such a column is not present, labels will be inserted for voice 1.

    harmony_layer : Optional[Literal[0, 1, 2]]
        | By default, the labels are written to the layer specified as an integer in the
          'harmony_layer' of the label. Pass an integer to select a particular layer:
        | * 0 to attach them as absolute ('guitar') chords, meaning that when opened next time,
        |   MuseScore will split and encode those beginning with a note name
        |   (resulting in ms3-internal harmony_layer 3).
        | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
        | * 2 to have MuseScore interpret them as Nashville Numbers

    check_for_clashes : bool
        By default, warnings are thrown when there already exists a label at a position
        (and in a notational layer) where a new one is attached. Pass False to deactivate these
        warnings.


    """
    output_tsv_dir = Path(output_tsv_dir)
    annotations_base_dir = Path(annotations_base_dir)
    outputs_are_under_corpus_path = (output_tsv_dir == annotations_base_dir) or (
        annotations_base_dir in annotations_base_dir.parents
    )
    corpus = ms3.Corpus(annotations_base_dir, only_metadata_fnames=False)
    if not outputs_are_under_corpus_path:
        # if the Corpus directory itself includes any labels we exclude them to avoid ambiguity
        if any(x.is_dir() and x.name == "labels" for x in annotations_base_dir.iterdir()):
            excluded_labels_path = annotations_base_dir / "labels"
            corpus.view.exclude("path", str(excluded_labels_path))
    _ = corpus.add_dir(
        output_tsv_dir,
        filter_other_fnames=True,
        file_re=r"\.tsv$",
        exclude_re="",
    )
    ms3.insert_labels_into_score(
        corpus,
        facet="labels",
        ask_for_input=ask_for_input,
        replace=replace,
        staff=staff,
        voice=voice,
        harmony_layer=harmony_layer,
        check_for_clashes=check_for_clashes,
        print_info=False,
    )
    ms3.store_scores(
        corpus,
        only_changed=True,
        root_dir=output_tsv_dir,
        folder=".",
        suffix=suffix,
        overwrite=True,
    )


def evaluate(
    model: HarmonicInferenceModel,
    pieces: List[Piece],
    output_tsv_dir: Union[Path, str] = None,
    force_segs: bool = False,
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
    force_segs : bool
        Force the model to use the ground truth segmentations.
    """
    if output_tsv_dir is not None:
        output_tsv_dir = Path(output_tsv_dir)

    for piece in tqdm(pieces, desc="Getting harmony for pieces"):
        piece: Piece
        if piece.name is not None:
            logging.info("Running piece %s", piece.name)

        forces_dict = {}
        if force_segs:
            forces_dict["forced_chord_changes"] = set(
                [i for i in range(len(piece.get_inputs())) if i in piece.get_chord_change_indices()]
            )
            forces_dict["forced_chord_non_changes"] = set(
                [
                    i
                    for i in range(len(piece.get_inputs()))
                    if i not in piece.get_chord_change_indices()
                ]
            )

        state, estimated_piece = model.get_harmony(piece, **forces_dict)

        if state is None:
            logging.info("Returned None")
            continue

        # Create results dfs
        results_annotation_df = eu.get_results_annotation_df(
            estimated_piece,
            piece,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            model.chord_classifier.use_inversions,
            model.chord_classifier.reduction,
            use_chord_pitches=True,
        )

        results_df = eu.get_results_df(
            piece,
            estimated_piece,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            PitchType.TPC,
            PitchType.TPC,
            True,
            None,
        )

        results_midi_df = eu.get_results_df(
            piece,
            estimated_piece,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            PitchType.MIDI,
            PitchType.MIDI,
            True,
            None,
        )

        # Perform evaluations
        if piece.get_chords() is None:
            logging.info("Cannot compute accuracy. Ground truth unknown.")
        else:
            eu.log_results_df_eval(results_df)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            eu.log_state(state, piece, model.CHORD_OUTPUT_TYPE, model.KEY_OUTPUT_TYPE)

        if piece.name is not None and output_tsv_dir is not None:
            piece_name = Path(piece.name.split(" ")[-1])
            output_tsv_path = output_tsv_dir / piece_name

            for suffix, name, df in (
                ["_results.tsv", "Results", results_df],
                ["_results_midi.tsv", "MIDI results", results_midi_df],
                [".tsv", "Results annotation", results_annotation_df],
            ):
                try:
                    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
                    tsv_path = output_tsv_path.parent / (output_tsv_path.name[:-4] + suffix)
                    df.to_csv(tsv_path, sep="\t")
                    logging.info("%s TSV written out to %s", name, tsv_path)
                except Exception:
                    logging.exception("Error writing to csv %s", tsv_path)
                    logging.debug(results_df)

        else:
            logging.debug(results_annotation_df)


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
        "--defaults",
        action="store_true",
        help=(
            "Use the default hyperparameter settings from the internal-corpus-trained system, "
            "according to the chosen `--csm-version` (0, 1, or 2)."
        ),
    )

    parser.add_argument(
        "--fh-defaults",
        action="store_true",
        help=(
            "Use the default hyperparameter settings from the FH-corpus-trained system, "
            "according to the chosen `--csm-version` (0, 1, or 2)."
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
        "--no_h5",
        action="store_true",
        help="Do not read file_ids and piece_dicts from the `--h5_dir` directory.",
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

    parser.add_argument(
        "--force-segs",
        action="store_true",
        help="Force the model to use ground truth segmentations.",
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

    if ARGS.average:
        for key, average in eu.average_results(ARGS.average).items():
            print(f"Average {key} = {average}")
        sys.exit(0)

    set_default_args(ARGS)

    # Load models
    models = load_models_from_argparse(ARGS)

    data_type = "test" if ARGS.test else "valid"

    # Load data for ctm to get file_ids
    h5_path = Path(ARGS.h5_dir / f"ChordTransitionDataset_{data_type}_seed_{ARGS.seed}.h5")
    if not ARGS.no_h5 and h5_path.exists():
        with h5py.File(h5_path, "r") as h5_file:
            if "file_ids" not in h5_file:
                logging.error("file_ids not found in %s. Re-create with create_h5_data.py", h5_path)
                sys.exit(1)

            file_ids = list(h5_file["file_ids"])
    else:
        file_ids = None

    # Load pieces
    pieces = load_pieces(
        xml=ARGS.xml,
        input_path=ARGS.input,
        piece_dicts_path=None
        if ARGS.no_h5
        else Path(ARGS.h5_dir / f"pieces_{data_type}_seed_{ARGS.seed}.pkl"),
        file_ids=file_ids,
        specific_id=ARGS.id,
    )

    if ARGS.force_segs:
        ARGS.max_chord_length = np.inf

    evaluate(
        from_args(models, ARGS),
        pieces,
        output_tsv_dir=ARGS.output,
        force_segs=ARGS.force_segs,
    )
