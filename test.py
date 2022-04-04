"""Script to test (evaluate) a joint model for harmonic inference."""
import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Set, Union

import h5py
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


def load_forces_from_json(json_path: Union[str, Path]) -> Dict[str, Dict[str, Union[Set, Dict]]]:
    """
    Load forced labels, changes, and non-changes from a json file and return them
    in a dictionary that can be passed through to JointModel.get_harmony(...) as kwargs.

    Parameters
    ----------
    json_path : Union[str, Path]
        A reference to a json file containing forced labels, changes, and non-changes,
        to be used for JointModel.get_harmony(...). It may have the following keys:
            - "forced_chord_changes": A list of integers at which the chord must change.
            - "forced_chord_non_changes": A list of integers at which the chord cannot change.
            - "forced_key_changes": A list of integers at which the key must change.
            - "forced_key_non_changes": A list of integers at which the key cannot change.
            - "forced_chords": A dictionary mapping the string form of a tuple in the form
                               (start, end) to a chord_id, saying that the input indexes on
                               the range start (inclusive) to end (exclusive) must be output
                               as the given chord_id.
            - "forced_keys": Same as forced_chords, but for keys.

    Returns
    -------
    forces_kwargs: Dict[str, Dict[str, Union[Set, Dict]]]
        A nested dictionary containing the loaded keyword arguments for each input.
        The outer-most keys should reference a specific input by string name,
        or be the keyword "default", in which case the loaded kwargs will be used for all
        input pieces not otherwise matched by string name.
        In the inner dictionaries, keyword arguments have been loaded (with the correct types)
        from the json file that can be passed directly as kwargs to JointModel.get_harmony(...)
        for that particular piece.
    """

    def load_forces_from_nested_json(raw_data: Dict) -> Dict[str, Union[Set, Dict]]:
        """
        Load an inner forces_kwargs dict from a nested json forces_kwargs dict data.

        Parameters
        ----------
        raw_data : Dict
            The inner nested dictionary from which we will load the kwargs.
            See load_forces_from_json for details.

        Returns
        -------
        Dict[str, Union[Set, Dict]]
            The kwargs for a single piece, unnested.
        """
        forces_kwargs = dict()

        for key in [
            "forced_chord_changes",
            "forced_chord_non_changes",
            "forced_key_changes",
            "forced_key_non_changes",
        ]:
            if key in raw_data:
                forces_kwargs[key] = set(raw_data[key])

        for key in ["forced_chords", "forced_keys"]:
            if key in raw_data:
                forces_kwargs[key] = {
                    tuple(map(int, range_tuple_str[1:-1].split(","))): label_id
                    for range_tuple_str, label_id in raw_data[key].items()
                }

        for key in raw_data:
            if key not in [
                "forced_chord_changes",
                "forced_chord_non_changes",
                "forced_key_changes",
                "forced_key_non_changes",
                "forced_chords",
                "forced_keys",
            ]:
                logging.warning(
                    "--forces-json inner key not recognized: %s. Ignoring that key.", key
                )

        logging.info("Forces:" if len(forces_kwargs) > 0 else "Forces: None")
        for key, item in sorted(forces_kwargs.items()):
            if type(item) == dict:
                logging.info("    %s:", key)
                for inner_key, inner_item in sorted(item.items()):
                    logging.info("        %s = %s", inner_key, inner_item)
            else:
                logging.info("    %s = %s", key, item)

        return forces_kwargs

    with open(json_path, "r") as json_file:
        raw_data = json.load(json_file)

    if (
        "forced_chord_changes" in raw_data
        or "forced_chord_non_changes" in raw_data
        or "forced_key_changes" in raw_data
        or "forced_key_non_changes" in raw_data
        or "forced_chords" in raw_data
        or "forced_keys" in raw_data
    ):
        logging.info(
            "Given --json-forces not a nested, piece-specific mapping. Treating as default for "
            "all inputs."
        )
        raw_data = {"default": raw_data}

    all_forces_kwargs = {}

    for key, nested_raw_data in raw_data.items():
        logging.info("Loading forces for %s", key)
        all_forces_kwargs[key] = load_forces_from_nested_json(nested_raw_data)

    return all_forces_kwargs


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
    forces_path: Union[Path, str] = None,
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
    forces_path : Union[Path, str]
        The path to a json file containing forced labels, changes, or non-changes.
    force_segs : bool
        Force the model to use the ground truth segmentations.
    """
    if output_tsv_dir is not None:
        output_tsv_dir = Path(output_tsv_dir)

    all_forces_dict = {} if forces_path is None else load_forces_from_json(forces_path)

    for piece in tqdm(pieces, desc="Getting harmony for pieces"):
        piece: Piece
        if piece.name is not None:
            logging.info("Running piece %s", piece.name)

        # Load forces dict for this file
        forces_dict = None
        if piece.name is not None:
            for key in set(all_forces_dict.keys()) - set(["default"]):
                if key in piece.name:
                    logging.info("Using forces with key %s", key)
                    forces_dict = all_forces_dict[key]
                    break

        if forces_dict is None:
            forces_dict = all_forces_dict["default"] if "default" in all_forces_dict else {}

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

        state = model.get_harmony(piece, **forces_dict)

        if state is None:
            logging.info("Returned None")
            continue

        # Create results dfs
        results_annotation_df = eu.get_results_annotation_df(
            state,
            piece,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            model.chord_classifier.use_inversions,
            model.chord_classifier.reduction,
            use_chord_pitches=True,
        )

        results_df = eu.get_results_df(
            piece,
            state,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            PitchType.TPC,
            PitchType.TPC,
        )

        results_midi_df = eu.get_results_df(
            piece,
            state,
            model.CHORD_OUTPUT_TYPE,
            model.KEY_OUTPUT_TYPE,
            PitchType.MIDI,
            PitchType.MIDI,
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
                results_annotation_df.to_csv(output_tsv_path, sep="\t")
                logging.info("Results annotation TSV written out to %s", output_tsv_path)
            except Exception:
                logging.exception("Error writing to csv %s", output_tsv_path)
                logging.debug(results_annotation_df)

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
        "--forces-json",
        type=Path,
        default=None,
        help=(
            "A json file containing forced labels changes or non-changes, to be passed to "
            "JointModel.get_harmony(...)."
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

    evaluate(
        from_args(models, ARGS),
        pieces,
        output_tsv_dir=ARGS.output,
        forces_path=ARGS.forces_json,
        force_segs=ARGS.force_segs,
    )
