"""Test the CPM by taking as input the GT pieces and estimating pitches."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import harmonic_inference.utils.eval_utils as eu
from harmonic_inference.data.data_types import (
    MAJOR_MINOR_REDUCTION,
    TRIAD_REDUCTION,
    ChordType,
    PitchType,
)
from harmonic_inference.data.datasets import ChordPitchesDataset
from harmonic_inference.data.piece import Piece, ScorePiece, get_score_piece_from_dict
from harmonic_inference.models.chord_pitches_models import (
    ChordPitchesModel,
    NoteBasedChordPitchesModel,
    decode_cpm_note_based_outputs,
    decode_cpm_outputs,
    get_rule_based_cpm_outputs,
)
from harmonic_inference.models.joint_model import (
    CPM_CHORD_TONE_THRESHOLD_DEFAULT,
    CPM_NON_CHORD_TONE_ADD_THRESHOLD_DEFAULT,
    CPM_NON_CHORD_TONE_REPLACE_THRESHOLD_DEFAULT,
    add_joint_model_args,
)
from harmonic_inference.utils.data_utils import load_models_from_argparse, load_pieces
from harmonic_inference.utils.harmonic_constants import MAX_CHORD_PITCH_INTERVAL_TPC
from harmonic_inference.utils.harmonic_utils import get_chord_inversion_count


def evaluate_cpm(
    cpm: ChordPitchesModel,
    pieces: List[Piece],
    output_tsv_dir: Union[str, Path] = None,
    cpm_chord_tone_threshold: float = CPM_CHORD_TONE_THRESHOLD_DEFAULT,
    cpm_non_chord_tone_add_threshold: float = CPM_NON_CHORD_TONE_ADD_THRESHOLD_DEFAULT,
    cpm_non_chord_tone_replace_threshold: float = CPM_NON_CHORD_TONE_REPLACE_THRESHOLD_DEFAULT,
    merge_changes: bool = False,
    merge_reduction: Dict[ChordType, ChordType] = None,
    rule_based: bool = False,
) -> None:
    """
    _summary_

    Parameters
    ----------
    cpm : ChordPitchesModel
        The CPM to evaluate.
    pieces : List[Piece]
        A List of pieces to run the CPM on.
    output_tsv_dir : Union[str, Path]
        The directory to write out label tsvs to.
    cpm_chord_tone_threshold : float
        The threshold above which a default chord tone must reach in the CPM output
        in order to be considered present in a given chord.
    cpm_non_chord_tone_add_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to be an added tone in a given chord.
    cpm_non_chord_tone_replace_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to replace a chord tone in a given chord.
    merge_changes : bool
        Merge chords which differ only by their chord tone changes into single chords
        as input. The targets will remain unchanged, so the CPM will ideally split
        such chords in its post-processing step.
    merge_reduction : Dict[ChordType, ChordType]
        Merge chords which no longer differ after this chord type reduction together
        as input. The targets will remain unchanged, so the CPM will ideally split
        such chords in its post-processing step.
    rule_based : bool
        Generate the rule-based output rather than running any CPM.
    """
    for piece in tqdm(pieces):
        reduced_piece: ScorePiece = piece
        if merge_changes or merge_reduction is not None:
            reduced_piece = get_score_piece_from_dict(
                reduced_piece.measures_df, reduced_piece.to_dict(), name=reduced_piece.name
            )
            if merge_reduction is not None:
                for chord in reduced_piece.get_chords():
                    chord.chord_type = merge_reduction[chord.chord_type]
                    if chord.inversion >= get_chord_inversion_count(chord.chord_type):
                        chord.inversion = 0
            reduced_piece.merge_chords(merge_changes)

        dataset = ChordPitchesDataset([reduced_piece], **cpm.get_dataset_kwargs())
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        if not rule_based:
            outputs = []
            note_outputs = []
            for batch in dl:
                if isinstance(cpm, NoteBasedChordPitchesModel):
                    output, note_output = cpm.get_output(batch, return_notes=True)
                    outputs.extend(output.numpy())
                    note_outputs.extend(note_output.numpy())
                else:
                    outputs.extend(cpm.get_output(batch).numpy())

            # Stack all outputs
            lengths = np.array([len(n) for n in note_outputs])
            if np.all(lengths == lengths[0]):
                note_outputs_np = np.vstack(note_outputs)
            else:
                note_outputs_np = np.zeros((len(note_outputs), np.max(lengths)))
                for i, note_output in enumerate(note_outputs):
                    note_outputs_np[i][: len(note_output)] = note_output

        if rule_based:
            chord_pitches = get_rule_based_cpm_outputs(
                reduced_piece.get_chord_note_inputs(window=dataset.window[0], notes_only=True),
                reduced_piece.get_chords(),
                np.vstack(
                    [
                        chord.get_chord_pitches_target_vector(default=True)
                        for chord in reduced_piece.get_chords()
                    ]
                ),
                [TRIAD_REDUCTION[chord.chord_type] for chord in reduced_piece.get_chords()],
                cpm.INPUT_PITCH,
                no_7ths=merge_reduction is not None,
                no_aug_and_dim=merge_reduction == MAJOR_MINOR_REDUCTION,
            )

        elif isinstance(cpm, NoteBasedChordPitchesModel):
            chord_pitches = decode_cpm_note_based_outputs(
                note_outputs_np,
                reduced_piece.get_chord_note_inputs(window=dataset.window[0], notes_only=True),
                reduced_piece.get_chords(),
                np.vstack(
                    [
                        chord.get_chord_pitches_target_vector(default=True)
                        for chord in reduced_piece.get_chords()
                    ]
                ),
                np.vstack(
                    [
                        chord.get_chord_pitches_target_vector(
                            reduction=TRIAD_REDUCTION, default=True
                        )
                        for chord in reduced_piece.get_chords()
                    ]
                ),
                [TRIAD_REDUCTION[chord.chord_type] for chord in reduced_piece.get_chords()],
                cpm_chord_tone_threshold,
                cpm_non_chord_tone_add_threshold,
                cpm_non_chord_tone_replace_threshold,
                cpm.INPUT_PITCH,
                no_aug_and_dim=merge_reduction == MAJOR_MINOR_REDUCTION,
            )

        else:
            chord_pitches = decode_cpm_outputs(
                np.vstack(outputs),
                np.vstack(
                    [
                        chord.get_chord_pitches_target_vector()
                        for chord in reduced_piece.get_chords()
                    ]
                ),
                np.vstack(
                    [
                        chord.get_chord_pitches_target_vector(reduction=TRIAD_REDUCTION)
                        for chord in reduced_piece.get_chords()
                    ]
                ),
                [TRIAD_REDUCTION[chord.chord_type] for chord in reduced_piece.get_chords()],
                cpm_chord_tone_threshold,
                cpm_non_chord_tone_add_threshold,
                cpm_non_chord_tone_replace_threshold,
                cpm.INPUT_PITCH,
                no_aug_and_dim=merge_reduction == MAJOR_MINOR_REDUCTION,
            )

        processed_piece = get_score_piece_from_dict(
            reduced_piece.measures_df, reduced_piece.to_dict(), name=reduced_piece.name
        )
        for pitches, chord in zip(chord_pitches, processed_piece.get_chords()):
            if isinstance(pitches, list):
                # Note-based (window-based) output. Need to do some special handling
                abs_pitches = []

                for window_pitches, window_index in pitches:
                    # Convert binary pitches array into root-relative indices
                    pitch_indices = np.where(window_pitches)[0]
                    if cpm.INPUT_PITCH == PitchType.TPC:
                        pitch_indices -= MAX_CHORD_PITCH_INTERVAL_TPC

                    # Convert root-relative indices into absolute pitches
                    abs_pitches.append(
                        [set([chord.root + pitch for pitch in pitch_indices]), window_index]
                    )

                # Remove list if only one window
                if len(abs_pitches) == 1:
                    abs_pitches = abs_pitches[0][0]

            else:
                # Convert binary pitches array into root-relative indices
                pitch_indices = np.where(pitches)[0]
                if cpm.INPUT_PITCH == PitchType.TPC:
                    pitch_indices -= MAX_CHORD_PITCH_INTERVAL_TPC

                # Convert root-relative indices into absolute pitches
                abs_pitches = set([chord.root + pitch for pitch in pitch_indices])

            chord.chord_pitches = abs_pitches

        processed_piece.split_chord_pitches_chords()

        # Create results dfs
        results_annotation_df = eu.get_results_annotation_df(
            processed_piece,
            piece,
            cpm.OUTPUT_PITCH,
            cpm.OUTPUT_PITCH,
            True,
            None,
            use_chord_pitches=True,
        )

        results_df = eu.get_results_df(
            piece,
            processed_piece,
            cpm.OUTPUT_PITCH,
            cpm.OUTPUT_PITCH,
            PitchType.TPC,
            PitchType.TPC,
            True,
            None,
        )

        results_midi_df = eu.get_results_df(
            piece,
            processed_piece,
            cpm.OUTPUT_PITCH,
            cpm.OUTPUT_PITCH,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a cpm on some data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Generate the rule-based output.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data"),
        help="The directory containing the raw corpus_data tsv files.",
    )

    parser.add_argument(
        "--average",
        type=Path,
        default=False,
        help="Calculate duration-weighted chord pitch accuracies from the given log file.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests on the actual test set, rather than the validation set.",
    )

    parser.add_argument(
        "--merge-changes",
        action="store_true",
        help="Merge input (but not target) chords which differ only by chord pitches.",
    )

    parser.add_argument(
        "--merge-reduction",
        type=str,
        choices=["triad", "Mm"],
        help="Merge input chords after reducing to their associated triad or 3rd type.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="outputs",
        help="The directory to write label tsvs to.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="The directory containing checkpoints for each type of model.",
    )

    DEFAULT_PATH = os.path.join(
        "`--checkpoint`", "cpm", "lightning_logs", "version_*", "checkpoints", "*.ckpt"
    )
    parser.add_argument(
        "--cpm",
        type=str,
        default=DEFAULT_PATH,
        help="A checkpoint file to load the cpm from.",
    )

    parser.add_argument(
        "--cpm-version",
        type=int,
        default=None,
        help=(
            "Specify a version number to load the model from. If given, --cpm is ignored"
            " and the cpm will be loaded from " + DEFAULT_PATH.replace("_*", "_`--cpm-version`")
        ),
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

    add_joint_model_args(parser, cpm_only=True)

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
        for key, average in eu.duration_weighted_pitch_average(ARGS.average).items():
            print(f"Average {key} = {average}")
        sys.exit(0)

    logging.info("Merge reduction: %s", ARGS.merge_reduction)
    logging.info("Merge changes: %s", ARGS.merge_changes)

    # Load models
    cpm = load_models_from_argparse(ARGS, model_type="cpm")["cpm"]

    data_type = "test" if ARGS.test else "valid"

    # Load data for cpm to get file_ids
    h5_path = Path(ARGS.h5_dir / f"ChordPitchesDataset_{data_type}_seed_{ARGS.seed}.h5")
    if h5_path.exists():
        with h5py.File(h5_path, "r") as h5_file:
            if "file_ids" not in h5_file:
                logging.error("file_ids not found in %s. Re-create with create_h5_data.py", h5_path)
                sys.exit(1)

            file_ids = list(h5_file["file_ids"])
    else:
        file_ids = None

    # Load pieces
    pieces = load_pieces(
        input_path=ARGS.input,
        piece_dicts_path=Path(ARGS.h5_dir / f"pieces_{data_type}_seed_{ARGS.seed}.pkl"),
        file_ids=file_ids,
    )

    evaluate_cpm(
        cpm,
        pieces,
        output_tsv_dir=ARGS.output,
        cpm_chord_tone_threshold=ARGS.cpm_chord_tone_threshold,
        cpm_non_chord_tone_add_threshold=ARGS.cpm_non_chord_tone_add_threshold,
        cpm_non_chord_tone_replace_threshold=ARGS.cpm_non_chord_tone_replace_threshold,
        merge_changes=ARGS.merge_changes,
        merge_reduction=(
            TRIAD_REDUCTION
            if ARGS.merge_reduction == "triad"
            else MAJOR_MINOR_REDUCTION
            if ARGS.merge_reduction == "Mm"
            else None
        ),
        rule_based=ARGS.rule_based,
    )
