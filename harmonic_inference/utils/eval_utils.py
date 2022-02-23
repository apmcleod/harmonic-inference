"""Utility functions for evaluating model outputs."""
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from ms3 import Parse

import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.chord import Chord
from harmonic_inference.data.data_types import TRIAD_REDUCTION, PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import State


def evaluate_features(results_df: pd.DataFrame, features: List[str]) -> float:
    """
    Evaluate a given list of features from a results_df.

    Parameters
    ----------
    results_df : pd.DataFrame
        A df containing the results of a harmonic inference procedure, returned by
        get_results_df.
    features : List[str]
        A list of the features to check. This will check if est_`feature` == gt_`feature`
        for all features in this list. Legal features are:
            - key
            - tonic
            - mode
            - chord
            - root
            - chord_type
            - triad
            - inversion

    Returns
    -------
    accuracy : float
        A weighted average of the given features in the results_df, weighted by duration.
    """
    correct_mask = np.full(len(results_df), True)
    for feature in features:
        correct_mask &= results_df[f"gt_{feature}"] == results_df[f"est_{feature}"]

    return float(np.average(correct_mask, weights=results_df["duration"]))


def get_results_df(
    piece: Piece,
    state: State,
    output_root_type: PitchType,
    output_tonic_type: PitchType,
    chord_root_type: PitchType,
    key_tonic_type: PitchType,
) -> pd.DataFrame:
    """
    Evaluate the piece's estimated chords.

    Parameters
    ----------
    piece : Piece
        The piece, containing the ground truth harmonic structure.
    state : State
        The state, containing the estimated harmonic structure.
    chord_root_type : PitchType
        The pitch type used for chord roots.
    key_tonic_type : PitchType
        The pitch type used for key tonics.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame containing the results of the given state, with the given settings.
    """
    labels_list = []

    # GT chords
    gt_chords = piece.get_chords()
    gt_changes = piece.get_chord_change_indices()
    gt_chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    gt_chord_roots = np.zeros(len(piece.get_inputs()), dtype=int)
    gt_chord_types = np.zeros(len(piece.get_inputs()), dtype=object)
    gt_chord_triads = np.zeros(len(piece.get_inputs()), dtype=object)
    gt_chord_inversions = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(gt_chords, gt_changes, gt_changes[1:]):
        chord = chord.to_pitch_type(chord_root_type)
        gt_chord_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )
        gt_chord_roots[start:end] = chord.root
        gt_chord_types[start:end] = chord.chord_type
        gt_chord_triads[start:end] = TRIAD_REDUCTION[chord.chord_type]
        gt_chord_inversions[start:end] = chord.inversion

    last_chord = gt_chords[-1].to_pitch_type(chord_root_type)
    gt_chord_labels[gt_changes[-1] :] = last_chord.get_one_hot_index(
        relative=False, use_inversion=True, pad=False
    )
    gt_chord_roots[gt_changes[-1] :] = last_chord.root
    gt_chord_types[gt_changes[-1] :] = last_chord.chord_type
    gt_chord_triads[gt_changes[-1] :] = TRIAD_REDUCTION[last_chord.chord_type]
    gt_chord_inversions[gt_changes[-1] :] = last_chord.inversion

    # Est chords
    chords, changes = state.get_chords()
    estimated_chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    estimated_chord_roots = np.zeros(len(piece.get_inputs()), dtype=int)
    estimated_chord_types = np.zeros(len(piece.get_inputs()), dtype=object)
    estimated_chord_triads = np.zeros(len(piece.get_inputs()), dtype=object)
    estimated_chord_inversions = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(chords, changes[:-1], changes[1:]):
        root, chord_type, inv = hu.get_chord_from_one_hot_index(chord, output_root_type)
        root = hu.get_pitch_from_string(
            hu.get_pitch_string(root, output_root_type), chord_root_type
        )
        chord = hu.get_chord_one_hot_index(chord_type, root, chord_root_type, inversion=inv)
        estimated_chord_labels[start:end] = chord
        estimated_chord_roots[start:end] = root
        estimated_chord_types[start:end] = chord_type
        estimated_chord_triads[start:end] = TRIAD_REDUCTION[chord_type]
        estimated_chord_inversions[start:end] = inv

    # GT keys
    gt_keys = piece.get_keys()
    gt_changes = piece.get_key_change_input_indices()
    gt_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    gt_key_tonics = np.zeros(len(piece.get_inputs()), dtype=int)
    gt_key_modes = np.zeros(len(piece.get_inputs()), dtype=object)
    for key, start, end in zip(gt_keys, gt_changes, gt_changes[1:]):
        key = key.to_pitch_type(key_tonic_type)
        gt_key_labels[start:end] = key.get_one_hot_index()
        gt_key_tonics[start:end] = key.relative_tonic
        gt_key_modes[start:end] = key.relative_mode
    last_key = gt_keys[-1].to_pitch_type(key_tonic_type)
    gt_key_labels[gt_changes[-1] :] = last_key.get_one_hot_index()
    gt_key_tonics[gt_changes[-1] :] = last_key.relative_tonic
    gt_key_modes[gt_changes[-1] :] = last_key.relative_mode

    # Est keys
    keys, changes = state.get_keys()
    estimated_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    estimated_key_tonics = np.zeros(len(piece.get_inputs()), dtype=int)
    estimated_key_modes = np.zeros(len(piece.get_inputs()), dtype=object)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        tonic, mode = hu.get_key_from_one_hot_index(key, output_tonic_type)
        tonic = hu.get_pitch_from_string(
            hu.get_pitch_string(tonic, output_tonic_type), key_tonic_type
        )
        key = hu.get_key_one_hot_index(mode, tonic, key_tonic_type)
        estimated_key_labels[start:end] = key
        estimated_key_tonics[start:end] = tonic
        estimated_key_modes[start:end] = mode

    chord_label_list = hu.get_chord_label_list(chord_root_type, use_inversions=True)
    key_label_list = hu.get_key_label_list(key_tonic_type)

    for (
        duration,
        est_chord_label,
        gt_chord_label,
        est_key_label,
        gt_key_label,
        gt_tonic,
        gt_mode,
        gt_root,
        gt_chord_type,
        gt_triad,
        gt_inversion,
        est_tonic,
        est_mode,
        est_root,
        est_chord_type,
        est_triad,
        est_inversion,
    ) in zip(
        piece.get_duration_cache(),
        estimated_chord_labels,
        gt_chord_labels,
        estimated_key_labels,
        gt_key_labels,
        gt_key_tonics,
        gt_key_modes,
        gt_chord_roots,
        gt_chord_types,
        gt_chord_triads,
        gt_chord_inversions,
        estimated_key_tonics,
        estimated_key_modes,
        estimated_chord_roots,
        estimated_chord_types,
        estimated_chord_triads,
        estimated_chord_inversions,
    ):
        if duration == 0:
            continue

        labels_list.append(
            {
                "gt_key": key_label_list[gt_key_label],
                "gt_tonic": gt_tonic,
                "gt_mode": gt_mode,
                "gt_chord": chord_label_list[gt_chord_label],
                "gt_root": gt_root,
                "gt_chord_type": gt_chord_type,
                "gt_triad": gt_triad,
                "gt_inversion": gt_inversion,
                "est_key": key_label_list[est_key_label],
                "est_tonic": est_tonic,
                "est_mode": est_mode,
                "est_chord": chord_label_list[est_chord_label],
                "est_root": est_root,
                "est_chord_type": est_chord_type,
                "est_triad": est_triad,
                "est_inversion": est_inversion,
                "duration": duration,
            }
        )

    return pd.DataFrame(labels_list)


def get_labels_df(piece: Piece, tpc_c: int = hc.TPC_C) -> pd.DataFrame:
    """
    Create and return a labels_df for a given Piece, containing all chord and key
    information for each segment of the piece, in all formats (TPC and MIDI pitch).

    Parameters
    ----------
    piece : Piece
        The piece to create a labels_df for.
    tpc_c : int
        Where C should be in the TPC output.

    Returns
    -------
    labels_df : pd.DataFrame
        A labels_df, with the columns:
            - chord_root_tpc
            - chord_root_midi
            - chord_type
            - chord_inversion
            - chord_suspension_midi
            - chord_suspension_tpc
            - key_tonic_tpc
            - key_tonic_midi
            - key_mode
            - duration
            - mc
            - onset_mc
    """

    def get_suspension_strings(chord: Chord) -> Tuple[str, str]:
        """
        Get the tpc and midi strings for the given chord's suspension and changes.

        Parameters
        ----------
        chord : Chord
            The chord whose string to return.

        Returns
        -------
        tpc_string : str
            A string representing the mapping of altered pitches in the given chord.
            Each altered pitch is represented as "orig:new", where orig is the pitch in the default
            chord voicing, and "new" is the altered pitch that is actually present. For added
            pitches, "orig" is the empty string. "new" can be prefixed with a "+", in which
            case this pitch is present in an upper octave. Pitches are represented as TPC,
            and multiple alterations are separated by semicolons.
        midi_string : str
            The same format as tpc_string, but using a MIDI pitch representation.
        """
        if chord.suspension is None:
            return "", ""

        change_mapping = hu.get_added_and_removed_pitches(
            chord.root,
            chord.chord_type,
            chord.suspension,
            chord.key_tonic,
            chord.key_mode,
        )

        mappings_midi = []
        mappings_tpc = []
        for orig, new in change_mapping.items():
            if orig == "":
                orig_midi = ""
                orig_tpc = ""
            else:
                orig_midi = str(
                    hu.get_pitch_from_string(
                        hu.get_pitch_string(int(orig), PitchType.TPC), PitchType.MIDI
                    )
                )
                orig_tpc = str(int(orig) - hc.TPC_C + tpc_c)

            prefix = ""
            if new[0] == "+":
                prefix = "+"
                new = new[1:]

            new_midi = prefix + str(
                hu.get_pitch_from_string(
                    hu.get_pitch_string(int(new), PitchType.TPC), PitchType.MIDI
                )
            )
            new_tpc = prefix + str(int(new) - hc.TPC_C + tpc_c)

            mappings_midi.append(f"{orig_midi}:{new_midi}")
            mappings_tpc.append(f"{orig_tpc}:{new_tpc}")

        return ";".join(mappings_tpc), ";".join(mappings_midi)

    labels_list = []

    chords = piece.get_chords()
    onsets = [note.onset for note in piece.get_inputs()]
    chord_changes = piece.get_chord_change_indices()
    chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    chord_suspensions_midi = np.full(len(piece.get_inputs()), "", dtype=object)
    chord_suspensions_tpc = np.full(len(piece.get_inputs()), "", dtype=object)
    for chord, start, end in zip(chords, chord_changes, chord_changes[1:]):
        chord_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )

        tpc_string, midi_string = get_suspension_strings(chord)

        chord_suspensions_tpc[start:end] = tpc_string
        chord_suspensions_midi[start:end] = midi_string

    chord_labels[chord_changes[-1] :] = chords[-1].get_one_hot_index(
        relative=False, use_inversion=True, pad=False
    )

    tpc_string, midi_string = get_suspension_strings(chords[-1])

    chord_suspensions_tpc[chord_changes[-1] :] = tpc_string
    chord_suspensions_midi[chord_changes[-1] :] = midi_string

    keys = piece.get_keys()
    key_changes = piece.get_key_change_input_indices()
    key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(keys, key_changes, key_changes[1:]):
        key_labels[start:end] = key.get_one_hot_index()
    key_labels[key_changes[-1] :] = keys[-1].get_one_hot_index()

    chord_labels_list = hu.get_chord_from_one_hot_index(
        slice(len(hu.get_chord_label_list(PitchType.TPC))), PitchType.TPC
    )
    key_labels_list = hu.get_key_from_one_hot_index(
        slice(len(hu.get_key_label_list(PitchType.TPC))), PitchType.TPC
    )

    for duration, chord_label, key_label, suspension_tpc, suspension_midi, onset in zip(
        piece.get_duration_cache(),
        chord_labels,
        key_labels,
        chord_suspensions_tpc,
        chord_suspensions_midi,
        onsets,
    ):
        if duration == 0:
            continue

        root_tpc, chord_type, inversion = chord_labels_list[chord_label]
        tonic_tpc, mode = key_labels_list[key_label]

        root_midi = hu.get_pitch_from_string(
            hu.get_pitch_string(root_tpc, PitchType.TPC), PitchType.MIDI
        )
        tonic_midi = hu.get_pitch_from_string(
            hu.get_pitch_string(tonic_tpc, PitchType.TPC), PitchType.MIDI
        )

        labels_list.append(
            {
                "chord_root_tpc": root_tpc - hc.TPC_C + tpc_c,
                "chord_root_midi": root_midi,
                "chord_type": chord_type,
                "chord_inversion": inversion,
                "chord_suspension_tpc": suspension_tpc,
                "chord_suspension_midi": suspension_midi,
                "key_tonic_tpc": tonic_tpc - hc.TPC_C + tpc_c,
                "key_tonic_midi": tonic_midi,
                "key_mode": mode,
                "duration": duration,
                "mc": onset[0],
                "mn_onset": onset[1],
            }
        )

    return pd.DataFrame(labels_list)


def get_annotation_df(
    state: State,
    piece: Piece,
    root_type: PitchType,
    tonic_type: PitchType,
) -> pd.DataFrame:
    """
    Get a df containing the labels of the given state.

    Parameters
    ----------
    state : State
        The state containing harmony annotations.
    piece : Piece
        The piece which was used as input when creating the given state.
    root_type : PitchType
        The pitch type to use for chord root labels.
    tonic_type : PitchType
        The pitch type to use for key tonic annotations.

    Returns
    -------
    annotation_df : pd.DataFrame[type]
        A DataFrame containing the harmony annotations from the given state.
    """
    labels_list = []

    chords, changes = state.get_chords()
    estimated_chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(chords, changes[:-1], changes[1:]):
        estimated_chord_labels[start:end] = chord

    keys, changes = state.get_keys()
    estimated_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        estimated_key_labels[start:end] = key

    chord_label_list = hu.get_chord_label_list(root_type, use_inversions=True)
    key_label_list = hu.get_key_label_list(tonic_type)

    prev_est_key_string = None
    prev_est_chord_string = None

    for duration, note, est_chord_label, est_key_label in zip(
        piece.get_duration_cache(),
        piece.get_inputs(),
        estimated_chord_labels,
        estimated_key_labels,
    ):
        if duration == 0:
            continue

        est_chord_string = chord_label_list[est_chord_label]
        est_key_string = key_label_list[est_key_label]

        # No change in labels
        if est_chord_string == prev_est_chord_string and est_key_string == prev_est_key_string:
            continue

        if est_key_string != prev_est_key_string:
            labels_list.append(
                {
                    "label": est_key_string,
                    "mc": note.onset[0],
                    "mc_onset": note.mc_onset,
                    "mn_onset": note.onset[1],
                }
            )

        if est_chord_string != prev_est_chord_string:
            labels_list.append(
                {
                    "label": est_chord_string,
                    "mc": note.onset[0],
                    "mc_onset": note.mc_onset,
                    "mn_onset": note.onset[1],
                }
            )

        prev_est_key_string = est_key_string
        prev_est_chord_string = est_chord_string

    return pd.DataFrame(labels_list)


def get_results_annotation_df(
    state: State,
    piece: Piece,
    root_type: PitchType,
    tonic_type: PitchType,
) -> pd.DataFrame:
    """
    Get a df containing the full labels of the given state, color-coded in terms of their
    accuracy according to the ground truth harmony in the given piece. This can be used
    to attach color-coded labels to a musescore3 score using ms3.

    Parameters
    ----------
    state : State
        The state, containing the estimated harmonic structure.
    piece : Piece
        The piece, containing the ground truth harmonic structure.
    root_type : PitchType
        The pitch type used for chord roots.
    tonic_type : PitchType
        The pitch type used for key tonics.

    Returns
    -------
    label_df : pd.DataFrame
        A DataFrame containing the labels of the given state.
    """
    labels_list = []

    gt_chord_labels = np.full(len(piece.get_inputs()), -1, dtype=int)
    if len(piece.get_chords()) > 0:
        gt_chords = piece.get_chords()
        gt_changes = piece.get_chord_change_indices()
        for chord, start, end in zip(gt_chords, gt_changes, gt_changes[1:]):
            gt_chord_labels[start:end] = chord.get_one_hot_index(
                relative=False, use_inversion=True, pad=False
            )
        gt_chord_labels[gt_changes[-1] :] = gt_chords[-1].get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )

    chords, changes = state.get_chords()
    estimated_chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(chords, changes[:-1], changes[1:]):
        estimated_chord_labels[start:end] = chord

    gt_key_labels = np.full(len(piece.get_inputs()), -1, dtype=int)
    if len(piece.get_keys()) > 0:
        gt_keys = piece.get_keys()
        gt_changes = piece.get_key_change_input_indices()
        gt_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
        for key, start, end in zip(gt_keys, gt_changes, gt_changes[1:]):
            gt_key_labels[start:end] = key.get_one_hot_index()
        gt_key_labels[gt_changes[-1] :] = gt_keys[-1].get_one_hot_index()

    keys, changes = state.get_keys()
    estimated_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        estimated_key_labels[start:end] = key

    chord_label_list = hu.get_chord_label_list(root_type, use_inversions=True)
    key_label_list = hu.get_key_label_list(tonic_type)

    prev_gt_chord_string = None
    prev_gt_key_string = None
    prev_est_key_string = None
    prev_est_chord_string = None

    for duration, note, est_chord_label, gt_chord_label, est_key_label, gt_key_label in zip(
        piece.get_duration_cache(),
        piece.get_inputs(),
        estimated_chord_labels,
        gt_chord_labels,
        estimated_key_labels,
        gt_key_labels,
    ):
        if duration == 0:
            continue

        gt_chord_string = chord_label_list[gt_chord_label]
        gt_key_string = key_label_list[gt_key_label]

        est_chord_string = chord_label_list[est_chord_label]
        est_key_string = key_label_list[est_key_label]

        # No change in labels
        if (
            gt_chord_string == prev_gt_chord_string
            and gt_key_string == prev_gt_key_string
            and est_chord_string == prev_est_chord_string
            and est_key_string == prev_est_key_string
        ):
            continue

        if gt_key_string != prev_gt_key_string or est_key_string != prev_est_key_string:
            gt_tonic, gt_mode = hu.get_key_from_one_hot_index(int(gt_key_label), tonic_type)
            est_tonic, est_mode = hu.get_key_from_one_hot_index(int(est_key_label), tonic_type)

            if gt_tonic == est_tonic and gt_mode == est_mode:
                color = "green"
            elif gt_tonic == est_tonic:
                color = "yellow"
            else:
                color = "red"

            labels_list.append(
                {
                    "label": est_key_string if est_key_string != prev_est_key_string else "--",
                    "mc": note.onset[0],
                    "mc_onset": note.mc_onset,
                    "mn_onset": note.onset[1],
                    "color_name": color,
                }
            )

        if gt_chord_string != prev_gt_chord_string or est_chord_string != prev_est_chord_string:
            gt_root, gt_chord_type, gt_inversion = hu.get_chord_from_one_hot_index(
                gt_chord_label, root_type, use_inversions=True
            )

            est_root, est_chord_type, est_inversion = hu.get_chord_from_one_hot_index(
                est_chord_label, root_type, use_inversions=True
            )

            if (
                gt_root == est_root
                and gt_chord_type == est_chord_type
                and gt_inversion == est_inversion
            ):
                color = "green"
            elif (
                gt_root == est_root
                and TRIAD_REDUCTION[gt_chord_type] == TRIAD_REDUCTION[est_chord_type]
            ):
                color = "yellow"
            else:
                color = "red"

            labels_list.append(
                {
                    "label": est_chord_string
                    if est_chord_string != prev_est_chord_string
                    else "--",
                    "mc": note.onset[0],
                    "mc_onset": note.mc_onset,
                    "mn_onset": note.onset[1],
                    "color_name": color,
                }
            )

        prev_gt_key_string = gt_key_string
        prev_gt_chord_string = gt_chord_string
        prev_est_key_string = est_key_string
        prev_est_chord_string = est_chord_string

    return pd.DataFrame(labels_list)


def write_labels_to_score(
    labels_dir: Union[str, Path],
    annotations_dir: Union[str, Path],
    basename: str,
):
    """
    Write the annotation labels from a given directory onto a musescore file.

    Parameters
    ----------
    labels_dir : Union[str, Path]
        The directory containing the tsv file containing the model's annotations.
    annotations_dir : Union[str, Path]
        The directory containing the ground truth annotations and MS3 score file.
    basename : str
        The basename of the annotation TSV and the ground truth annotations/MS3 file.
    """
    if isinstance(labels_dir, Path):
        labels_dir = str(labels_dir)

    if isinstance(annotations_dir, Path):
        annotations_dir = str(annotations_dir)

    # Add musescore and tsv suffixes to filename match
    filename_regex = re.compile(basename + "\\.(mscx|tsv)")

    # Parse scores and tsvs
    parse = Parse(annotations_dir, file_re=filename_regex)
    parse.add_dir(labels_dir, key="labels", file_re=filename_regex)
    parse.parse()

    # Write annotations to score
    parse.add_detached_annotations("MS3", "labels")
    parse.attach_labels(staff=2, voice=1, check_for_clashes=False)

    # Write score out to file
    parse.store_mscx(root_dir=labels_dir, suffix="_inferred", overwrite=True)


def average_results(results_path: Union[Path, str], split_on: str = " = ") -> Dict[str, float]:
    """
    Average accuracy values from a file.

    Parameters
    ----------
    results_path : Union[Path, str]
        The file to read results from.
    split_on : str
        The symbol which separates an accuracy's key from its value.

    Returns
    -------
    averages : Dict[str, float]
        A dictionary mapping each accuracy key to its average value.
    """
    averages = defaultdict(list)

    with open(results_path, "r") as results_file:
        for line in results_file:
            if split_on not in line:
                continue

            line_split = line.split(split_on)
            if len(line_split) != 2:
                continue

            key, value = line_split
            key = key.strip()

            if "accuracy" in key:
                averages[key].append(float(value))

    return {key: np.mean(value_list) for key, value_list in averages.items()}


def log_state(state: State, piece: Piece, root_type: PitchType, tonic_type: PitchType):
    """
    Print the full state harmonic structure (in comparison to that of the given piece),
    as debug logging messages.

    Parameters
    ----------
    state : State
        The state whose harmonic structure to print.
    piece : Piece
        The piece with the ground truth harmonic structure, to note where the state's
        structure is incorrect.
    root_type : PitchType
        The pitch type used for the chord roots.
    tonic_type : PitchType
        The pitch type used for the key tonics.
    """
    gt_chords = piece.get_chords()
    gt_changes = piece.get_chord_change_indices()
    gt_chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(gt_chords, gt_changes, gt_changes[1:]):
        gt_chord_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )
    gt_chord_labels[gt_changes[-1] :] = gt_chords[-1].get_one_hot_index(
        relative=False, use_inversion=True, pad=False
    )

    chords, changes = state.get_chords()
    est_chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(chords, changes[:-1], changes[1:]):
        est_chord_labels[start:end] = chord

    gt_keys = piece.get_keys()
    gt_changes = piece.get_key_change_input_indices()
    gt_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(gt_keys, gt_changes, gt_changes[1:]):
        gt_key_labels[start:end] = key.get_one_hot_index()
    gt_key_labels[gt_changes[-1] :] = gt_keys[-1].get_one_hot_index()

    keys, changes = state.get_keys()
    est_key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        est_key_labels[start:end] = key

    chord_label_list = hu.get_chord_label_list(root_type, use_inversions=True)
    key_label_list = hu.get_key_label_list(tonic_type)

    structure = list(zip(gt_key_labels, gt_chord_labels, est_key_labels, est_chord_labels))
    changes = [True] + [
        prev_structure != next_structure
        for prev_structure, next_structure in zip(structure, structure[1:])
    ]

    input_starts = np.array([note.onset for note in piece.get_inputs()])[changes]
    input_ends = list(input_starts[1:]) + [piece.get_inputs()[-1].offset]

    indexes = np.arange(len(changes))[changes]
    durations = [
        np.sum(piece.get_duration_cache()[start:end])
        for start, end in zip(indexes, list(indexes[1:]) + [len(changes)])
    ]

    for gt_chord, est_chord, gt_key, est_key, input_start, input_end, duration in zip(
        np.array(gt_chord_labels)[changes],
        np.array(est_chord_labels)[changes],
        np.array(gt_key_labels)[changes],
        np.array(est_key_labels)[changes],
        input_starts,
        input_ends,
        durations,
    ):
        gt_chord_label = chord_label_list[gt_chord]
        est_chord_label = chord_label_list[est_chord]

        gt_key_label = key_label_list[gt_key]
        est_key_label = key_label_list[est_key]

        logging.debug("%s - %s (duration %s):", input_start, input_end, duration)
        logging.debug("    Estimated structure: %s\t%s", est_key_label, est_chord_label)
        if gt_key_label != est_key_label or gt_chord_label != est_chord_label:
            logging.debug("      Correct structure: %s\t%s", gt_key_label, gt_chord_label)
