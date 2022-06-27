"""Utility functions for evaluating model outputs."""
import logging
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from ms3 import Parse

import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.chord import Chord
from harmonic_inference.data.data_types import TRIAD_REDUCTION, ChordType, PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import State


def log_results_df_eval(results_df: pd.DataFrame):
    """
    Log evaluation metrics from a results_df.

    Parameters
    ----------
    results_df : pd.DataFrame
        A results_df to log.
    """
    chord_acc_with_pitches = evaluate_features(results_df, ["chord", "pitches"])
    chord_acc_full = evaluate_features(results_df, ["chord"])
    chord_acc_no_inv = evaluate_features(results_df, ["chord_type", "root"])
    chord_acc_triad = evaluate_features(results_df, ["triad", "root", "inversion"])
    chord_acc_triad_no_inv = evaluate_features(results_df, ["triad", "root"])
    chord_acc_root_only = evaluate_features(results_df, ["root"])
    chord_acc_pitches_only = evaluate_features(results_df, ["pitches"])
    pitch_acc_only_default = evaluate_features(
        results_df,
        ["pitches"],
        filter=results_df["gt_is_default"],
    )
    pitch_acc_only_non_default = evaluate_features(
        results_df,
        ["pitches"],
        filter=~results_df["gt_is_default"],
    )

    pitch_acc_default = evaluate_features(
        results_df,
        ["pitches"],
        filter=(
            results_df["gt_is_default"]
            & (results_df["gt_root"] == results_df["est_root"])
            & (results_df["gt_chord_type"] == results_df["est_chord_type"])
        ),
    )
    pitch_acc_non_default = evaluate_features(
        results_df,
        ["pitches"],
        filter=(
            ~results_df["gt_is_default"]
            & (results_df["gt_root"] == results_df["est_root"])
            & (results_df["gt_chord_type"] == results_df["est_chord_type"])
        ),
    )

    logging.info(
        "GT default duration = %s",
        results_df.loc[results_df["gt_is_default"], "duration"].sum(),
    )
    logging.info(
        "GT non-default duration = %s",
        results_df.loc[~results_df["gt_is_default"], "duration"].sum(),
    )

    logging.info("Chord accuracy = %s", chord_acc_full)
    logging.info("Chord accuracy with pitches = %s", chord_acc_with_pitches)
    logging.info("Chord accuracy, no inversions = %s", chord_acc_no_inv)
    logging.info("Chord accuracy, triads = %s", chord_acc_triad)
    logging.info("Chord accuracy, triad, no inversions = %s", chord_acc_triad_no_inv)
    logging.info("Chord accuracy, root only = %s", chord_acc_root_only)
    logging.info("Chord accuracy, pitches only = %s", chord_acc_pitches_only)
    logging.info(
        "Chord accuracy, pitches only, GT default only = %s",
        pitch_acc_only_default,
    )
    logging.info(
        "Chord accuracy, pitches only, GT non-default only = %s",
        pitch_acc_only_non_default,
    )

    logging.info(
        "Chord pitch accuracy on correct chord root+type, GT default only = %s",
        pitch_acc_default,
    )
    logging.info(
        "Chord pitch accuracy on correct chord root+type, GT non-default only = %s",
        pitch_acc_non_default,
    )

    key_acc_full = evaluate_features(results_df, ["key"])
    key_acc_tonic = evaluate_features(results_df, ["tonic"])

    logging.info("Key accuracy = %s", key_acc_full)
    logging.info("Key accuracy, tonic only = %s", key_acc_tonic)

    full_acc_with_pitches = evaluate_features(results_df, ["chord", "key", "pitches"])
    full_acc = evaluate_features(results_df, ["chord", "key"])
    logging.info("Full accuracy = %s", full_acc)
    logging.info("Full accuracy with pitches = %s", full_acc_with_pitches)


def evaluate_features(
    results_df: pd.DataFrame,
    features: List[str],
    filter: Union[pd.Series, List[bool]] = None,
) -> float:
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
            - pitches
            - is_default
    filter : Union[pd.Series, List[bool]]
        A boolean mask that can be used to filter the results_df before performing the
        evaluation. If given, the results_df is filtered first (leaving only rows where
        filter == True), before the evaluation is run on that resulting df.

    Returns
    -------
    accuracy : float
        A weighted average of the given features in the results_df, weighted by duration.
    """
    if filter is not None:
        results_df = results_df.loc[filter]

    if len(results_df) == 0:
        logging.warning("Filter contains nothing to evaluate. Returning 0.")
        return 0

    correct_mask = np.full(len(results_df), True)
    for feature in features:
        correct_mask &= results_df[f"gt_{feature}"] == results_df[f"est_{feature}"]

    return float(np.average(correct_mask, weights=results_df["duration"]))


def get_full_results_from_piece(
    piece: Piece,
    root_type: PitchType,
    tonic_type: PitchType,
    use_inversions: bool,
    reduction: Dict[ChordType, ChordType],
    chord_pitches_type: Union[None, str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Get Lists of chord and key info from a Piece object, returned in arrays with one
    element per piece input.

    Parameters
    ----------
    piece : Piece
        The piece whose labels we will return.
    root_type : PitchType
        The pitch type to use to encode chord roots.
    tonic_type : PitchType
        The pitch type to use to encode key tonics.
    use_inversions : bool
        Whether to use inversions or not in the labels.
    reduction : Dict[ChordType, ChordType]
        A chord type reduction to use.
    chord_pitches_type : Union[None, str]
        The encoding type for the returned chord pitches array. Either None (to return
        an array of all Nones), `set`, or `str`.

    Returns
    -------
    chord_labels : np.ndarray
        An array of chord one-hot indexes.
    chord_roots : np.ndarray
        An array of chord root pitches.
    chord_types : np.ndarray
        An array of chord types.
    chord_triads : np.ndarray
        An array of chord types, reduced to their triad form.
    chord_inversions : np.ndarray
        An array of chord inversions (or 0 if use_inversions is False).
    chord_pitches : np.ndarray
        An array of strings of chord pitches for each chord, or an array of Nones if
        chord_pitches_type is None.
    chord_is_default : np.ndarray
        An array of booleans indicating whether each chord uses its default pitches.
    key_labels : np.ndarray
        An array of key one-hot indexes, one per piece input.
    key_tonics : np.ndarray
        An array of key tonic pitches.
    key_modes : np.ndarray
        An array of key modes.
    """
    # Chord info
    chords = piece.get_chords()
    changes = piece.get_chord_change_indices()
    changes[0] = 0  # Bugfix for if first label isn't at the beginning
    chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    chord_roots = np.zeros(len(piece.get_inputs()), dtype=int)
    chord_types = np.zeros(len(piece.get_inputs()), dtype=object)
    chord_triads = np.zeros(len(piece.get_inputs()), dtype=object)
    chord_inversions = np.zeros(len(piece.get_inputs()), dtype=int)
    chord_pitches = np.full(len(piece.get_inputs()), None, dtype=object)
    chord_is_default = np.zeros(len(piece.get_inputs()), dtype=bool)
    for chord, start, end in zip(chords, changes, changes[1:]):
        chord = chord.to_pitch_type(root_type)
        chord_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=use_inversions, pad=False, reduction=reduction
        )
        chord_roots[start:end] = chord.root
        chord_types[start:end] = reduction[chord.chord_type] if reduction else chord.chord_type
        chord_triads[start:end] = TRIAD_REDUCTION[chord.chord_type]
        chord_inversions[start:end] = chord.inversion if use_inversions else 0
        if chord_pitches_type == "str":
            chord_pitches[start:end] = str(tuple(sorted(chord.chord_pitches)))
        elif chord_pitches_type == "set":
            chord_pitches[start:end] = chord.chord_pitches
        chord_is_default[start:end] = chord.chord_pitches == hu.get_default_chord_pitches(
            chord.root, chord.chord_type, chord.pitch_type
        )

    last_chord = chords[-1].to_pitch_type(root_type)
    chord_labels[changes[-1] :] = last_chord.get_one_hot_index(
        relative=False, use_inversion=use_inversions, pad=False, reduction=reduction
    )
    chord_roots[changes[-1] :] = last_chord.root
    chord_types[changes[-1] :] = (
        reduction[last_chord.chord_type] if reduction else last_chord.chord_type
    )
    chord_triads[changes[-1] :] = TRIAD_REDUCTION[last_chord.chord_type]
    chord_inversions[changes[-1] :] = last_chord.inversion if use_inversions else 0
    if chord_pitches_type == "str":
        chord_pitches[changes[-1] :] = str(tuple(sorted(last_chord.chord_pitches)))
    elif chord_pitches_type == "set":
        chord_pitches[changes[-1] :] = last_chord.chord_pitches
    chord_is_default[changes[-1] :] = last_chord.chord_pitches == hu.get_default_chord_pitches(
        last_chord.root, last_chord.chord_type, last_chord.pitch_type
    )

    # Key info
    keys = piece.get_keys()
    changes = piece.get_key_change_input_indices()
    changes[0] = 0  # Bugfix for if first label isn't at the beginning
    key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    key_tonics = np.zeros(len(piece.get_inputs()), dtype=int)
    key_modes = np.zeros(len(piece.get_inputs()), dtype=object)
    for key, start, end in zip(keys, changes, changes[1:]):
        key = key.to_pitch_type(tonic_type)
        key_labels[start:end] = key.get_one_hot_index()
        key_tonics[start:end] = key.relative_tonic
        key_modes[start:end] = key.relative_mode
    last_key = keys[-1].to_pitch_type(tonic_type)
    key_labels[changes[-1] :] = last_key.get_one_hot_index()
    key_tonics[changes[-1] :] = last_key.relative_tonic
    key_modes[changes[-1] :] = last_key.relative_mode

    return (
        chord_labels,
        chord_roots,
        chord_types,
        chord_triads,
        chord_inversions,
        chord_pitches,
        chord_is_default,
        key_labels,
        key_tonics,
        key_modes,
    )


def get_full_results_from_state(
    state: State,
    num_inputs: int,
    input_root_type: PitchType,
    input_tonic_type: PitchType,
    output_root_type: PitchType,
    output_tonic_type: PitchType,
    use_inversions: bool,
    reduction: Dict[ChordType, ChordType],
    chord_pitches_type: Union[None, str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Get Lists of chord and key info from a Piece object, returned in arrays with one
    element per piece input.

    Parameters
    ----------
    piece : Piece
        The piece whose labels we will return.
    num_inputs : int
        The number of inputs to return. This will be the length of the returned arrays.
    root_type : PitchType
        The pitch type to use to encode chord roots.
    tonic_type : PitchType
        The pitch type to use to encode key tonics.
    use_inversions : bool
        Whether to use inversions or not in the labels.
    reduction : Dict[ChordType, ChordType]
        A chord type reduction to use.
    use_chord_pitches : bool
        True to use chord pitches. False to instead return an array of Nones.

    Returns
    -------
    chord_labels : np.ndarray
        An array of chord one-hot indexes.
    chord_roots : np.ndarray
        An array of chord root pitches.
    chord_types : np.ndarray
        An array of chord types.
    chord_triads : np.ndarray
        An array of chord types, reduced to their triad form.
    chord_inversions : np.ndarray
        An array of chord inversions (or 0 if use_inversions is False).
    chord_pitches : np.ndarray
        An array of strings of chord pitches for each chord, or an array of Nones if
        use_chord_pitches is False.
    chord_is_default : np.ndarray
        An array of booleans indicating whether each chord uses its default pitches.
    key_labels : np.ndarray
        An array of key one-hot indexes, one per piece input.
    key_tonics : np.ndarray
        An array of key tonic pitches.
    key_modes : np.ndarray
        An array of key modes.
    """
    # Est chords
    chords, changes, all_pitches = state.get_chords()
    chord_labels = np.zeros(num_inputs, dtype=int)
    chord_roots = np.zeros(num_inputs, dtype=int)
    chord_types = np.zeros(num_inputs, dtype=object)
    chord_triads = np.zeros(num_inputs, dtype=object)
    chord_inversions = np.zeros(num_inputs, dtype=int)
    chord_pitches = np.full(num_inputs, None, dtype=object)
    chord_is_default = np.zeros(num_inputs, dtype=bool)
    for chord, pitches, start, end in zip(chords, all_pitches, changes[:-1], changes[1:]):
        root, chord_type, inv = hu.get_chord_from_one_hot_index(
            chord, input_root_type, use_inversions=use_inversions, reduction=reduction
        )
        root = hu.get_pitch_from_string(
            hu.get_pitch_string(root, input_root_type), output_root_type
        )
        chord = hu.get_chord_one_hot_index(
            chord_type,
            root,
            output_root_type,
            inversion=inv,
            use_inversion=use_inversions,
            reduction=reduction,
        )
        chord_labels[start:end] = chord
        chord_roots[start:end] = root
        chord_types[start:end] = chord_type
        chord_triads[start:end] = TRIAD_REDUCTION[chord_type]
        chord_inversions[start:end] = inv
        if chord_pitches_type == "str":
            chord_pitches[start:end] = str(tuple(sorted(pitches)))
        elif chord_pitches_type == "set":
            chord_pitches[start:end] = set(pitches)
        chord_is_default[start:end] = set(pitches) == hu.get_default_chord_pitches(
            root, chord_type, output_root_type
        )

    # Est keys
    keys, changes = state.get_keys()
    key_labels = np.zeros(num_inputs, dtype=int)
    key_tonics = np.zeros(num_inputs, dtype=int)
    key_modes = np.zeros(num_inputs, dtype=object)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        tonic, mode = hu.get_key_from_one_hot_index(key, input_tonic_type)
        tonic = hu.get_pitch_from_string(
            hu.get_pitch_string(tonic, input_tonic_type), output_tonic_type
        )
        key = hu.get_key_one_hot_index(mode, tonic, output_tonic_type)
        key_labels[start:end] = key
        key_tonics[start:end] = tonic
        key_modes[start:end] = mode

    return (
        chord_labels,
        chord_roots,
        chord_types,
        chord_triads,
        chord_inversions,
        chord_pitches,
        chord_is_default,
        key_labels,
        key_tonics,
        key_modes,
    )


def get_results_df(
    gt_piece: Piece,
    estimated: Union[State, Piece],
    input_root_type: PitchType,
    input_tonic_type: PitchType,
    output_root_type: PitchType,
    output_tonic_type: PitchType,
    use_inversions: bool,
    reduction: Dict[ChordType, ChordType],
) -> pd.DataFrame:
    """
    Evaluate the piece's estimated chords.

    Parameters
    ----------
    gt_piece : Piece
        The piece, containing the ground truth harmonic structure.
    estimated : Union[State, Piece]
        The state or piece, containing the estimated harmonic structure.
    input_root_type : PitchType
        The pitch type used for chord roots in the input piece and state.
    input_tonic_type : PitchType
        The pitch type used for key tonics in the input piece and state.
    output_root_type : PitchType
        The pitch type to use for chord roots in the output df.
    output_tonic_type : PitchType
        The pitch type to use for key tonics in the output df.
    use_inversions : bool
        Whether to use inversions in the chords.
    reduction : Dict[ChordType, ChordType]
        The chord type reduction to use.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame containing the results of the given state, with the given settings.
    """
    labels_list = []

    (
        gt_chord_labels,
        gt_chord_roots,
        gt_chord_types,
        gt_chord_triads,
        gt_chord_inversions,
        gt_chord_pitches,
        gt_chord_is_default,
        gt_key_labels,
        gt_key_tonics,
        gt_key_modes,
    ) = get_full_results_from_piece(
        gt_piece, output_root_type, output_tonic_type, use_inversions, reduction, "str"
    )

    (
        estimated_chord_labels,
        estimated_chord_roots,
        estimated_chord_types,
        estimated_chord_triads,
        estimated_chord_inversions,
        estimated_chord_pitches,
        estimated_chord_is_default,
        estimated_key_labels,
        estimated_key_tonics,
        estimated_key_modes,
    ) = (
        get_full_results_from_piece(
            estimated, output_root_type, output_tonic_type, use_inversions, reduction, "str"
        )
        if isinstance(estimated, Piece)
        else get_full_results_from_state(
            estimated,
            len(gt_piece.get_inputs()),
            input_root_type,
            input_tonic_type,
            output_root_type,
            output_tonic_type,
            use_inversions,
            reduction,
            "str",
        )
    )

    chord_label_list = hu.get_chord_label_list(output_root_type, use_inversions=True)
    key_label_list = hu.get_key_label_list(output_tonic_type)

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
        gt_pitches,
        gt_is_default,
        est_tonic,
        est_mode,
        est_root,
        est_chord_type,
        est_triad,
        est_inversion,
        est_pitches,
        est_is_default,
    ) in zip(
        gt_piece.get_duration_cache(),
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
        gt_chord_pitches,
        gt_chord_is_default,
        estimated_key_tonics,
        estimated_key_modes,
        estimated_chord_roots,
        estimated_chord_types,
        estimated_chord_triads,
        estimated_chord_inversions,
        estimated_chord_pitches,
        estimated_chord_is_default,
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
                "gt_pitches": gt_pitches,
                "gt_is_default": gt_is_default,
                "est_key": key_label_list[est_key_label],
                "est_tonic": est_tonic,
                "est_mode": est_mode,
                "est_chord": chord_label_list[est_chord_label],
                "est_root": est_root,
                "est_chord_type": est_chord_type,
                "est_triad": est_triad,
                "est_inversion": est_inversion,
                "est_pitches": est_pitches,
                "est_is_default": est_is_default,
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
    estimated: Union[State, Piece],
    gt_piece: Piece,
    root_type: PitchType,
    tonic_type: PitchType,
    use_inversions: bool,
    reduction: Dict[ChordType, ChordType],
    use_chord_pitches: bool = False,
) -> pd.DataFrame:
    """
    Get a df containing the labels of the given estimated output.

    Parameters
    ----------
    estimated : Union[State, Piece]
        The estimated harmony annotations.
    gt_piece : Piece
        The piece which was used as input when creating the given state.
    root_type : PitchType
        The pitch type to use for chord root labels.
    tonic_type : PitchType
        The pitch type to use for key tonic annotations.
    use_inversions : bool
        True if the state's chord indexes contain inversions. False otherwise.
    reduction : Dict[ChordType, ChordType]
        A mapping of the state's chord types to reduced chord types. This must be the
        reduction used by the CCM during the search process.
    use_chord_pitches : bool
        True to include altered tones in the output chord labels.

    Returns
    -------
    annotation_df : pd.DataFrame[type]
        A DataFrame containing the harmony annotations from the given state.
    """
    labels_list = []

    estimated_chord_labels, estimated_chord_pitches, estimated_key_labels = (
        get_labels_from_state(estimated, len(gt_piece.get_inputs()), use_chord_pitches)
        if isinstance(estimated, State)
        else get_labels_from_piece(
            estimated, root_type, tonic_type, use_inversions, reduction, use_chord_pitches
        )
    )

    chord_label_list = hu.get_chord_label_list(
        root_type, use_inversions=use_inversions, reduction=reduction
    )
    chord_list = hu.get_chord_from_one_hot_index(
        slice(None), root_type, use_inversions=use_inversions, reduction=reduction
    )
    key_label_list = hu.get_key_label_list(tonic_type)
    key_list = hu.get_key_from_one_hot_index(slice(None), tonic_type)

    prev_est_key_string = None
    prev_est_chord_string = None
    prev_est_chord_pitches = None

    for duration, note, est_chord_label, est_key_label, est_pitches in zip(
        gt_piece.get_duration_cache(),
        gt_piece.get_inputs(),
        estimated_chord_labels,
        estimated_key_labels,
        estimated_chord_pitches,
    ):
        if duration == 0:
            continue

        est_chord_string = chord_label_list[est_chord_label]
        est_key_string = key_label_list[est_key_label]

        # No change in labels
        if (
            est_chord_string == prev_est_chord_string
            and est_key_string == prev_est_key_string
            and est_pitches == prev_est_chord_pitches
        ):
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

        if est_chord_string != prev_est_chord_string or est_pitches != prev_est_chord_pitches:
            est_root, est_chord_type, _ = chord_list[est_chord_label]
            est_tonic, est_mode = key_list[est_key_label]

            chord_pitches_string = (
                hu.get_chord_pitches_string(
                    est_root, est_chord_type, est_pitches, est_tonic, est_mode, root_type
                )
                if use_chord_pitches
                else ""
            )

            labels_list.append(
                {
                    "label": est_chord_string + chord_pitches_string,
                    "mc": note.onset[0],
                    "mc_onset": note.mc_onset,
                    "mn_onset": note.onset[1],
                }
            )

        prev_est_key_string = est_key_string
        prev_est_chord_string = est_chord_string
        prev_est_chord_pitches = est_pitches

    return pd.DataFrame(labels_list)


def get_labels_from_state(
    state: State,
    num_inputs: int,
    use_chord_pitches: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get Lists of chord, pitches, and key labels from a State object. Each list will
    contain one label per piece input (the total length of which is given by num_inputs).

    Parameters
    ----------
    state : State
        The state chain whose labels we will return.
    num_inputs : int
        The number of inputs for the associated piece. Each returned array will be this length.
    use_chord_pitches : bool
        True to calculate chord pitches. False otherwise.

    Returns
    -------
    chord_labels : np.ndarray
        An array of chord one-hot indexes, one per piece input.
    chord_pitches : np.ndarray
        An array of sets of chord pitches for each chord. This will be an array of Nones
        if use_chord_pitches is False.
    key_labels : np.ndarray
        An array of key one-hot indexes, one per piece input.
    """
    # Chord info
    chords, changes, chord_pitches = state.get_chords()
    state_chord_labels = np.zeros(num_inputs, dtype=int)
    state_chord_pitches = np.full(num_inputs, None, dtype=object)
    for chord, pitches, start, end in zip(chords, chord_pitches, changes[:-1], changes[1:]):
        state_chord_labels[start:end] = chord
        if use_chord_pitches:
            for i in range(start, end):
                state_chord_pitches[i] = pitches

    # Key info
    keys, changes = state.get_keys()
    state_key_labels = np.zeros(num_inputs, dtype=int)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        state_key_labels[start:end] = key

    return state_chord_labels, state_chord_pitches, state_key_labels


def get_labels_from_piece(
    piece: Piece,
    root_type: PitchType,
    tonic_type: PitchType,
    use_inversions: bool,
    reduction: Dict[ChordType, ChordType],
    use_chord_pitches: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get Lists of chord, pitches, and key labels from a Piece object. Each list will
    contain one label per piece input.

    Parameters
    ----------
    piece : Piece
        The piece whose labels we will return.
    root_type : PitchType
        The pitch type to use to encode chord roots.
    tonic_type : PitchType
        The pitch type to use to encode key tonics.
    use_inversions : bool
        Whether to use inversions or not in the labels.
    reduction : Dict[ChordType, ChordType]
        A chord type reduction to use.
    use_chord_pitches : bool
        True to calculate chord pitches. False otherwise.

    Returns
    -------
    chord_labels : np.ndarray
        An array of chord one-hot indexes, one per piece input.
    chord_pitches : np.ndarray
        An array of sets of chord pitches for each chord. This will be an array of Nones
        if use_chord_pitches is False.
    key_labels : np.ndarray
        An array of key one-hot indexes, one per piece input.
    """
    chord_labels, _, _, _, _, chord_pitches, _, key_labels, _, _ = get_full_results_from_piece(
        piece,
        root_type,
        tonic_type,
        use_inversions,
        reduction,
        "set" if use_chord_pitches else None,
    )

    return chord_labels, chord_pitches, key_labels


def get_results_annotation_df(
    estimated: Union[State, Piece],
    gt_piece: Piece,
    root_type: PitchType,
    tonic_type: PitchType,
    use_inversions: bool,
    reduction: Dict[ChordType, ChordType],
    use_chord_pitches: bool = False,
) -> pd.DataFrame:
    """
    Get a df containing the full labels of the given estimated output, color-coded in terms
    of their accuracy according to the ground truth harmony in the given piece. This can be used
    to attach color-coded labels to a musescore3 score using ms3.

    Parameters
    ----------
    estimated : Union[State, Piece]
        A state or piece, containing the estimated harmonic structure.
    gt_piece : Piece
        The piece, containing the ground truth harmonic structure.
    root_type : PitchType
        The pitch type used for chord roots.
    tonic_type : PitchType
        The pitch type used for key tonics.
    use_inversions : bool
        True if the state's chord indexes contain inversions. False otherwise.
    reduction : Dict[ChordType, ChordType]
        A mapping of the state's chord types to reduced chord types. This must be the
        reduction used by the CCM during the search process.
    use_chord_pitches : bool
        True to include altered tones in the output chord labels.

    Returns
    -------
    label_df : pd.DataFrame
        A DataFrame containing the labels of the given state.
    """
    labels_list = []

    gt_chord_labels, gt_chord_pitches, gt_key_labels = get_labels_from_piece(
        gt_piece, root_type, tonic_type, use_inversions, reduction, use_chord_pitches
    )

    estimated_chord_labels, estimated_chord_pitches, estimated_key_labels = (
        get_labels_from_state(estimated, len(gt_piece.get_inputs()), use_chord_pitches)
        if isinstance(estimated, State)
        else get_labels_from_piece(
            estimated, root_type, tonic_type, use_inversions, reduction, use_chord_pitches
        )
    )

    chord_label_list = hu.get_chord_label_list(
        root_type, use_inversions=use_inversions, reduction=reduction
    )
    chord_list = hu.get_chord_from_one_hot_index(
        slice(None), root_type, use_inversions=use_inversions, reduction=reduction
    )
    key_label_list = hu.get_key_label_list(tonic_type)
    key_list = hu.get_key_from_one_hot_index(slice(None), tonic_type)

    prev_gt_chord_string = None
    prev_gt_key_string = None
    prev_gt_chord_pitches = None
    prev_est_key_string = None
    prev_est_chord_string = None
    prev_est_chord_pitches = None

    for (
        duration,
        note,
        est_chord_label,
        gt_chord_label,
        est_key_label,
        gt_key_label,
        est_pitches,
        gt_pitches,
    ) in zip(
        gt_piece.get_duration_cache(),
        gt_piece.get_inputs(),
        estimated_chord_labels,
        gt_chord_labels,
        estimated_key_labels,
        gt_key_labels,
        estimated_chord_pitches,
        gt_chord_pitches,
    ):
        if duration == 0:
            continue

        gt_chord_string = chord_label_list[gt_chord_label]
        gt_root, gt_chord_type, gt_inversion = chord_list[gt_chord_label]
        gt_key_string = key_label_list[gt_key_label]
        gt_tonic, gt_mode = key_list[gt_key_label]

        est_chord_string = chord_label_list[est_chord_label]
        est_root, est_chord_type, est_inversion = chord_list[est_chord_label]
        est_key_string = key_label_list[est_key_label]
        est_tonic, est_mode = key_list[est_key_label]

        # No change in labels
        if (
            gt_chord_string == prev_gt_chord_string
            and gt_key_string == prev_gt_key_string
            and est_chord_string == prev_est_chord_string
            and est_key_string == prev_est_key_string
            and gt_pitches == prev_gt_chord_pitches
            and est_pitches == prev_est_chord_pitches
        ):
            continue

        if gt_key_string != prev_gt_key_string or est_key_string != prev_est_key_string:
            if gt_tonic == est_tonic and gt_mode == est_mode:
                color = "green"
            elif gt_tonic == est_tonic:
                color = "orange"
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

        if (
            gt_chord_string != prev_gt_chord_string
            or est_chord_string != prev_est_chord_string
            or est_pitches != prev_est_chord_pitches
            or gt_pitches != prev_gt_chord_pitches
        ):
            chord_pitches_string = (
                hu.get_chord_pitches_string(
                    est_root, est_chord_type, est_pitches, est_tonic, est_mode, root_type
                )
                if use_chord_pitches
                else ""
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
                color = "orange"
            else:
                color = "red"

            labels_list.append(
                {
                    "label": est_chord_string + chord_pitches_string
                    if est_chord_string != prev_est_chord_string
                    or est_pitches != prev_est_chord_pitches
                    else "--",
                    "mc": note.onset[0],
                    "mc_onset": note.mc_onset,
                    "mn_onset": note.onset[1],
                    "color_name": color,
                }
            )

        prev_gt_key_string = gt_key_string
        prev_gt_chord_string = gt_chord_string
        prev_gt_chord_pitches = gt_pitches
        prev_est_key_string = est_key_string
        prev_est_chord_string = est_chord_string
        prev_est_chord_pitches = est_pitches

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


def duration_weighted_pitch_average(results_path: Union[Path, str]) -> Dict[str, float]:
    """
    Calculate duration-weighted average pitch accuracies across a log file.

    Parameters
    ----------
    results_path : Union[Path, str]
        A log file with pitch accuracies and durations for each piece.

    Returns
    -------
    accuracies : Dict[str, float]
        A duration-weighted pitch accuracy for both default and non-default pitches.
    """
    split_on = " = "

    tot_default_dur = 0
    tot_non_default_dur = 0

    default_dur = 0
    non_default_dur = 0

    tot_default_acc = 0
    tot_non_default_acc = 0

    with open(results_path, "r") as results_file:
        for line in results_file:
            if split_on not in line:
                continue

            line_split = line.split(split_on)
            if len(line_split) != 2:
                continue

            key, value = line_split
            key = key.strip()

            if "non-default" in key:
                if "duration" in key:
                    non_default_dur = Fraction(value)
                else:
                    tot_non_default_dur += non_default_dur
                    tot_non_default_acc += non_default_dur * float(value)
            elif "default" in key:
                if "duration" in key:
                    default_dur = Fraction(value)
                else:
                    tot_default_dur += default_dur
                    tot_default_acc += default_dur * float(value)

    return {
        "GT default pitch accuracy": tot_default_acc / tot_default_dur,
        "GT non-default pitch accuracy": tot_non_default_acc / tot_non_default_dur,
    }


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
