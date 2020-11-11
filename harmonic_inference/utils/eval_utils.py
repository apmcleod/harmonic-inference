"""Utility functions for evaluating model outputs."""
import logging
from typing import Dict

import numpy as np

import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.data_types import NO_REDUCTION, ChordType, KeyMode, PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import State


def evaluate_chords(
    piece: Piece,
    state: State,
    pitch_type: PitchType,
    use_inversion: bool = True,
    reduction: Dict[ChordType, ChordType] = NO_REDUCTION,
) -> float:
    """
    Evaluate the piece's estimated chords.

    Parameters
    ----------
    piece : Piece
        The piece, containing the ground truth harmonic structure.
    state : State
        The state, containing the estimated harmonic structure.
    pitch_type : PitchType
        The pitch type used for chord roots.
    use_inversion : bool
        True to use inversion when checking the chord type. False to ignore inversion.
    reduction : Dict[ChordType, ChordType]
        A reduction to reduce chord types to another type.

    Returns
    -------
    accuracy : float
        The average accuracy of the state's chord estimates for the full duration of
        the piece.
    """
    gt_chords = piece.get_chords()
    gt_changes = piece.get_chord_change_indices()
    gt_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(gt_chords, gt_changes, gt_changes[1:]):
        gt_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )
    gt_labels[gt_changes[-1] :] = gt_chords[-1].get_one_hot_index(
        relative=False, use_inversion=True, pad=False
    )

    chords, changes = state.get_chords()
    estimated_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for chord, start, end in zip(chords, changes[:-1], changes[1:]):
        estimated_labels[start:end] = chord

    accuracy = 0.0
    for duration, est_label, gt_label in zip(
        piece.get_duration_cache(),
        estimated_labels,
        gt_labels,
    ):
        if duration == 0:
            continue

        gt_root, gt_chord_type, gt_inversion = hu.get_chord_from_one_hot_index(
            gt_label, pitch_type, use_inversions=True
        )

        est_root, est_chord_type, est_inversion = hu.get_chord_from_one_hot_index(
            est_label, pitch_type, use_inversions=True
        )

        distance = get_chord_distance(
            gt_root,
            gt_chord_type,
            gt_inversion,
            est_root,
            est_chord_type,
            est_inversion,
            use_inversion=use_inversion,
            reduction=reduction,
        )
        accuracy += (1.0 - distance) * duration

    return accuracy / np.sum(piece.get_duration_cache())


def get_chord_distance(
    gt_root: int,
    gt_chord_type: ChordType,
    gt_inversion: int,
    est_root: int,
    est_chord_type: ChordType,
    est_inversion: int,
    use_inversion: bool = True,
    reduction: Dict[ChordType, ChordType] = NO_REDUCTION,
) -> float:
    """
    Get the distance from a ground truth chord to an estimated chord.

    Parameters
    ----------
    gt_root : int
        The root pitch of the ground truth chord.
    gt_chord_type : ChordType
        The chord type of the ground truth chord.
    gt_inversion : int
        The inversion of the ground truth chord.
    est_root : int
        The root pitch of the estimated chord.
    est_chord_type : ChordType
        The chord type of the estimated chord.
    est_inversion : int
        The inversion of the estimated chord.
    use_inversion : bool
        True to use inversion when checking the chord type. False to ignore inversion.
    reduction : Dict[ChordType, ChordType]
        A reduction to reduce chord types to another type.

    Returns
    -------
    distance : float
        A distance between 0 (completely correct), and 1 (completely incorrect).
    """
    gt_chord_type = reduction[gt_chord_type]
    est_chord_type = reduction[est_chord_type]

    if not use_inversion:
        gt_inversion = 0
        est_inversion = 0

    if gt_root == est_root and gt_chord_type == est_chord_type and gt_inversion == est_inversion:
        return 0.0

    return 1.0


def evaluate_keys(
    piece: Piece,
    state: State,
    pitch_type: PitchType,
    tonic_only: bool = False,
) -> float:
    """
    Evaluate the piece's estimated keys.

    Parameters
    ----------
    piece : Piece
        The piece, containing the ground truth harmonic structure.
    state : State
        The state, containing the estimated harmonic structure.
    pitch_type : PitchType
        The pitch type used for key tonics.
    tonic_only : bool
        True to only evaluate the tonic pitch. False to also take mode into account.

    Returns
    -------
    accuracy : float
        The average accuracy of the state's key estimates for the full duration of
        the piece.
    """
    gt_keys = piece.get_keys()
    gt_changes = piece.get_key_change_input_indices()
    gt_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(gt_keys, gt_changes, gt_changes[1:]):
        gt_labels[start:end] = key.get_one_hot_index()
    gt_labels[gt_changes[-1] :] = gt_keys[-1].get_one_hot_index()

    keys, changes = state.get_keys()
    estimated_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(keys, changes[:-1], changes[1:]):
        estimated_labels[start:end] = key

    accuracy = 0.0
    for duration, est_label, gt_label in zip(
        piece.get_duration_cache(),
        estimated_labels,
        gt_labels,
    ):
        if duration == 0:
            continue

        gt_tonic, gt_mode = hu.get_key_from_one_hot_index(int(gt_label), pitch_type)
        est_tonic, est_mode = hu.get_key_from_one_hot_index(int(est_label), pitch_type)

        distance = get_key_distance(
            gt_tonic,
            gt_mode,
            est_tonic,
            est_mode,
            tonic_only=tonic_only,
        )
        accuracy += (1.0 - distance) * duration

    return accuracy / np.sum(piece.get_duration_cache())


def get_key_distance(
    gt_tonic: int,
    gt_mode: KeyMode,
    est_tonic: int,
    est_mode: KeyMode,
    tonic_only: bool = False,
) -> float:
    """
    Get the distance from one key to another.

    Parameters
    ----------
    gt_tonic : int
        The tonic pitch of the ground truth key.
    gt_mode : KeyMode
        The mode of the ground truth key.
    est_tonic : int
        The tonic pitch of the estimated key.
    est_mode : KeyMode
        The mode of the estimated key.
    tonic_only : bool
        True to only evaluate the tonic pitch. False to also take mode into account.

    Returns
    -------
    distance : float
        The distance between the estimated and ground truth keys.
    """
    if tonic_only:
        return 0.0 if gt_tonic == est_tonic else 1.0

    return 0.0 if gt_tonic == est_tonic and gt_mode == est_mode else 1.0


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

        logging.info("%s - %s (duration %s):", input_start, input_end, duration)
        logging.info("    Estimated structure: %s\t%s", est_key_label, est_chord_label)
        if gt_key_label != est_key_label or gt_chord_label != est_chord_label:
            logging.info("      Correct structure: %s\t%s", gt_key_label, gt_chord_label)
