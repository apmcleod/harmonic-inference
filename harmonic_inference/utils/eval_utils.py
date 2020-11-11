"""Utility functions for evaluating model outputs."""
from typing import Dict

import numpy as np

import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.data_types import NO_REDUCTION, ChordType, PitchType
from harmonic_inference.data.piece import Chord, Piece
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
    gt_labels = np.zeros(len(piece.get_inputs()))
    for chord, start, end in zip(gt_chords, gt_changes, gt_changes[1:]):
        gt_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )
    gt_labels[gt_changes[-1] :] = gt_chords[-1].get_one_hot_index(
        relative=False, use_inversion=True, pad=False
    )

    chords, changes = state.get_chords()
    estimated_labels = np.zeros(len(piece.get_inputs()))
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

        root, chord_type, inversion = hu.get_chord_from_one_hot_index(
            int(gt_label), pitch_type, use_inversions=True
        )
        gt_chord = Chord(
            root,
            hu.get_bass_note(chord_type, root, inversion, pitch_type),
            None,
            None,
            chord_type,
            inversion,
            None,
            None,
            None,
            None,
            None,
            pitch_type,
        )

        root, chord_type, inversion = hu.get_chord_from_one_hot_index(
            int(est_label), pitch_type, use_inversions=True
        )
        est_chord = Chord(
            root,
            hu.get_bass_note(chord_type, root, inversion, pitch_type),
            None,
            None,
            chord_type,
            inversion,
            None,
            None,
            None,
            None,
            None,
            pitch_type,
        )

        distance = get_distance(
            gt_chord,
            est_chord,
            use_inversion=use_inversion,
            reduction=reduction,
        )
        accuracy += (1.0 - distance) * duration

    return accuracy / np.sum(piece.get_duration_cache())


def get_distance(
    target: Chord,
    estimate: Chord,
    use_inversion: bool = True,
    reduction: Dict[ChordType, ChordType] = NO_REDUCTION,
) -> float:
    """
    Get the distance between a target and estimated chord.

    Parameters
    ----------
    target : Chord
        The target chord.
    estimate : Chord
        The estimated chord.
    use_inversion : bool
        True to use inversion when checking the chord type. False to ignore inversion.
    reduction : Dict[ChordType, ChordType]
        A reduction to reduce chord types to another type.

    Returns
    -------
    distance : float
        A distance between 0 (completely correct), and 1 (completely incorrect).
    """
    for chord in [target, estimate]:
        chord.chord_type = reduction[chord.chord_type]

    return 0.0 if target.is_repeated(estimate, use_inversion=use_inversion) else 1.0


def log_state(state: State, piece: Piece):
    """
    [summary]

    Parameters
    ----------
    state : State
        [description]
    piece : Piece
        [description]
    """
    pass
