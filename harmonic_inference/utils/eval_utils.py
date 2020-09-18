"""Utility functions for evaluating model outputs."""
import numpy as np

from harmonic_inference.data.piece import Piece
from harmonic_inference.models.joint_model import State

def evaluate(piece: Piece, state: State):
    """
    Evaluate the piece's estimated harmonic structure.

    Parameters
    ----------
    piece : Piece
        The piece, containing the ground truth harmonic structure.
    state : State
        The state, containing the estimated harmonic structure.
    """
    gt_chords = piece.get_chords()
    gt_changes = piece.get_chord_change_indices()
    gt_labels = np.zeros(len(piece.get_inputs()))
    for chord, start, end in zip(gt_chords, gt_changes, gt_changes[1:]):
        gt_labels[start:end] = chord.get_one_hot_index()
    gt_labels[gt_changes[-1]:] = gt_chords[-1].get_one_hot_index()

    chords, changes = state.get_chords()
    estimated_labels = np.zeros(len(piece.get_inputs()))
    for chord, start, end in zip(chords, changes[:-1], changes[1:]):
        estimated_labels[start:end] = chord

    accuracy = 0.0
    for duration, estimated_label, gt_label in zip(
        piece.get_duration_cache(),
        estimated_labels,
        gt_labels,
    ):
        accuracy += (1.0 - get_distance(gt_label, estimated_label)) * duration

    return accuracy / np.sum(piece.get_duration_cache())


def get_distance(target: int, estimate: int):
    """
    Get the distance between two chord symbols.

    Parameters
    ----------
    target : int
        [description]
    estimate : int
        [description]

    Returns
    -------
    distance : float
        A distance between 0 (completely correct), and 1 (completely incorrect).
    """
    return 0.0 if target == estimate else 1.0
