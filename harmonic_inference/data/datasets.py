from typing import List

import numpy as np
from torch.utils.data import Dataset

from harmonic_inference.data.piece import Piece
import harmonic_inference.utils.harmonic_utils as hu


class HarmonicDataset(Dataset):
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            "inputs": self.inputs[item],
            "targets": self.targets[item],
        }


class ChordTransitionDataset(HarmonicDataset):
    def __init__(self, pieces: List[Piece]):
        self.targets = [piece.get_chord_change_indices() for piece in pieces]
        self.inputs = [
            np.vstack([note.to_vec() for note in piece.get_inputs()]) for piece in pieces
        ]


class ChordClassificationDataset(HarmonicDataset):
    def __init__(self, pieces: List[Piece]):
        self.targets = [
            hu.get_chord_one_hot_index(
                chord.chord_type,
                chord.root,
                chord.pitch_type,
                inversion=chord.inversion,
                use_inversion=True,
            )
            for piece in pieces
            for chord in piece.get_chords()
        ]
        self.inputs = [piece.get_chord_note_inputs(window=2) for piece in pieces]


class ChordSequenceDataset(HarmonicDataset):
    pass


class KeyTransitionDataset(HarmonicDataset):
    pass


class KeySequenceDataset(HarmonicDataset):
    pass
