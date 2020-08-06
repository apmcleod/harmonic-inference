"""A class storing a musical piece from score, midi, or audio format."""
from typing import Union, Tuple
from fractions import Fraction
from .data_types import *


position = Union[float, Tuple[int, Fraction]]

class Note():
    """
    A representation of a musical Note, with pitch, onset, duration, and offset.
    """
    def __init__(self, pitch: int, octave: int, onset: position, duration: position,
                 offset: position, pitch_type: PitchType):
        self.pitch = pitch
        self.octave = octave
        self.onset = onset
        self.duration = duration
        self.offset = offset

    @staticmethod
    def from_series(note_row: pd.Series, measures_df: pd.DataFrame, pitch_type: PitchType) -> Note:
        pass


class Chord():
    """
    A musical chord, with a root and base note.
    """
    def __init__(self, root: int, bass: int, quality: ChordQuality, inversion: int,
                 pitch_type: PitchType):
        self.root = root
        self.bass = bass
        self.quality = quality

    @staticmethod
    def from_series(chord_row: pd.Series, measures_df: pd.DataFrame,
                    pitch_type: PitchType) -> Chord:
        pass


class Key():
    """
    A musical key, with tonic and model
    """
    def __init__(self, tonic: int, mode: KeyMode, tonic_type: PitchType):
        self.tonic = tonic
        self.mode = mode

    @staticmethod
    def from_series(chord_row: pd.Series, tonic_type: PitchType) -> Key:
        pass


class Piece():
    """
    A single musical piece, which can be from score, midi, or audio.
    """
    def __init__(self, data_type: PieceType):
        # pylint: disable=invalid-name
        self.DATA_TYPE = data_type

    def get_inputs(self):
        raise NotImplementedError

    def get_chord_change_indices(self):
        raise NotImplementedError

    def get_chords(self):
        raise NotImplementedError

    def get_key_change_indices(self):
        raise NotImplementedError

    def get_keys(self):
        raise NotImplementedError


class ScorePiece(Piece):
    """
    A single musical piece, in score format.
    """
    def __init__(self, notes_df, chords_df, measures_df):
        super().__init__(PieceType.SCORE)
        self.notes = [Note.from_series(note, measures_df, PitchType.TPC) for _, note in notes_df.iterrows()]
        self.chords = [Chord.from_series(chord, measures_df, PitchType.TPC) for _, chord in chords_df.iterrows()]
        # TODO: Get keys
        self.keys = []

    def get_inputs(self):
        return self.notes

    def get_chord_change_indices(self):
        raise NotImplementedError

    def get_chords(self):
        return self.chords

    def get_key_change_indices(self):
        raise NotImplementedError

    def get_keys(self):
        return self.keys
