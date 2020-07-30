"""Data types and converters for model input and output formats."""
from enum import Enum


class PieceType(Enum):
    """
    The type of input data represented by a Piece. Either score, midi, or audio.
    """
    SCORE = 0
    MIDI = 1
    AUDIO = 2


class PitchType(Enum):
    """
    An Enum representing the way pitches (or tonics, or chord roots) are represented.
    Either TPC (tonal pitch class) or MIDI.
    """
    TPC = 0
    MIDI = 1


class RelativeType(Enum):
    """
    Whether a given type is absolute or relative.
    """
    ABSOLUTE = 0
    RELATIVE = 1
