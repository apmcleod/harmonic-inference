"""Data types for model input and output formats."""
from enum import Enum


class PitchType(Enum):
    """
    An Enum representing the way pitches (or tonics, or chord roots) are represented.
    Either TPC (tonal pitch class) or MIDI, and absolute or relative.
    """
    TPC_ABSOLUTE = 0
    TPC_RELATIVE = 1
    MIDI_ABSOLUTE = 2
    MIDI_RELATIVE = 3
