"""Data types and converters for model input and output formats."""
from enum import Enum


class PieceType(Enum):
    """
    The type of input data represented by a Piece. Either score, midi, or audio.
    """

    SCORE = 0
    MIDI = 1
    AUDIO = 2

    def __lt__(self, other):
        if not isinstance(other, PieceType):
            raise NotImplementedError()
        return self.value < other.value


class PitchType(Enum):
    """
    An Enum representing the way pitches (or tonics, or chord roots) are represented.
    Either TPC (tonal pitch class) or MIDI.
    """

    TPC = 0
    MIDI = 1

    def __lt__(self, other):
        if not isinstance(other, PitchType):
            raise NotImplementedError()
        return self.value < other.value


class KeyMode(Enum):
    """
    The modes that are supported for keys.
    """

    MAJOR = 0
    MINOR = 1

    def __lt__(self, other):
        if not isinstance(other, KeyMode):
            raise NotImplementedError()
        return self.value < other.value


class ChordType(Enum):
    """
    The types of chords that are supported.
    """

    MAJOR = 0
    MINOR = 1
    DIMINISHED = 2
    AUGMENTED = 3
    MAJ_MAJ7 = 4
    MAJ_MIN7 = 5
    MIN_MAJ7 = 6
    MIN_MIN7 = 7
    DIM7 = 8
    HALF_DIM7 = 9
    AUG_MIN7 = 10
    AUG_MAJ7 = 11

    def __lt__(self, other):
        if not isinstance(other, ChordType):
            raise NotImplementedError()
        return self.value < other.value


NO_REDUCTION = {chord_type: chord_type for chord_type in ChordType}


TRIAD_REDUCTION = {
    ChordType.MAJOR: ChordType.MAJOR,
    ChordType.MINOR: ChordType.MINOR,
    ChordType.DIMINISHED: ChordType.DIMINISHED,
    ChordType.AUGMENTED: ChordType.AUGMENTED,
    ChordType.MAJ_MAJ7: ChordType.MAJOR,
    ChordType.MAJ_MIN7: ChordType.MAJOR,
    ChordType.MIN_MAJ7: ChordType.MINOR,
    ChordType.MIN_MIN7: ChordType.MINOR,
    ChordType.DIM7: ChordType.DIMINISHED,
    ChordType.HALF_DIM7: ChordType.DIMINISHED,
    ChordType.AUG_MIN7: ChordType.AUGMENTED,
    ChordType.AUG_MAJ7: ChordType.AUGMENTED,
}


MAJOR_MINOR_REDUCTION = {
    ChordType.MAJOR: ChordType.MAJOR,
    ChordType.MINOR: ChordType.MINOR,
    ChordType.DIMINISHED: ChordType.MINOR,
    ChordType.AUGMENTED: ChordType.MAJOR,
    ChordType.MAJ_MAJ7: ChordType.MAJOR,
    ChordType.MAJ_MIN7: ChordType.MAJOR,
    ChordType.MIN_MAJ7: ChordType.MINOR,
    ChordType.MIN_MIN7: ChordType.MINOR,
    ChordType.DIM7: ChordType.MINOR,
    ChordType.HALF_DIM7: ChordType.MINOR,
    ChordType.AUG_MIN7: ChordType.MAJOR,
    ChordType.AUG_MAJ7: ChordType.MAJOR,
}


ALL_ONE_TYPE_REDUCTION = {
    ChordType.MAJOR: ChordType.MAJOR,
    ChordType.MINOR: ChordType.MAJOR,
    ChordType.DIMINISHED: ChordType.MAJOR,
    ChordType.AUGMENTED: ChordType.MAJOR,
    ChordType.MAJ_MAJ7: ChordType.MAJOR,
    ChordType.MAJ_MIN7: ChordType.MAJOR,
    ChordType.MIN_MAJ7: ChordType.MAJOR,
    ChordType.MIN_MIN7: ChordType.MAJOR,
    ChordType.DIM7: ChordType.MAJOR,
    ChordType.HALF_DIM7: ChordType.MAJOR,
    ChordType.AUG_MIN7: ChordType.MAJOR,
    ChordType.AUG_MAJ7: ChordType.MAJOR,
}
