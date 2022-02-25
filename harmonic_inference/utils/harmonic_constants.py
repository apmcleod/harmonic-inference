"""Pithc and chord constants."""
from harmonic_inference.data.data_types import ChordType, KeyMode, PitchType

TPC_NATURAL_PITCHES = 7
TPC_ACCIDENTALS = 5  # bb, b, natural, #, ##. natural must be in the exact middle
TPC_C_WITHIN_PITCHES = 1
TPC_C = int(TPC_ACCIDENTALS / 2) * TPC_NATURAL_PITCHES + TPC_C_WITHIN_PITCHES


C = {
    PitchType.MIDI: 0,
    PitchType.TPC: TPC_C,
}


STRING_TO_PITCH = {
    PitchType.TPC: {
        "A": TPC_C + 3,
        "B": TPC_C + 5,
        "C": TPC_C,
        "D": TPC_C + 2,
        "E": TPC_C + 4,
        "F": TPC_C - 1,
        "G": TPC_C + 1,
    },
    PitchType.MIDI: {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11},
}

NUM_PITCHES = {PitchType.TPC: TPC_NATURAL_PITCHES * TPC_ACCIDENTALS, PitchType.MIDI: 12}


PITCH_TO_STRING = {
    PitchType.MIDI: [
        "C",
        "C#/Db",
        "D",
        "D#/Eb",
        "E",
        "F",
        "F#/Gb",
        "G",
        "G#/Ab",
        "A",
        "A#/Bb",
        "B",
    ],
    PitchType.TPC: {index: string for string, index in STRING_TO_PITCH[PitchType.TPC].items()},
}


for note_string in ["A", "B", "C", "D", "E", "F", "G"]:
    for pitch_type in PitchType:
        STRING_TO_PITCH[pitch_type][note_string.lower()] = STRING_TO_PITCH[pitch_type][note_string]


SCALE_INTERVALS = {
    KeyMode.MAJOR: {
        PitchType.TPC: [0, 0, 2, 4, -1, 1, 3, 5],
        PitchType.MIDI: [0, 0, 2, 4, 5, 7, 9, 11],
    },
    KeyMode.MINOR: {
        PitchType.TPC: [0, 0, 2, -3, -1, 1, -4, -2],
        PitchType.MIDI: [0, 0, 2, 3, 5, 7, 8, 10],
    },
}


ACCIDENTAL_ADJUSTMENT = {PitchType.TPC: TPC_NATURAL_PITCHES, PitchType.MIDI: 1}


# How many semitones is one TPC
TPC_INTERVAL_SEMITONES = 7


SCALE_DEGREE_TO_NUMBER = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
}


NUMBER_TO_SCALE_DEGREE = {
    0: "I",
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
}

# Triad types indexes of ones for a C chord of the given type in a one-hot presence vector
CHORD_PITCHES = {}
for pitch_type in [PitchType.MIDI, PitchType.TPC]:
    CHORD_PITCHES[pitch_type] = {
        ChordType.MAJOR: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"],
            STRING_TO_PITCH[pitch_type]["G"],
        ],
        ChordType.MINOR: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"] - ACCIDENTAL_ADJUSTMENT[pitch_type],
            STRING_TO_PITCH[pitch_type]["G"],
        ],
        ChordType.DIMINISHED: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"] - ACCIDENTAL_ADJUSTMENT[pitch_type],
            STRING_TO_PITCH[pitch_type]["G"] - ACCIDENTAL_ADJUSTMENT[pitch_type],
        ],
        ChordType.AUGMENTED: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"],
            STRING_TO_PITCH[pitch_type]["G"] + ACCIDENTAL_ADJUSTMENT[pitch_type],
        ],
    }

    # Add major triad 7th chords
    for chord in [ChordType.MAJ_MAJ7, ChordType.MAJ_MIN7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][ChordType.MAJOR].copy()

    # Add minor triad 7th chords
    for chord in [ChordType.MIN_MAJ7, ChordType.MIN_MIN7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][ChordType.MINOR].copy()

    # Add diminished triad 7th chords
    for chord in [ChordType.DIM7, ChordType.HALF_DIM7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][ChordType.DIMINISHED].copy()

    # Add augmented triad 7th chords
    for chord in [ChordType.AUG_MAJ7, ChordType.AUG_MIN7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][ChordType.AUGMENTED].copy()

    # Add major 7ths
    for chord in [ChordType.MAJ_MAJ7, ChordType.MIN_MAJ7, ChordType.AUG_MAJ7]:
        CHORD_PITCHES[pitch_type][chord].append(STRING_TO_PITCH[pitch_type]["B"])

    # Add minor 7ths
    for chord in [ChordType.MAJ_MIN7, ChordType.MIN_MIN7, ChordType.HALF_DIM7, ChordType.AUG_MIN7]:
        CHORD_PITCHES[pitch_type][chord].append(
            STRING_TO_PITCH[pitch_type]["B"] - ACCIDENTAL_ADJUSTMENT[pitch_type]
        )

    # Add diminished 7ths
    for chord in [ChordType.DIM7]:
        CHORD_PITCHES[pitch_type][chord].append(
            STRING_TO_PITCH[pitch_type]["B"] - 2 * ACCIDENTAL_ADJUSTMENT[pitch_type]
        )


STRING_TO_CHORD_TYPE = {
    "M": ChordType.MAJOR,
    "m": ChordType.MINOR,
    "o": ChordType.DIMINISHED,
    "+": ChordType.AUGMENTED,
    "MM7": ChordType.MAJ_MAJ7,
    "Mm7": ChordType.MAJ_MIN7,
    "mM7": ChordType.MIN_MAJ7,
    "mm7": ChordType.MIN_MIN7,
    "o7": ChordType.DIM7,
    "%7": ChordType.HALF_DIM7,
    "+7": ChordType.AUG_MIN7,
    "+M7": ChordType.AUG_MAJ7,
}


CHORD_TYPE_TO_STRING = {chord_type: string for string, chord_type in STRING_TO_CHORD_TYPE.items()}

STRING_TO_CHORD_TYPE["It"] = ChordType.DIMINISHED
STRING_TO_CHORD_TYPE["Ger"] = ChordType.DIM7
STRING_TO_CHORD_TYPE["Fr"] = ChordType.MAJ_MIN7


FIGBASS_INVERSIONS = {"7": 0, "6": 1, "65": 1, "43": 2, "64": 2, "2": 3}

CHORD_INVERSION_COUNT = {
    ChordType.MAJOR: 3,
    ChordType.MINOR: 3,
    ChordType.DIMINISHED: 3,
    ChordType.AUGMENTED: 3,
    ChordType.MAJ_MAJ7: 4,
    ChordType.MAJ_MIN7: 4,
    ChordType.MIN_MAJ7: 4,
    ChordType.MIN_MIN7: 4,
    ChordType.DIM7: 4,
    ChordType.HALF_DIM7: 4,
    ChordType.AUG_MIN7: 4,
    ChordType.AUG_MAJ7: 4,
}

# Chord relative pitches
MIN_RELATIVE_TPC = -14  # Inclusive
MAX_RELATIVE_TPC = 15  # Exclusive
# Extra space to add because chord is sometimes relative to different key
RELATIVE_TPC_EXTRA = 5

# Key change relative pitches
MIN_KEY_CHANGE_INTERVAL_TPC = -14  # Inclusive
MAX_KEY_CHANGE_INTERVAL_TPC = 15  # Exclusive

# Chord pitch vector
MAX_CHORD_PITCH_INTERVAL_TPC = 13

# Diatonic chords
DIATONIC_CHORDS = {
    PitchType.TPC: {
        KeyMode.MAJOR: {
            0: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            2: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            4: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            -1: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            1: set([ChordType.MAJOR, ChordType.MAJ_MIN7]),
            3: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            5: set([ChordType.DIMINISHED, ChordType.HALF_DIM7]),
        },
        KeyMode.MINOR: {
            0: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            2: set([ChordType.DIMINISHED, ChordType.HALF_DIM7]),
            -3: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            -1: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            1: set([ChordType.MAJOR, ChordType.MAJ_MIN7, ChordType.MINOR, ChordType.MIN_MIN7]),
            -4: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            -2: set([ChordType.MAJOR, ChordType.MAJ_MIN7]),  # VII
            5: set([ChordType.DIMINISHED, ChordType.DIM7]),  # vii
        },
    },
    PitchType.MIDI: {
        KeyMode.MAJOR: {
            0: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            2: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            4: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            5: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            7: set([ChordType.MAJOR, ChordType.MAJ_MIN7]),
            9: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            11: set([ChordType.DIMINISHED, ChordType.HALF_DIM7]),
        },
        KeyMode.MINOR: {
            0: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            2: set([ChordType.DIMINISHED, ChordType.HALF_DIM7]),
            3: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            5: set([ChordType.MINOR, ChordType.MIN_MIN7]),
            7: set([ChordType.MAJOR, ChordType.MAJ_MIN7, ChordType.MINOR, ChordType.MIN_MIN7]),
            8: set([ChordType.MAJOR, ChordType.MAJ_MAJ7]),
            10: set([ChordType.MAJOR, ChordType.MAJ_MIN7]),  # VII
            11: set([ChordType.DIMINISHED, ChordType.DIM7]),  # #vii
        },
    },
}
