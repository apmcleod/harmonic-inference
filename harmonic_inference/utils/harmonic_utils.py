"""Utility functions for getting harmonic and pitch information from the corpus DataFrames."""
from typing import List
import pandas as pd
import numpy as np

from harmonic_inference.data.data_types import  KeyMode, PitchType, ChordType


TPC_C = 15


STRING_TO_PITCH = {
    PitchType.TPC: {
        'A': TPC_C + 3,
        'B': TPC_C + 5,
        'C': TPC_C,
        'D': TPC_C + 2,
        'E': TPC_C + 4,
        'F': TPC_C - 1,
        'G': TPC_C + 1
    },
    PitchType.MIDI: {
        'C': 0,
        'D': 2,
        'E': 4,
        'F': 5,
        'G': 7,
        'A': 9,
        'B': 11
    }
}

for note_string in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    for pitch_type in PitchType:
        STRING_TO_PITCH[pitch_type][note_string.lower()] = STRING_TO_PITCH[pitch_type][note_string]


SCALE_INTERVALS = {
    KeyMode.MAJOR: {
        PitchType.TPC: [0, 0, 2, 4, -1, 1, 3, 5],
        PitchType.MIDI: [0, 0, 2, 4, 5, 7, 9, 11],
    },
    KeyMode.MINOR: {
        PitchType.TPC: [0, 0, 2, -3, -1, 1, -4, -2],
        PitchType.MIDI: [0, 0, 2, 3, 5, 7, 8, 10]
    }
}


ACCIDENTAL_ADJUSTMENT = {
    PitchType.TPC: 7,
    PitchType.MIDI: 1
}


NUM_PITCHES = {
    PitchType.TPC: 35,
    PitchType.MIDI: 12
}


SCALE_DEGREE_TO_NUMBER = {
    'I':   1,
    'II':  2,
    'III': 3,
    'IV':  4,
    'V':   5,
    'VI':  6,
    'VII': 7,
    'i':   1,
    'ii':  2,
    'iii': 3,
    'iv':  4,
    'v':   5,
    'vi':  6,
    'vii': 7,
    '1':   1,
    '2':   2,
    '3':   3,
    '4':   4,
    '5':   5,
    '6':   6,
    '7':   7
}

# Triad types indexes of ones for a C chord of the given type in a one-hot presence vector
CHORD_PITCHES = {}
for pitch_type in [PitchType.MIDI, PitchType.TPC]:
    CHORD_PITCHES[pitch_type] = {
        ChordType.MAJOR: [
            STRING_TO_PITCH[pitch_type]['C'],
            STRING_TO_PITCH[pitch_type]['E'],
            STRING_TO_PITCH[pitch_type]['G']
        ],
        ChordType.MINOR: [
            STRING_TO_PITCH[pitch_type]['C'],
            STRING_TO_PITCH[pitch_type]['E'] - ACCIDENTAL_ADJUSTMENT[pitch_type],
            STRING_TO_PITCH[pitch_type]['G']
        ],
        ChordType.DIMINISHED: [
            STRING_TO_PITCH[pitch_type]['C'],
            STRING_TO_PITCH[pitch_type]['E'] - ACCIDENTAL_ADJUSTMENT[pitch_type],
            STRING_TO_PITCH[pitch_type]['G'] - ACCIDENTAL_ADJUSTMENT[pitch_type]
        ],
        ChordType.AUGMENTED: [
            STRING_TO_PITCH[pitch_type]['C'],
            STRING_TO_PITCH[pitch_type]['E'],
            STRING_TO_PITCH[pitch_type]['G'] + ACCIDENTAL_ADJUSTMENT[pitch_type]
        ]
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
        CHORD_PITCHES[pitch_type][chord].append(STRING_TO_PITCH[pitch_type]['B'])

    # Add minor 7ths
    for chord in [ChordType.MAJ_MIN7, ChordType.MIN_MIN7, ChordType.HALF_DIM7, ChordType.AUG_MIN7]:
        CHORD_PITCHES[pitch_type][chord].append(
            STRING_TO_PITCH[pitch_type]['B'] - ACCIDENTAL_ADJUSTMENT[pitch_type]
        )

    # Add diminished 7ths
    for chord in [ChordType.DIMINISHED]:
        CHORD_PITCHES[pitch_type][chord].append(
            STRING_TO_PITCH[pitch_type]['B'] - 2 * ACCIDENTAL_ADJUSTMENT[pitch_type]
        )




def get_one_hot_labels(pitch_type: PitchType) -> List[str]:
    """
    Get the human-readable label of every one-hot chord value.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type representation being used.

    Returns
    -------
    labels : List[String]
        A List, where labels[0] is the String interpretation of the one-hot chord label 0, etc.
    """
    roots = []
    if pitch_type == PitchType.MIDI:
        roots = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    elif pitch_type == PitchType.TPC:
        for accidental in ['bb', 'b', '', '#', '##']:
            for root in ['F', 'C', 'G', 'D', 'A', 'E', 'B']:
                roots.append(f'{root}{accidental}')

    labels = []
    for chord_type in ChordType:
        for root in roots:
            labels.append(f'{root}:{chord_type}')

    return labels


def get_accidental_adjustment(string: str, in_front: bool = True) -> (int, str):
    """
    Get the accidental adjustment of the accidentals at the beginning of a string.

    Parameters
    ----------
    string : string
        The string whose accidentals we want. It should begin with some number of either 'b'
        or '#', and then anything else.

    in_front : boolean
        True if the accidentals come at the beginning of the string. False for the end.

    Returns
    -------
    adjustment : int
        -1 for each 'b' at the beginning of the string. +1 for each '#'.

    new_string : string
        The given string without the accidentals.
    """
    adjustment = 0

    if in_front:
        while string[0] == 'b' and len(string) > 1:
            string = string[1:]
            adjustment -= 1

        while string[0] == '#':
            string = string[1:]
            adjustment += 1

    else:
        while string[-1] == 'b' and len(string) > 1:
            string = string[:-1]
            adjustment -= 1

        while string[-1] == '#':
            string = string[:-1]
            adjustment += 1

    return adjustment, string



def transpose_chord_vector(chord_vector: List[int], interval: int,
                           pitch_type: PitchType) -> List[int]:
    """
    Transpose a chord vector by a given interval.

    Parameters
    ----------
    chord_vector : List[int]
        A binary vector representation of the given chord type, where 1 indicates
        the presence of a pitch in the given chord type, and 0 represents non-presence.
        The vector is length 12.

    interval : int
        The interval to transpose the given chord vector by, either in semitones (if pitch_type
        is MIDI) or in fifths (if pitch_type is TPC).

    pitch_type : int
        The pitch type representation to use. If MIDI, the transposition will roll the given
        chord vector. Otherwise, it will only shift values (but if any ones move to an index
        < 0 or >= NUM_PITCHES[TPC], they are removed from the returned vector.

    Returns
    -------
    transposed_chord_vector : list(int)
        The input vector, with each chord tone transposed by the given amount.
    """
    # For MIDI, roll the vector
    if pitch_type == PitchType.MIDI:
        return np.roll(np.array(chord_vector), interval)

    # For TPC, it is a shift without wraparound
    result = np.empty_like(chord_vector)
    if interval > 0:
        result[:interval] = 0
        result[interval:] = chord_vector[:-interval]
    elif num < 0:
        result[interval:] = 0
        result[:interval] = chord_vector[-interval:]
    else:
        result[:] = chord_vector
    return result


def get_vector_from_chord_type(chord_type: ChordType, pitch_type: PitchType) -> List[int]:
    """
    Convert a chord type string into a one-hot vector representation of pitch presence.

    Parameters
    ----------
    chord_type : ChordType
        The type of chord whose pitch vector to return.
    pitch_type : PitchType
        The type of pitch vector to return.

    Returns
    -------
    chord_vector : List[int]
        A one-hot vector of the pitches present in a chord of the given chord type
        with root C in the given pitch representation.
    """
    chord_vector = np.zeros(NUM_PITCHES[pitch_type])
    chord_vector[CHORD_PITCHES[pitch_type][chord_type]] = 1
    return chord_vector


def get_interval_from_numeral(numeral: str, mode: KeyMode, pitch_type: PitchType) -> int:
    """
    Get the interval from the key tonic to the given scale degree numeral.

    Parameters
    ----------
    numeral : str
        Usually an upper or lowercase roman numeral, prepended by accidentals. Can also be
        Ger, It, or Fr, for augmented 6th chord symbols.
    mode : KeyMode
        The mode of the key from which to measure the scale degree.
    pitch_type : PitchType
        The type of interval to return. Either TPC (to return circle of fifths difference) or MIDI
        (for semitones).

    Returns
    -------
    interval : int
        The interval from the key tonic to the scale degree numeral.
    """
    if numeral in ['Ger', 'It', 'Fr']:
        # TODO: Special processing
        return 0
    return get_interval_from_scale_degree(numeral, True, True, mode, pitch_type)


def get_interval_from_bass_step(bass_step: str, mode: KeyMode, pitch_type: PitchType) -> int:
    """
    Get the interval from the tonic of the current key to the given bass step.

    Parameters
    ----------
    bass_step : str
        The bass step. Usually a number pre-pended with accidentals. Can also be Error or Unclear.
    mode : KeyMode
        The mode of the key from which to measure the scale degree.
    pitch_type : PitchType
        The type of interval to return. Either TPC (to return circle of fifths difference) or MIDI
        (for semitones).

    Returns
    -------
    interval : int
        The interval from the key tonic to the bass note.
    """
    if bass_step in ['Error', 'Unclear']:
        # TODO: Special processing
        return 0
    return get_interval_from_scale_degree(bass_step, False, True, mode, pitch_type)


def get_interval_from_scale_degree(scale_degree: str, numeral: bool, accidentals_prefixed: bool,
                                   mode: KeyMode, pitch_type: PitchType) -> int:
    """
    Get an interval from the string of a scale degree, prefixed or suffixed with flats or sharps.

    Parameters
    ----------
    scale_degree : str
        A string of a scale degree, either as a roman numeral or a number, with any accidentals
        suffixed or prefixed as 'b' (flats) or '#' (sharps).
    numeral : bool
        True if the scale degree is given as a roman numeral. False if it is given as a number.
    accidentals_prefixed : bool
        True if the given scale degree might be prefixed with accidentals. False if they can be
        suffixed.
    mode : KeyMode
        The mode of the key from which to measure the scale degree.
    pitch_type : PitchType
        The type of interval to return. Either TPC (to return circle of fifths difference) or MIDI
        (for semitones).

    Returns
    -------
    interval : int
        The interval above the root note the given numeral lies.
    """
    accidental_adjustment, scale_degree = get_accidental_adjustment(scale_degree,
                                                                    in_front=accidentals_prefixed)

    interval = SCALE_INTERVALS[mode][pitch_type][SCALE_DEGREE_TO_NUMBER[scale_degree]]
    interval += accidental_adjustment * ACCIDENTAL_ADJUSTMENT[pitch_type]

    return interval


def transpose_pitch(pitch: int, interval: int, pitch_type: PitchType) -> int:
    """
    Transpose the given pitch by the given interval, and then mod the result to ensure it is in
    the valid range of the given pitch type.

    Parameters
    ----------
    pitch : int
        The original pitch.
    interval : int
        The amount to transpose the given pitch by.
    pitch_type : PitchType
        The pitch type. If MIDI, the returned pitch will be on the range [0, 12), and the given
        interval is interpreted as semitones. If TPC, the returned pitch will be on the range
        [0, 35), and the given interval is interpreted as fifths.

    Returns
    -------
    pitch : int
        The given pitch, transposed by the given interval.
    """
    return (pitch + interval) % NUM_PITCHES[pitch_type]


def get_chord_type(is_major: bool, form: str, figbass: str) -> ChordType:
    """
    Get the chord type, given some features about the chord.

    Parameters
    ----------
    is_major : boolean
        True if the basic chord (triad) is major. False otherwise. Ignored if
        the triad is diminished or augmented (in which case, form disambiguates).

    form : string
        A string representing the form of the chord:
            'o':  Diminished 7th or triad
            '%':  Half-diminished 7th chord
            '+':  Augmented seventh or triad
            '+M': Augmented major 7th chord
            'M':  7th chord with a major 7th
            None: Other chord. Either a 7th chord with a minor 7th, or a major or minor triad.

    figbass : string
        The figured bass notation for the chord, representing its inversion. Importantly:
            None: Triad in first inversion
            '6':  Triad with 3rd in the bass
            '64': Triad with 5th in the bass

    Returns
    -------
    chord_type : ChordType
        The chord type of the given features. One of:
            - Major triad
            - Minor triad
            - Diminished triad
            - Augmented triad
            - Minor seventh chord
            - Dominant seventh chord
            - Major seventh chord
            - Minor major seventh chord
            - Diminished seventh chord
            - Half-diminished seventh chord
            - Augmented seventh chord
            - Augmented major seventh chord
    """
    if pd.isnull(figbass):
        figbass = None
    if pd.isnull(form):
        form = None

    # Triad
    if figbass in [None, '6', '64']:
        if form == 'o':
            return ChordType.DIMINISHED
        if form == '+':
            return ChordType.AUGMENTED
        return ChordType.MAJOR if is_major else ChordType.MINOR

    # Seventh chord
    if form == 'o':
        return ChordType.DIM7
    if form == '%':
        return ChordType.HALF_DIM7
    if form == '+':
        return ChordType.AUG_MIN7
    if form == '+M':
        # TODO: Check this
        return ChordType.AUG_MAJ7

    if is_major:
        return ChordType.MAJ_MAJ7 if form == 'M' else ChordType.MAJ_MIN7
    return ChordType.MIN_MAJ7 if form == 'M' else ChordType.MIN_MIN7


INVERSIONS = {
    '9':  0,
    '7':  0,
    '6':  1,
    '65': 1,
    '43': 2,
    '64': 2,
    '2':  3,
    '42': 3
}


def get_chord_inversion(figbass: str) -> int:
    """
    Get the chord inversion number from a figured bass string.

    Parameters
    ----------
    figbass : str
        The figured bass representation of the chord's inversion.

    Returns
    -------
    inversion : int
        An integer representing the chord inversion. 0 for root position, 1 for 1st inversion, etc.
    """
    if pd.isnull(figbass):
        return 0

    return INVERSIONS[figbass]




def get_pitch_from_string(pitch_string: str, pitch_type: PitchType) -> int:
    """
    Get the pitch from a pitch string.

    Parameters
    ----------
    pitch_string : str
        The pitch we want. Should be an uppercase letter [A-G], suffixed with some number of
        flats (b) or sharps (#).
    pitch_type : PitchType
        The type of pitch to return. MIDI for C = 0, TPC for C = 15.

    Returns
    -------
    pitch : int
        An integer representing the given pitch string as the given type.
    """
    accidental_adjustment, pitch_string = get_accidental_adjustment(pitch_string, in_front=False)

    pitch = STRING_TO_PITCH[pitch_type][pitch_string]
    pitch += accidental_adjustment * ACCIDENTAL_ADJUSTMENT[pitch_type]

    return pitch
