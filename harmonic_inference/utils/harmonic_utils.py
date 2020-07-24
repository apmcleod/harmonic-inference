"""Utility functions for getting harmonic and pitch information from the corpus DataFrames."""

from typing import List
import pandas as pd
import numpy as np


MAX_PITCH_DEFAULT = 127
PITCHES_PER_OCTAVE = 12

# Scale tone semitone difference from root
MAJOR_SCALE = [0, 0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 0, 2, 3, 5, 7, 8, 10]

NUMERAL_TO_NUMBER = {
    'I':   1,
    'II':  2,
    'III': 3,
    'IV':  4,
    'V':   5,
    'VI':  6,
    'VII': 7
}

NOTE_TO_INDEX = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11
}

# Triad types as one-hot semitone presence vectors
TRIAD_TYPES_SEMITONES = {
    'M': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'o': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    '+': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
}
TRIAD_TYPES_SEMITONES['%'] = TRIAD_TYPES_SEMITONES['o'] # Half-diminished

# Seventh types as semitone distance above root
SEVENTH_TYPES_SEMITONES = {
    'M': 11,
    'm': 10,
    'o': 9
}
SEVENTH_TYPES_SEMITONES['+'] = SEVENTH_TYPES_SEMITONES['m'] # Augmented
SEVENTH_TYPES_SEMITONES['%'] = SEVENTH_TYPES_SEMITONES['m'] # Half-diminished


CHORD_TYPES = [
    'M',
    'm',
    'o',
    '+',
    'mm7',
    'Mm7',
    'MM7',
    'mM7',
    'o7',
    '%7',
    '+7'
]


def get_one_hot_labels() -> List(str):
    """
    Get the human-readable label of every one-hot chord value.

    Returns
    -------
    labels : list
        A List, where labels[0] is the String interpretation of the one-hot chord label 0, etc.
    """
    labels = []
    for chord_type in CHORD_TYPES:
        for root in ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']:
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



def get_numeral_semitones(numeral: str, is_major: bool) -> (int, bool):
    """
    Convert the numeral of a chord tonic to a semitone offset, and return whether it is major
    or minor.

    Parameters
    ----------
    numeral : string
        The numeral of a chord, like I, bii, etc.

    is_major : boolean
        True if the current key is major. False otherwise.

    Returns
    -------
    semitones : int
         The number of semitones above the key tonic the given chord is.

    is_major : boolean
        True if the chord is major (upper-case). False otherwise.
    """
    adjustment, numeral = get_accidental_adjustment(numeral)

    if numeral.upper() in ['GER', 'IT', 'FR']:
        numeral = numeral.lower()
        semitones = MINOR_SCALE[6]
    elif is_major:
        semitones = MAJOR_SCALE[NUMERAL_TO_NUMBER[numeral.upper()]]
    else:
        semitones = MINOR_SCALE[NUMERAL_TO_NUMBER[numeral.upper()]]

    return semitones + adjustment, numeral.isupper()



def get_bass_step_semitones(bass_step: str, is_major: bool) -> int:
    """
    Get the given bass step in semitones.

    Parameters
    ----------
    bass_step : string
        The bass step of a chord, 1, b7, etc.

    is_major : boolean
        True if the current key is major. False otherwise.

    Returns
    -------
    semitones : int
        The number of semitones above the chord root the given bass step is.
        None if the data is malformed ("Error" or "Unclear").
    """
    adjustment, bass_step = get_accidental_adjustment(bass_step)

    try:
        if is_major:
            semitones = MAJOR_SCALE[int(bass_step)]
        else:
            semitones = MINOR_SCALE[int(bass_step)]
        return semitones + adjustment
    except ValueError:
        return None



def get_key(key: str) -> (int, bool):
    """
    Get the tonic index of a given key string.

    Parameters
    ----------
    key : string
        The key, C, db, etc.

    Returns
    -------
    tonic_index : int
        The tonic index of the key, with 0 = C, 1 = C#/Db, etc.

    is_major : boolean
        True if the given key is major (the tonic is upper-case). False otherwise.
    """
    adjustment, key = get_accidental_adjustment(key, in_front=False)

    is_major = key.isupper()
    if is_major:
        tonic_index = NOTE_TO_INDEX[key]
    else:
        tonic_index = NOTE_TO_INDEX[key.upper()]

    return tonic_index + adjustment, is_major



def transpose_chord_vector(chord_vector: List(int), transposition: int) -> List(int):
    """
    Transpose a chord vector by a certain number of semitones.

    Parameters
    ----------
    chord_vector : list(int)
        A binary vector representation of the given chord type, where 1 indicates
        the presence of a note in the given chord type, and 0 represents non-presence.
        The vector is length 12. The root may be in any position.

    transposition : int
        Rotate the vector by the given amount. If the chord's root was previously at
        index 0, the returned vector will have the root at index (0 + transposition) % 12.

    Returns
    -------
    transposed_chord_vector : list(int)
        The input vector, with each chord tone transposed by the given amount.
    """
    if isinstance(chord_vector, list):
        # Equivalent to np.roll(chord_vector, transposition), but without np conversion.
        return chord_vector[-transposition:] + chord_vector[:-transposition]
    return np.roll(chord_vector, transposition)



def get_vector_from_chord_type(type_string: str) -> List(int):
    """
    Convert a chord type string into a vector representation of semitone presence.

    Parameters
    ----------
    type_string : string
        A type of chord. One of:
            'M':   Major triad
            'm':   Minor triad
            'o':   Diminished triad
            '+':   Augmented triad
            'mm7': Minor seventh chord
            'Mm7': Dominant seventh chord
            'MM7': Major seventh chord
            'mM7': Minor major seventh chord
            'o7':  Diminished seventh chord
            '%7':  Half-diminished seventh chord
            '+7':  Augmented seventh chord

    Returns
    -------
    chord_vector : list(int)
        A binary vector representation of the given chord type, where 1 indicates
        the presence of a note in the given chord type, and 0 represents non-presence.
        The vector is length 12, where chord_vector[0] is the root note, chord_vector[1]
        is a half step up from the root, etc.
    """
    chord_vector = TRIAD_TYPES_SEMITONES[type_string[0]].copy()

    if type_string[-1] == '7':
        chord_vector[SEVENTH_TYPES_SEMITONES[type_string[-2]]] = 1

    return chord_vector



def get_chord_type_string(is_major: bool, form: str = None, figbass: str = None) -> str:
    """
    Get the chord type string, given some features about the chord.

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
            'M':  7th chord with a major 7th
            None: Other chord. Either a 7th chord with a minor 7th, or a major or minor triad.

    figbass : string
        The figured bass notation for the chord, representing its inversion. Importantly:
            None: Triad in first inversion
            '6':  Triad with 3rd in the bass
            '64': Triad with 5th in the bass

    Returns
    -------
    chord_type : string
        A string representing the chord type of this chord. One of:
            'M':   Major triad
            'm':   Minor triad
            'o':   Diminished triad
            '+':   Augmented triad
            'mm7': Minor seventh chord
            'Mm7': Dominant seventh chord
            'MM7': Major seventh chord
            'mM7': Minor major seventh chord
            'o7':  Diminished seventh chord
            '%7':  Half-diminished seventh chord
            '+7':  Augmented seventh chord
    """
    if pd.isnull(figbass):
        figbass = None
    if pd.isnull(form):
        form = None

    # Triad
    if figbass in [None, '6', '64']:
        if form in ['o', '+']:
            return form
        return 'M' if is_major else 'm'

    # Seventh chord
    if form in ['o', '%', '+']:
        return f"{form}7"

    triad = 'M' if is_major else 'm'
    seventh = 'M' if form == 'M' else 'm'
    return f"{triad}{seventh}7"
