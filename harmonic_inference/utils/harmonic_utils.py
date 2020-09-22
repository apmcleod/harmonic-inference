"""Utility functions for getting harmonic and pitch information from the corpus DataFrames."""
from typing import List, Tuple
import pandas as pd
import numpy as np

from harmonic_inference.data.data_types import KeyMode, PitchType, ChordType
import harmonic_inference.utils.harmonic_constants as hc


def get_chord_label_list(pitch_type: PitchType, use_inversions=True) -> List[str]:
    """
    Get the human-readable label of every chord label.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type representation being used.
    use_inversions : bool
        True to count the different inversions of each chord as different chords.

    Returns
    -------
    labels : List[String]
        A List, where labels[0] is the String interpretation of the one-hot chord label 0, etc.
    """
    roots = [get_pitch_string(i, pitch_type) for i in range(hc.NUM_PITCHES[pitch_type])]

    labels = []
    for chord_type in ChordType:
        for root in roots:
            if use_inversions:
                for inv in range(get_chord_inversion_count(chord_type)):
                    labels.append(f'{root}:{get_chord_string(chord_type)}, inv:{inv}')
            else:
                labels.append(f'{root}:{get_chord_string(chord_type)}')

    return labels


def get_chord_one_hot_index(
    chord_type: ChordType,
    root_pitch: int,
    pitch_type: PitchType,
    inversion: int = 0,
    use_inversion: bool = True
) -> int:
    """
    Get the one hot index of a given chord.

    Parameters
    ----------
    chord_type : ChordType
        The chord type of the chord.
    root_pitch : int
        The pitch of the root of this chord.
    pitch_type : int
        The representation used for `root_pitch`.
    inversion : int
        The inversion of this chord. Used only if use_inversion is True.
    use_inversion : inv
        True to take the chord's inversion into acccount.

    Returns
    -------
    index : int
        The index of the given chord's label in the list of all possible chord labels.
    """
    if root_pitch < 0 or root_pitch >= hc.NUM_PITCHES[pitch_type]:
        raise ValueError("Given root is outside of valid range")
    if use_inversion:
        if inversion < 0 or inversion >= get_chord_inversion_count(chord_type):
            raise ValueError(
                f"inversion {inversion} outside of valid range for chord {chord_type}"
            )
        chord_inversions = np.array(
            [get_chord_inversion_count(chord) for chord in ChordType], dtype=int
        )
    else:
        chord_inversions = np.ones(len(ChordType), dtype=int)

    index = np.sum(hc.NUM_PITCHES[pitch_type] * chord_inversions[:chord_type.value])
    index += chord_inversions[chord_type.value] * root_pitch

    if use_inversion:
        index += inversion

    return index


def get_key_label_list(pitch_type: PitchType) -> List[str]:
    """
    Get the list of all key labels, in human-readable format.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type of the tonic labels.

    Returns
    -------
    key_labels : List[str]
        A List where key_labels[i] is the human-readable label for key index i.
    """
    tonics = [get_pitch_string(i, pitch_type) for i in range(hc.NUM_PITCHES[pitch_type])]

    labels = []
    for key_mode in KeyMode:
        for tonic in tonics:
            if key_mode == KeyMode.MINOR:
                tonic = str(tonic).lower()
            labels.append(f'{tonic}:{key_mode}')

    return labels


def get_key_one_hot_index(key_mode: KeyMode, tonic: int, pitch_type: PitchType) -> int:
    """
    Get the one hot index of a given key.

    Parameters
    ----------
    key_mode : KeyMode
        The mode of the key.
    tonic : int
        The pitch of the tonic of this key.
    pitch_type : int
        The representation used for `tonic`.

    Returns
    -------
    index : int
        The index of the given key's label in the list of all possible key labels.
    """
    if tonic < 0 or tonic >= hc.NUM_PITCHES[pitch_type]:
        raise ValueError("Given root is outside of valid range")

    return hc.NUM_PITCHES[pitch_type] * key_mode.value + tonic


def get_chord_inversion_count(chord_type: ChordType) -> int:
    """
    Get the number of possible inversions of the given chord.

    Parameters
    ----------
    chord_type : ChordType
        The chord type whose inversion count to return.

    Returns
    -------
    inversions : int
        The number of possible inversions of the given chord.
    """
    return hc.CHORD_INVERSION_COUNT[chord_type]


def get_accidental_adjustment(string: str, in_front: bool = True) -> Tuple[int, str]:
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
    elif interval < 0:
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
    chord_vector = np.zeros(hc.NUM_PITCHES[pitch_type], dtype=int)
    chord_vector[hc.CHORD_PITCHES[pitch_type][chord_type]] = 1
    return chord_vector


def get_interval_from_numeral(numeral: str, mode: KeyMode, pitch_type: PitchType) -> int:
    """
    Get the interval from the key tonic to the given scale degree numeral.

    Parameters
    ----------
    numeral : str
        An upper or lowercase roman numeral, prepended by accidentals.
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
    return get_interval_from_scale_degree(numeral, True, mode, pitch_type)


def tpc_interval_to_midi_interval(tpc_interval: int) -> int:
    """
    Convert a TPC interval (in fifths, where 0 is a unison interval) into a MIDI interval
    (in semitones).

    Parameters
    ----------
    tpc_interval : int
        The TPC representation of the given interval, measured in fifths above a given pitch.

    Returns
    -------
    midi_interval : int
        The MIDI representation of the given interval, measured in semitones above a given pitch.
    """
    return (hc.TPC_INTERVAL_SEMITONES * tpc_interval) % hc.NUM_PITCHES[PitchType.MIDI]


def get_interval_from_scale_degree(scale_degree: str, accidentals_prefixed: bool,
                                   mode: KeyMode, pitch_type: PitchType) -> int:
    """
    Get an interval from the string of a scale degree, prefixed or suffixed with flats or sharps.

    Parameters
    ----------
    scale_degree : str
        A string of a scale degree, either as a roman numeral or a number, with any accidentals
        suffixed or prefixed as 'b' (flats) or '#' (sharps).
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

    interval = hc.SCALE_INTERVALS[mode][pitch_type][hc.SCALE_DEGREE_TO_NUMBER[scale_degree]]
    interval += accidental_adjustment * hc.ACCIDENTAL_ADJUSTMENT[pitch_type]

    return interval


def transpose_pitch(pitch: int, interval: int, pitch_type: PitchType) -> int:
    """
    Transpose the given pitch by the given interval.

    Parameters
    ----------
    pitch : int
        The original pitch.
    interval : int
        The amount to transpose the given pitch by.
    pitch_type : PitchType
        The pitch type. If MIDI, the returned pitch will be on the range [0, 12), and the given
        interval is interpreted as semitones, which is modded to fall in the range. If TPC, the
        returned pitch must be on the range [0, 35), and the given interval is interpreted as
        fifths. This is not modded. Rather, an exception is raised if the returned pitch would be
        outside of the TPC range.

    Returns
    -------
    pitch : int
        The given pitch, transposed by the given interval.
    """
    if pitch_type == PitchType.MIDI:
        return (pitch + interval) % hc.NUM_PITCHES[PitchType.MIDI]
    pitch = pitch + interval

    if pitch < 0 or pitch >= hc.NUM_PITCHES[PitchType.TPC]:
        raise ValueError(f"pitch_type is TPC but transposed pitch {pitch} lies outside of TPC "
                         "range.")

    return pitch


def get_chord_inversion(figbass: str) -> int:
    """
    Get the chord inversion number from a figured bass string.

    Parameters
    ----------
    figbass : str
        The figured bass representation of the chord.

    Returns
    -------
    inversion : int
        An integer representing the chord inversion. 0 for root position, 1 for 1st inversion, etc.
    """
    if pd.isnull(figbass) or len(figbass) == 0:
        return 0
    try:
        return hc.FIGBASS_INVERSIONS[figbass]
    except Exception:
        raise ValueError(f"{str} is not a valid figured bass symbol for detecting inversions.")


# Chord/Pitch string <--> object conversion functions =============================================


def get_chord_type_from_string(chord_string: str) -> ChordType:
    """
    Map a chord type string to a ChordType.

    Parameters
    ----------
    chord_string : str
        A string representing a chord type.
        One of: M, m, +, o, MM7, Mm7, mM7, mm7, %7, o7, +7, +M7.

    Returns
    -------
    chord_type : ChordType
        The chord type object corresponding to the given chord_string.

    Raises
    ------
    ValueError
        If the given chord_string is not one of the recognized chord type strings.
    """
    try:
        return hc.STRING_TO_CHORD_TYPE[chord_string]
    except Exception:
        raise ValueError(f"String type {chord_string} not recognized.")


def get_chord_string(chord_type: ChordType) -> str:
    """
    Get a chord type string from a given ChordType.

    Parameters
    ----------
    chord_type : ChordType
        A ChordType to convert to a string.

    Returns
    -------
    chord_type_str : str
        The string representation of the given ChordType.
    """
    return hc.CHORD_TYPE_TO_STRING[chord_type]


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

    pitch = hc.STRING_TO_PITCH[pitch_type][pitch_string]
    pitch += accidental_adjustment * hc.ACCIDENTAL_ADJUSTMENT[pitch_type]

    if pitch_type == PitchType.MIDI:
        pitch %= hc.NUM_PITCHES[PitchType.MIDI]
    elif pitch_type == PitchType.TPC and (pitch < 0 or pitch >= hc.NUM_PITCHES[PitchType.TPC]):
        raise ValueError(
            f"Pitch type is TPC, but returned pitch {pitch} would be outside of valid range."
        )

    return pitch


def get_pitch_string(pitch: int, pitch_type: PitchType) -> str:
    """
    Get the string representation of a pitch.

    Parameters
    ----------
    pitch : int
        The pitch to convert to a string.
    pitch_type : PitchType
        The type of pitch this is.

    Returns
    -------
    pitch_string : str
        A string representation of the given pitch.
    """
    if pitch_type == PitchType.MIDI:
        return hc.PITCH_TO_STRING[PitchType.MIDI][pitch % hc.NUM_PITCHES[PitchType.MIDI]]

    accidental = 0
    accidental_string = '#'
    while pitch < min(hc.PITCH_TO_STRING[PitchType.TPC].keys()):
        pitch += hc.TPC_NATURAL_PITCHES
        accidental -= 1
    while pitch > max(hc.PITCH_TO_STRING[PitchType.TPC].keys()):
        pitch -= hc.TPC_NATURAL_PITCHES
        accidental += 1

    if accidental < 0:
        accidental_string = 'b'
        accidental = -accidental

    return hc.PITCH_TO_STRING[PitchType.TPC][pitch] + (accidental_string * accidental)
