"""Utility functions for getting harmonic and pitch information from the corpus DataFrames."""
import itertools
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd

import harmonic_inference.utils.harmonic_constants as hc
from harmonic_inference.data.data_types import NO_REDUCTION, ChordType, KeyMode, PitchType


def get_chord_label_list(
    pitch_type: PitchType,
    use_inversions: bool = True,
    relative: bool = False,
    relative_to: int = None,
    pad: bool = False,
    reduction: Dict[ChordType, ChordType] = None,
) -> List[str]:
    """
    Get the human-readable label of every chord label.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type representation being used.
    use_inversions : bool
        True to count the different inversions of each chord as different chords.
    relative : bool
        True to get relative chord labels. False otherwise.
    relative_to : int
        The pitch to be relative to, if any.
    pad : bool
        True to add padding around possible pitches, if relative is True.
    reduction : Dict[ChordType, ChordType]
        The reduction mapping of chord types.

    Returns
    -------
    labels : List[String]
        A List, where labels[0] is the String interpretation of the one-hot chord label 0, etc.
    """
    if reduction is None:
        reduction = NO_REDUCTION

    if relative:
        if pitch_type == PitchType.TPC:
            minimum = hc.MIN_RELATIVE_TPC
            maximum = hc.MAX_RELATIVE_TPC
            if pad:
                minimum -= hc.RELATIVE_TPC_EXTRA
                maximum += hc.RELATIVE_TPC_EXTRA

            int_roots = list(range(minimum, maximum))

        else:
            int_roots = list(range(0, hc.NUM_PITCHES[pitch_type]))

        if relative_to is None:
            roots = [str(root) for root in int_roots]
        else:
            roots = []
            for root in int_roots:
                try:
                    roots.append(
                        get_pitch_string(
                            transpose_pitch(root, relative_to, pitch_type),
                            pitch_type,
                        )
                    )
                except ValueError:
                    roots.append(str(root))

    else:
        roots = [get_pitch_string(i, pitch_type) for i in range(hc.NUM_PITCHES[pitch_type])]

    return [
        f"{root}{get_chord_string(chord_type, True)}{get_figbass_string(inv, chord_type)}"
        for chord_type, root in itertools.product(sorted(set(reduction.values())), roots)
        for inv in (range(get_chord_inversion_count(chord_type)) if use_inversions else [None])
    ]


def split_changes_into_list(changes: str) -> List[str]:
    """
    Given a string of chord alterations (e.g., suspensions or added tones),
    split them into a list of individual alterations.

    Parameters
    ----------
    changes : str
        A string representing a chord alterations, like "64" or "13+b2".

    Returns
    -------
    changes_list : List[str]
        The given changes, split into a list of strings representing each individual
        alteration.
    """

    def get_first_change_length(changes: str) -> int:
        """
        Get the length of the first chord change in the given changes string.

        Parameters
        ----------
        changes : str
            A string representing a chord alterations, like "64" or "13+b2".

        Returns
        -------
        length : int
            The length of the first change in the given string. For example, given "b42",
            this will return 2.
        """
        if changes is None or len(changes) == 0:
            return 0

        # Find the first numeric character
        length = 0
        while not changes[length].isnumeric():
            length += 1

        if changes[length] == "1":
            length += 2
        else:
            length += 1

        return length

    changes_list = []
    length = get_first_change_length(changes)
    while length != 0:
        changes_list.append(changes[:length])
        changes = changes[length:]
        length = get_first_change_length(changes)

    return changes_list


def get_added_and_removed_pitches(
    chord_root_tpc: int,
    chord_type: ChordType,
    changes: str,
    key_tonic_tpc: int,
    key_mode: KeyMode,
) -> Dict[str, str]:
    """
    Get a mapping of pitch alterations from the given chord. Pitches are given
    and returned with PitchType TPC because accidental-specific math is required
    to correctly apply accidentals.

    Parameters
    ----------
    chord_root_tpc : int
        The root pitch of the given chord, in TPC notation.
    chord_type : ChordType
        The type of the given chord.
    changes : str
        A string of the changes or alterations of a given chord, like "64" or "+b2".
    key_tonic_tpc : int
        The tonic pitch of the current key, including any relative root, in TPC notation.
    key_mode : KeyMode
        The mode of the current key, including any relative root.

    Returns
    -------
    changed_pitches : Dict[str, str]
        A dictionary representing pitch alterations to the given chord. Each entry represents
        a mapping of original_pitch -> new_pitch, represented as a string of their TPC integer.
        If original_pitch is empty, then the new_pitch is simply added. If new_pitch begins
        with "+", then it is added in an upper octave.
    """
    added_pitches = []
    removed_pitches = []

    # First, we have to find the chord numeral degree, since changes are notated numerically
    # relative to the chord's tonal pitch class.
    # i.e., any "2" change to a IV chord will have some V in it, regardless of any accidentals.
    chord_root_str = get_pitch_string(chord_root_tpc, PitchType.TPC)

    for degree in range(1, 8):
        interval = get_interval_from_scale_degree(str(degree), True, key_mode, PitchType.TPC)
        pitch_str = get_pitch_string(interval + key_tonic_tpc, PitchType.TPC)
        if pitch_str[0] == chord_root_str[0]:
            break

    changes_list = split_changes_into_list(changes)

    # Calculate added pitches first
    for change in changes_list:
        while change[0] in "v^+":
            change = change[1:]

        # Find the scale degree for this change
        accidental_count, new_change = get_accidental_adjustment(change, in_front=True)
        accidental_count = abs(accidental_count)

        octave = "+" if int(new_change) >= 8 else ""

        # Convert change to be relative to the key tonic, including accidentals
        change_degree = (int(new_change) + degree - 2) % 7  # -2 since both are 1-indexed
        change_degree += 1  # Conver back to 1-indexing
        change_degree_str = change[:accidental_count] + str(change_degree)

        # Calculate interval above scale degree, including additional octaves
        interval = get_interval_from_scale_degree(change_degree_str, True, key_mode, PitchType.TPC)

        # Store added pitch, including "+" if the pitch is an octave up
        added_pitches.append(octave + str(interval + key_tonic_tpc))

    # Calculate chord vector in ascending pitch order
    chord_vector = get_vector_from_chord_type(chord_type, PitchType.TPC, chord_root_tpc)
    chord_vector = np.where(chord_vector == 1)[0]
    ascending_chord_vector = []

    for degree in range(1, 8):
        interval = get_interval_from_scale_degree(str(degree), True, key_mode, PitchType.TPC)
        pitch_str = get_pitch_string(interval + chord_root_tpc, PitchType.TPC)

        for pitch in chord_vector:
            if get_pitch_string(pitch, PitchType.TPC)[0] == pitch_str[0]:
                ascending_chord_vector.append(pitch)

    # Calculate removed pitches
    for change in changes_list:
        if change[0] == "+":
            # Added pitch only - no deletion
            removed_pitches.append("")

        _, new_change = get_accidental_adjustment(change, in_front=True)

        if change[0] == "^" or (new_change in "246" and change[0] == "#"):
            # Replaces the above pitch

            if change == "#6" and len(ascending_chord_vector) == 3:
                # Special case: If #6 occurs for a triad, it is an addition,
                # since it cannot be a lower replacement to a non-existent 7
                removed_pitches.append("")
                continue

            # 2 replaces the 2nd chord pitch, 4 replaces the 3rd, etc.
            removed_pitches.append(str(ascending_chord_vector[int(change[-1]) // 2]))

        elif change[0] == "v" or (new_change in "246" and change[0] != "#"):
            # Replaces the below pitch

            # 2 replaces the 1st chord pitch, 4 replaces the 2nd, etc.
            removed_pitches.append(str(ascending_chord_vector[int(change[-1]) // 2 - 1]))

        else:
            # No removed pitch
            removed_pitches.append("")

    return {removed: added for removed, added in zip(removed_pitches, added_pitches)}


def get_chord_from_one_hot_index(
    one_hot_index: Union[int, slice],
    pitch_type: PitchType,
    use_inversions: bool = True,
    relative: bool = False,
    pad: bool = False,
    reduction: Dict[ChordType, ChordType] = None,
) -> Union[Tuple[int, ChordType, int], List[Tuple[int, ChordType, int]]]:
    """
    Get a chord object from a one hot index.

    Parameters
    ----------
    one_hot_index : int
        The one-hot index of the chord to return.
    pitch_type : PitchType
        The pitch type representation being used.
    use_inversions : bool
        True to count the different inversions of each chord as different chords.
    relative : bool
        True to get relative chord labels. False otherwise.
    relative_to : int
        The pitch to be relative to, if any.
    pad : bool
        True to add padding around possible pitches, if relative is True.
    reduction : Dict[ChordType, ChordType]
        The reduction mapping of chord types.

    Returns
    -------
    root : int
        The root pitch of the chord.
    chord_type : ChordType
        The chord type of the corresponding chord.
    inversion : int
        The inversion of the corresponding chord.
    """
    if reduction is None:
        reduction = NO_REDUCTION

    if relative:
        if pitch_type == PitchType.TPC:
            minimum = hc.MIN_RELATIVE_TPC
            maximum = hc.MAX_RELATIVE_TPC
            if pad:
                minimum -= hc.RELATIVE_TPC_EXTRA
                maximum += hc.RELATIVE_TPC_EXTRA

            roots = list(range(minimum, maximum))

        else:
            roots = list(range(0, hc.NUM_PITCHES[pitch_type]))

    else:
        roots = list(range(hc.NUM_PITCHES[pitch_type]))

    return [
        (root, chord_type, inv)
        for chord_type, root in itertools.product(sorted(set(reduction.values())), roots)
        for inv in (range(get_chord_inversion_count(chord_type)) if use_inversions else [0])
    ][one_hot_index]


def get_chord_one_hot_index(
    chord_type: ChordType,
    root_pitch: int,
    pitch_type: PitchType,
    inversion: int = 0,
    use_inversion: bool = True,
    relative: bool = False,
    pad: bool = False,
    reduction: Dict[ChordType, ChordType] = None,
) -> int:
    """
    Get the one hot index of a given chord.

    Parameters
    ----------
    chord_type : ChordType
        The chord type of the chord.
    root_pitch : int
        The pitch of the root of this chord, either as an absolute or relative pitch,
        depending on `relative`.
    pitch_type : int
        The representation used for `root_pitch`.
    inversion : int
        The inversion of this chord. Used only if use_inversion is True.
    use_inversion : inv
        True to take the chord's inversion into acccount.
    relative : bool
        True to get the chord's relative one-hot index. False for absolute.
    pad : bool
        If True, pitch_type is TPC, and relative is True, add extra pitches to the number
        of possible chord roots.
    reduction : Dict[ChordType, ChordType]
        A reduction for the chord_type.

    Returns
    -------
    index : int
        The index of the given chord's label in the list of all possible chord labels.
    """
    if reduction is None:
        reduction = NO_REDUCTION
    else:
        # Correct inversion to root on reduction
        if reduction[chord_type] != chord_type and inversion >= get_chord_inversion_count(
            reduction[chord_type]
        ):
            inversion = 0

    chord_types = sorted(set(reduction.values()))
    chord_type = reduction[chord_type]
    chord_type_index = chord_types.index(chord_type)

    if pitch_type == PitchType.MIDI or not relative:
        num_pitches = hc.NUM_PITCHES[pitch_type]
    else:
        num_pitches = hc.MAX_RELATIVE_TPC - hc.MIN_RELATIVE_TPC
        if pad:
            num_pitches += 2 * hc.RELATIVE_TPC_EXTRA

    if root_pitch < 0 or root_pitch >= num_pitches:
        raise ValueError(f"Given root ({root_pitch}) is outside of valid range")
    if use_inversion:
        if inversion < 0 or inversion >= get_chord_inversion_count(chord_type):
            raise ValueError(f"inversion {inversion} outside of valid range for chord {chord_type}")
        chord_inversions = np.array(
            [get_chord_inversion_count(chord) for chord in chord_types], dtype=int
        )
    else:
        chord_inversions = np.ones(len(chord_types), dtype=int)

    index = np.sum(num_pitches * chord_inversions[:chord_type_index])
    index += chord_inversions[chord_type_index] * root_pitch

    if use_inversion:
        index += inversion

    return index


def get_key_label_list(
    pitch_type: PitchType,
    relative: bool = False,
    relative_to: int = None,
) -> List[str]:
    """
    Get the list of all key labels, in human-readable format.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type of the tonic labels.
    relative : bool
        True to output relative key labels. False for absolute.
    relative_to : int
        The pitch to be relative to, if any.

    Returns
    -------
    key_labels : List[str]
        A List where key_labels[i] is the human-readable label for key index i.
    """
    if relative:
        if pitch_type == PitchType.TPC:
            minimum = hc.MIN_KEY_CHANGE_INTERVAL_TPC
            maximum = hc.MAX_KEY_CHANGE_INTERVAL_TPC

            int_tonics = list(range(minimum, maximum))

        else:
            int_tonics = list(range(0, hc.NUM_PITCHES[pitch_type]))

        if relative_to is None:
            tonics = [str(tonic) for tonic in int_tonics]
        else:
            tonics = []
            for tonic in int_tonics:
                try:
                    tonics.append(
                        get_pitch_string(
                            transpose_pitch(tonic, relative_to, pitch_type),
                            pitch_type,
                        )
                    )
                except ValueError:
                    tonics.append(str(tonic))

    else:
        tonics = [get_pitch_string(i, pitch_type) for i in range(hc.NUM_PITCHES[pitch_type])]

    return [
        tonic.lower() if key_mode == KeyMode.MINOR else tonic
        for key_mode, tonic in itertools.product(KeyMode, tonics)
    ]


def get_key_from_one_hot_index(
    one_hot: Union[int, slice],
    pitch_type: PitchType,
    relative: bool = False,
) -> Union[Tuple[int, KeyMode], List[Tuple[int, KeyMode]]]:
    """
    Get the key tonic and mode of the given one hot key label index.

    Parameters
    ----------
    one_hot : Union[int, slice]
        The one hot key information to return.
    pitch_type : PitchType
        The pitch type of the tonic labels.
    relative : bool
        True to output relative key pitch. False for absolute.

    Returns
    -------
    key_tonic : int
        The tonic of the corresponding key.
    key_mode : KeyMode
        The mode of the corresponding key.
    """
    if relative:
        if pitch_type == PitchType.TPC:
            minimum = hc.MIN_KEY_CHANGE_INTERVAL_TPC
            maximum = hc.MAX_KEY_CHANGE_INTERVAL_TPC

            tonics = list(range(minimum, maximum))

        else:
            tonics = list(range(0, hc.NUM_PITCHES[pitch_type]))

    else:
        tonics = list(range(hc.NUM_PITCHES[pitch_type]))

    return [(tonic, key_mode) for key_mode, tonic in itertools.product(KeyMode, tonics)][one_hot]


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


def decode_relative_keys(
    relative_string: str,
    tonic: int,
    mode: KeyMode,
    pitch_type: PitchType,
) -> Tuple[int, KeyMode]:
    """
    Decode a relative key symbol (which might be itself applied, e.g. `V/V`)
    into a relative mode and tonic.

    Parameters
    ----------
    relative_string : str
        A relative key symbol. Either a Roman Numeral, like `V` or `iii`,
        or multiple Roman Numerals, separated by slashes, like `V/V` or
        `vii/V`.
    tonic : int
        The absolute tonic pitch of the current local key to which the given
        root is applied.
    mode : KeyMode
        The mode of the current local key to which the given root is applied.
    pitch_type : PitchType
        The pitch type used for the tonic pitch, and to be used to return the
        relative applied tonic.

    Returns
    -------
    relative_tonic : int
        The tonic pitch of the applied key.

    relative_mode : KeyMode
        The mode of the applied key.
    """
    # Handle doubly-relative chords iteratively
    for relative in reversed(relative_string.split("/")):
        # Relativeroot is listed relative to local key. We want it absolute.
        relative_transposition = get_interval_from_numeral(relative, mode, pitch_type=pitch_type)
        mode = KeyMode.MINOR if relative[-1].islower() else KeyMode.MAJOR
        tonic = transpose_pitch(tonic, relative_transposition, pitch_type)

    return tonic, mode


def get_bass_note(
    chord_type: ChordType,
    root: int,
    inversion: int,
    pitch_type: PitchType,
    modulo: bool = False,
) -> int:
    """
    Get the bass note of a chord given a chord type and inversion.

    Parameters
    ----------
    chord_type : ChordType
        The chord type.
    root : int
        The root note of the given chord.
    inversion : int
        The inversion of the chord.
    pitch_type : PitchType
        The desired pitch type to return.
    modulo : bool
        If True, become robust to errors by taking inversion modulo of the size of
        the chord.

    Returns
    -------
    bass : int
        The bass note of a chord of the given type, root, and inversion.
    """
    chord_pitches = hc.CHORD_PITCHES[pitch_type][chord_type]
    bass = chord_pitches[(inversion % len(chord_pitches)) if modulo else inversion]
    return transpose_pitch(bass, root - hc.C[pitch_type], pitch_type)


def get_default_chord_pitches(
    root: int,
    chord_type: ChordType,
    pitch_type: PitchType,
) -> Set[int]:
    """
    Get the default chord pitches list given a root and a chord type. The returned chord
    pitches set will contain the absolute pitch classes for the notes in the default version
    of that chord.

    Parameters
    ----------
    root : int
        The root pitch of the chord.
    chord_type : ChordType
        The chord type.
    pitch_type : PitchType
        The pitch type used for the root, and to use for the returned chord pitches.

    Returns
    -------
    chord_pitches : Set[int]
        The absolute pitches present in the default version of the given chord,
        represented with the given pitch type.
    """
    return set(
        transpose_pitch(pitch, root - hc.C[pitch_type], pitch_type)
        for pitch in hc.CHORD_PITCHES[pitch_type][chord_type]
    )


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


def absolute_to_relative(
    pitch: int,
    key_tonic: int,
    pitch_type: PitchType,
    is_key_change: bool,
    pad: bool = False,
) -> int:
    """
    Convert an absolute pitch to one relative to the given tonic.

    Parameters
    ----------
    pitch : int
        The absolute pitch.
    key_tonic : int
        The pitch be relative to.
    pitch_type : PitchType
        The pitch type for the given pitch and key tonic.
    is_key_change : bool
        True if the given absolute interval refers to a key change. False if it refers to
        a relative chord pitch. For key changes with PitchType.TPC, we adjust so that
        hc.MIN_KEY_CHANGE_INTERVAL_TPC is set to 0. For chord pitches, hc.MIN_RELATIVE_TPC
        is used instead.
    pad : bool
        If True or is_key_change is True, use the exact bounds. If False, and is_key_change
        is False, add hc.RELATIVE_TPC_EXTRA to the upper and lower bounds.

    Returns
    -------
    relative : int
        The given pitch, expressed relative to the given key_tonic.

    Raises
    ------
    ValueError
        If the resulting relative is outside of the allowed bounds.
    """
    if pitch_type == PitchType.MIDI:
        return transpose_pitch(pitch, -key_tonic, PitchType.MIDI)

    relative = pitch - key_tonic
    minimum = hc.MIN_KEY_CHANGE_INTERVAL_TPC if is_key_change else hc.MIN_RELATIVE_TPC
    maximum = hc.MAX_KEY_CHANGE_INTERVAL_TPC if is_key_change else hc.MAX_RELATIVE_TPC

    if pad and not is_key_change:
        minimum -= hc.RELATIVE_TPC_EXTRA
        maximum += hc.RELATIVE_TPC_EXTRA

    if relative < minimum or relative >= maximum:
        raise ValueError(f"Resulting relative pitch {relative} is outside of valid range.")

    return relative - minimum


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
        while string[0] == "b" and len(string) > 1:
            string = string[1:]
            adjustment -= 1

        while string[0] == "#":
            string = string[1:]
            adjustment += 1

    else:
        while string[-1] == "b" and len(string) > 1:
            string = string[:-1]
            adjustment -= 1

        while string[-1] == "#":
            string = string[:-1]
            adjustment += 1

    return adjustment, string


def transpose_chord_vector(
    chord_vector: List[int], interval: int, pitch_type: PitchType
) -> List[int]:
    """
    Transpose a chord vector by a given interval.

    Parameters
    ----------
    chord_vector : List[int]
        A binary vector representation of the given chord type, where 1 indicates
        the presence of a pitch in the given chord type, and 0 represents non-presence.

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


def get_vector_from_chord_type(
    chord_type: ChordType,
    pitch_type: PitchType,
    root: int = None,
) -> List[int]:
    """
    Convert a chord type into a one-hot vector representation of pitch presence.

    Parameters
    ----------
    chord_type : ChordType
        The type of chord whose pitch vector to return.
    pitch_type : PitchType
        The type of pitch vector to return.
    root : int
        If given, transpose the vector to this root. Otherwise, the returned vector
        will have a root of C.

    Returns
    -------
    chord_vector : List[int]
        A one-hot vector of the pitches present in a chord of the given chord type,
        with a specified root (if given), or root C (default, if root=None).
    """
    chord_vector = np.zeros(hc.NUM_PITCHES[pitch_type], dtype=int)
    chord_vector[hc.CHORD_PITCHES[pitch_type][chord_type]] = 1

    if root is not None:
        chord_vector = transpose_chord_vector(chord_vector, root - hc.C[pitch_type], pitch_type)

    return chord_vector


def get_chord_pitches_string(
    root: int,
    chord_type: ChordType,
    pitches: List[int],
    tonic: int,
    mode: KeyMode,
    pitch_type: PitchType,
) -> str:
    """
    Get a string encoding in a readable format the chord pitches for a given chord.
    If the pitches are the chord defaults, this will be the empty string. Otherwise,
    this will contain any alterations within parentheses.

    Parameters
    ----------
    root : int
        The chord's root pitch.
    chord_type : ChordType
        The chord type.
    pitches : List[int]
        The pitches contained in the chord.
    tonic : int
        The tonic of the current key.
    mode : KeyMode
        The mode of the current key.
    pitch_type : PitchType
        The pitch type used to encode the root and tonic.

    Returns
    -------
    chord_pitches_string : str
        A human-readable string of any alterations to the default chord tones. If the default
        tones are in the given pitches list, the empty string is returned. Otherwise, an
        alteration string is returned within parentheses.
    """

    def get_pitch_label(
        pitch: int, root: int, tonic: int, mode: KeyMode, pitch_type: PitchType
    ) -> Tuple[str, int]:
        """
        Get an accidental string and a pitch integer for a given pitch. The
        accidental + pitch_int will be the one by which the pitch should be referred to
        in the chord alteration string.

        Parameters
        ----------
        pitch : int
            The pitch we want to refer to.
        root : int
            The root of the chord.
        tonic : int
            The tonic of the key.
        mode : KeyMode
            The mode of the key.
        pitch_type : PitchType
            The pitch type used for the given pitch.

        Returns
        -------
        accidental : str
            An accidental string, which should be pre-pended to the returned pitch_int.
        pitch_int : int
            The pitch integer, in steps above the given pitch.
        """
        if pitch_type == PitchType.MIDI:
            return get_pitch_label(
                get_pitch_from_string(get_pitch_string(pitch, PitchType.MIDI), PitchType.TPC),
                get_pitch_from_string(get_pitch_string(root, PitchType.MIDI), PitchType.TPC),
                get_pitch_from_string(get_pitch_string(tonic, PitchType.MIDI), PitchType.TPC),
                mode,
                PitchType.TPC,
            )

        pitch_int = (
            "ABCDEFG".index(get_pitch_string(pitch, PitchType.TPC)[0])
            - "ABCDEFG".index(get_pitch_string(root, PitchType.TPC)[0])
        ) % 7
        sd_string = get_scale_degree_from_interval(pitch - tonic, mode, pitch_type)
        accidentals = sd_string.replace("V", "").replace("I", "")

        return accidentals, pitch_int + 1

    default_pitches = get_default_chord_pitches(root, chord_type, pitch_type)

    pitch_set = set(pitches)
    if pitch_set == default_pitches:
        return ""

    removed_pitches = np.array(list(default_pitches - pitch_set))
    added_pitches = np.array(list(pitch_set - default_pitches))

    if len(removed_pitches) == 0:
        # We only have added pitches
        accidentals, pitch_ints = zip(
            *[get_pitch_label(pitch, root, tonic, mode, pitch_type) for pitch in added_pitches]
        )
        pitch_strings = [f"+{accidentals[i]}{pitch_ints[i]}" for i in np.argsort(pitch_ints)]

        return "(" + "".join(reversed(pitch_strings)) + ")"

    # Here we have added and removed pitches
    added_accidentals, added_pitch_ints = zip(
        *[get_pitch_label(pitch, root, tonic, mode, pitch_type) for pitch in added_pitches]
    )
    added_accidentals = list(added_accidentals)
    added_pitch_ints = list(added_pitch_ints)
    removed_pitch_ints = [
        get_pitch_label(pitch, root, tonic, mode, pitch_type)[1] for pitch in removed_pitches
    ]

    additional_marks = []
    pitch_ints = []
    accidentals = []

    # Check for accidentally-altered tones (same step, different accidental)
    for step in range(8):
        if step in added_pitch_ints and step in removed_pitch_ints:
            added_idx = added_pitch_ints.index(step)
            pitch_ints.append(added_pitch_ints[added_idx])
            accidentals.append(added_accidentals[added_idx])
            additional_marks.append("")

            del added_accidentals[added_idx]
            del added_pitch_ints[added_idx]
            del removed_pitch_ints[removed_pitch_ints.index(step)]

    # Check for specific common replacements
    for added, removed, if_sharp, if_natural, if_flat in [
        (6, 7, "", "^", "^"),
        (6, 5, "v", "", ""),
        (4, 5, "", "^", "^"),
        (4, 3, "v", "", ""),
        (2, 3, "", "^", "^"),
        (2, 1, "v", "", ""),
    ]:
        if added in added_pitch_ints and removed in removed_pitch_ints:
            added_idx = added_pitch_ints.index(added)
            pitch_ints.append(added_pitch_ints[added_idx])
            accidentals.append(added_accidentals[added_idx])

            additional_marks.append(
                if_sharp
                if "#" in accidentals[-1]
                else if_flat
                if "b" in accidentals[-1]
                else if_natural
            )

            del added_accidentals[added_idx]
            del added_pitch_ints[added_idx]
            del removed_pitch_ints[removed_pitch_ints.index(removed)]

    # Add and remove remaining pitches ad hoc
    if len(added_pitch_ints) != 0:
        for accidental, pitch_int in zip(added_accidentals, added_pitch_ints):
            if pitch_int == 2:
                pitch_int = 9

            accidentals.append(accidental)
            pitch_ints.append(pitch_int)

            if pitch_int in removed_pitch_ints:
                additional_marks.append("")
                del removed_pitch_ints[removed_pitch_ints.index(pitch_int)]
            else:
                additional_marks.append("" if pitch_int == 9 else "+")

    if len(removed_pitch_ints) != 0:
        for pitch_int in removed_pitch_ints:
            accidentals.append("")
            pitch_ints.append(pitch_int)
            additional_marks.append("-")

    pitch_strings = [
        f"{additional_marks[i]}{accidentals[i]}{pitch_ints[i]}" for i in np.argsort(pitch_ints)
    ]

    return "(" + "".join(reversed(pitch_strings)) + ")"


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


def get_interval_from_scale_degree(
    scale_degree: str, accidentals_prefixed: bool, mode: KeyMode, pitch_type: PitchType
) -> int:
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
    accidental_adjustment, scale_degree = get_accidental_adjustment(
        scale_degree, in_front=accidentals_prefixed
    )

    interval = hc.SCALE_INTERVALS[mode][pitch_type][hc.SCALE_DEGREE_TO_NUMBER[scale_degree]]
    interval += accidental_adjustment * hc.ACCIDENTAL_ADJUSTMENT[pitch_type]

    return interval


def get_scale_degree_from_interval(interval: int, mode: KeyMode, pitch_type: PitchType) -> str:
    """
    Get a scale degree from a TPC or MIDI interval.

    Parameters
    ----------
    interval : int
        The integer representation of an interval, relative to the key tonic.
    mode : KeyMode
        The mode of the current key.
    pitch_type : PitchType
        The pitch type of the given interval.

    Returns
    -------
    scale_degree : str
        The roman numeral of the scale degree of the given interval, pre-pended with accidentals.
        For MIDI intervals, all accidentals will be sharps.
    """
    scale_intervals = hc.SCALE_INTERVALS[mode][pitch_type]
    range_min = min(scale_intervals)
    range_max = max(scale_intervals)

    if pitch_type == PitchType.MIDI:
        interval = interval % hc.NUM_PITCHES[PitchType.MIDI]

        num_sharps = 0
        while interval not in scale_intervals:
            num_sharps += 1
            interval = (interval - 1) % hc.NUM_PITCHES[PitchType.MIDI]

        return "#" * num_sharps + hc.NUMBER_TO_SCALE_DEGREE[scale_intervals.index(interval)]

    num_sharps = 0
    num_flats = 0

    while interval < range_min:
        num_flats += 1
        interval += hc.ACCIDENTAL_ADJUSTMENT[PitchType.TPC]

    while interval > range_max:
        num_sharps += 1
        interval -= hc.ACCIDENTAL_ADJUSTMENT[PitchType.TPC]

    return (
        "#" * num_sharps
        + "b" * num_flats
        + hc.NUMBER_TO_SCALE_DEGREE[scale_intervals.index(interval)]
    )


def transpose_pitch(
    pitch: int,
    interval: int,
    pitch_type: PitchType,
    ignore_range: bool = False,
) -> int:
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
    ignore_range : bool
        True to ignore any errors resulting from a TPC pitch being transposed out of the legal
        TPC range. Useful if the value represents something like a TPC interval.

    Returns
    -------
    pitch : int
        The given pitch, transposed by the given interval.
    """
    if pitch_type == PitchType.MIDI:
        return (pitch + interval) % hc.NUM_PITCHES[PitchType.MIDI]
    pitch = pitch + interval

    if not ignore_range and (pitch < 0 or pitch >= hc.NUM_PITCHES[PitchType.TPC]):
        raise ValueError(
            f"pitch_type is TPC but transposed pitch {pitch} lies outside of TPC " "range."
        )

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
    except Exception as exception:
        raise ValueError(
            f"{str} is not a valid figured bass symbol for detecting inversions."
        ) from exception


def get_figbass_string(inversion: int, chord_type: ChordType) -> str:
    """
    Get the figured bass string for a particular inversion of the given chord type.

    Parameters
    ----------
    inversion : int
        The inversion.
    chord_type : ChordType
        The chord type. Used to disambiguate triads and tetrads.

    Returns
    -------
    figbass : str
        The figured bass string for the given inversion.
    """
    if inversion is None:
        return ""

    is_triad = len(hc.CHORD_PITCHES[PitchType.TPC][chord_type]) == 3

    return hc.FIGBASS_STRINGS[is_triad][inversion]


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
    except Exception as exception:
        raise ValueError(f"String type {chord_string} not recognized.") from exception


def get_chord_string(chord_type: ChordType, for_labels: bool) -> str:
    """
    Get a chord type string from a given ChordType.

    Parameters
    ----------
    chord_type : ChordType
        A ChordType to convert to a string.

    for_labels : bool
        True if the returned string is for labels. False if it should be directly
        readable as a type. The main difference is that for labels, "7" is never
        included, since that is instead included in the figured bass string.

    Returns
    -------
    chord_type_str : str
        The string representation of the given ChordType.
    """
    return (
        hc.CHORD_TYPE_TO_STRING_LABELS[chord_type]
        if for_labels
        else hc.CHORD_TYPE_TO_STRING_READABLE[chord_type]
    )


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
    if pitch_type == PitchType.MIDI and "/" in pitch_string:
        pitch_string = pitch_string.split("/")[0]

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
    accidental_string = "#"
    while pitch < min(hc.PITCH_TO_STRING[PitchType.TPC].keys()):
        pitch += hc.TPC_NATURAL_PITCHES
        accidental -= 1
    while pitch > max(hc.PITCH_TO_STRING[PitchType.TPC].keys()):
        pitch -= hc.TPC_NATURAL_PITCHES
        accidental += 1

    if accidental < 0:
        accidental_string = "b"
        accidental = -accidental

    return hc.PITCH_TO_STRING[PitchType.TPC][pitch] + (accidental_string * accidental)
