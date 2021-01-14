"""Functions for decoding data vectors."""
from typing import Tuple

import numpy as np

from harmonic_inference.data.chord import Chord, get_chord_vector_length
from harmonic_inference.data.data_types import ChordType, KeyMode, PitchType
from harmonic_inference.data.key import Key, get_key_change_vector_length
from harmonic_inference.data.note import Note, get_note_vector_length
from harmonic_inference.utils.harmonic_constants import (
    MAX_RELATIVE_TPC,
    MIN_KEY_CHANGE_INTERVAL_TPC,
    MIN_RELATIVE_TPC,
    NUM_PITCHES,
    RELATIVE_TPC_EXTRA,
)
from harmonic_inference.utils.harmonic_utils import get_pitch_string


def decode_note_vector(note_vector: np.array) -> Note:
    """
    Print out information about the given note vector.

    Parameters
    ----------
    note_vector : np.array
        A note vector to decode.

    Returns
    -------
    note : Note
        The note, decoded from the given vector.
    """
    # Infer pitch_type from vector length
    pitch_type = None
    for check_pitch_type in PitchType:
        if len(note_vector) == get_note_vector_length(check_pitch_type):
            pitch_type = check_pitch_type
            break

    if pitch_type is None:
        raise ValueError("Key change vector is not a valid length for any PitchType.")

    pitch = np.arange(NUM_PITCHES[pitch_type])[np.where(note_vector == 1)[0][0]]

    current_idx = NUM_PITCHES[pitch_type]
    num_octaves = 127 // NUM_PITCHES[PitchType.MIDI]
    octave_vec = note_vector[current_idx : current_idx + num_octaves]
    octave = np.where(octave_vec == 1)[0][0]

    current_idx += num_octaves
    onset_level_vector = note_vector[current_idx : current_idx + 4]
    onset_level = np.where(onset_level_vector == 1)[0][0]

    current_idx += 4
    offset_level_vector = note_vector[current_idx : current_idx + 4]
    offset_level = np.where(offset_level_vector == 1)[0][0]

    current_idx += 4
    mc, onset, dur, dur_from_prev, dur_to_next = note_vector[current_idx : current_idx + 3]
    is_min = note_vector[-1] == 1

    print(f"Pitch: {get_pitch_string(pitch, pitch_type)}, octave {octave}")
    print(f"Onset level: {onset_level}")
    print(f"Offset level: {offset_level}")
    print(f"Onset position: ({mc}, {onset})")
    print(f"Duration: {dur} (from prev: {dur_from_prev}, to_next: {dur_to_next}")
    print(f"Is{' not ' if is_min else ' '}lowest pitch")

    return Note(pitch, octave, onset, onset_level, dur, None, offset_level, pitch_type)


def decode_key_change_vector(key_change_vector: np.array) -> Key:
    """
    Print out information about the given key change vector.

    Parameters
    ----------
    key_vector : np.array
        A key vector to decode.

    Returns
    -------
    key : Key
        The key object, decoded from the given vector.
    """
    # Infer pitch_type from vector length
    pitch_type = None
    for check_pitch_type in PitchType:
        if len(key_change_vector) == get_key_change_vector_length(check_pitch_type, one_hot=False):
            pitch_type = check_pitch_type
            break

    if pitch_type is None:
        raise ValueError("Key change vector is not a valid length for any PitchType.")

    relative_tonic = np.where(key_change_vector == 1)[0][0]
    if pitch_type == PitchType.TPC:
        relative_tonic += MIN_KEY_CHANGE_INTERVAL_TPC

    # Absolute mode of next key
    key_mode = np.array(KeyMode)[key_change_vector[-2:] == 1]

    print(f"Relative tonic: {relative_tonic}, Mode: {key_mode}")

    return Key(relative_tonic, None, None, key_mode, None, None, pitch_type)


def decode_chord_vector(
    chord_vector: np.array,
    pad: bool = False,
) -> Chord:
    """
    Print out information about the given chord vector.

    Parameters
    ----------
    chord_vector : np.array
        A chord vector to decode.
    pad : bool
        If True and pitch_type is PitchType.TPC, the given vector allows for a padded
        pitch range for root and bass pitches.

    Returns
    -------
    chord : Chord
        A Chord object, decoded from the given vector.
    """
    # Infer pitch_type from vector length
    pitch_type = None
    for check_pitch_type in PitchType:
        if len(chord_vector) == get_chord_vector_length(check_pitch_type, one_hot=False, pad=pad):
            pitch_type = check_pitch_type
            break

    if pitch_type is None:
        raise ValueError("Key change vector is not a valid length for any PitchType.")

    if pitch_type == PitchType.MIDI:
        num_pitches = NUM_PITCHES[PitchType.MIDI]
        to_add = 0

    elif pitch_type == PitchType.TPC:
        num_pitches = MAX_RELATIVE_TPC - MIN_RELATIVE_TPC
        to_add = MIN_RELATIVE_TPC
        if pad:
            num_pitches += 2 * RELATIVE_TPC_EXTRA
            to_add -= RELATIVE_TPC_EXTRA

    else:
        raise ValueError(f"Pitch Type {pitch_type} not recognized.")

    relative_root = np.where(chord_vector == 1)[0][0]
    if pitch_type == PitchType.TPC:
        relative_root += to_add

    chord_type_vector = chord_vector[num_pitches : num_pitches + len(ChordType)]
    chord_type = np.array(ChordType)[chord_type_vector == 1][0]

    bass_vector = chord_vector[num_pitches + len(ChordType) : 2 * num_pitches + len(ChordType)]
    relative_bass = np.where(bass_vector == 1)[0][0]
    if pitch_type == PitchType.TPC:
        relative_bass += to_add

    current_idx = 2 * num_pitches + len(ChordType)
    inversion_vector = chord_vector[current_idx : current_idx + 4]
    inversion = np.where(inversion_vector == 1)[0][0]

    current_idx += 4
    onset_level_vector = chord_vector[current_idx : current_idx + 4]
    onset_level = np.where(onset_level_vector == 1)[0][0]

    current_idx += 4
    offset_level_vector = chord_vector[current_idx : current_idx + 4]
    offset_level = np.where(offset_level_vector == 1)[0][0]

    is_major = chord_vector[-1] == 1
    key_mode = KeyMode.MAJOR if is_major else KeyMode.MINOR

    print(f"Relative root: {relative_root}")
    print(f"Chord type: {chord_type}")
    print(f"Relative bass note: {relative_bass}")
    print(f"Inversion: {inversion}")
    print(f"Onset level: {onset_level}")
    print(f"Offset_level: {offset_level}")
    print(f"Key mode: {key_mode}")

    return Chord(
        relative_root,
        relative_bass,
        0,
        key_mode,
        chord_type,
        inversion,
        None,
        onset_level,
        None,
        offset_level,
        None,
        pitch_type,
    )


def decode_chord_and_key_change_vector(
    vector: np.array,
    root_type: PitchType = None,
    tonic_type: PitchType = None,
    pad: bool = False,
) -> Tuple[Chord, Key]:
    """
    Print out information about the given relative chord (with optional key change) vector.

    Parameters
    ----------
    vector : np.array
        A relative chord (and optional key change) vector.
    root_type : PitchType
        The pitch type used to store the chord vector's root and bass pitches. Either this or
        tonic_type is required.
    tonic_type : PitchType
        The pitch type used to store the key vector's tonic pitch. Either this or tonic_type
        is required.
    pad : bool
        If True and root_type is PitchType.TPC, additional padded pitches are used to store
        the chord's root and bass ptiches.

    Returns
    -------
    chord : Chord
        The chord, decoded from the given vector.
    key : Key
        If this is a key change, the key decoded from the given vector. Otherwise, None.
    """
    if root_type is None and tonic_type is None:
        raise ValueError("Either root_type or tonic_type is required.")

    # Infer root_type or tonic_type from vector length
    if root_type is not None:
        chord_vector_length = get_chord_vector_length(
            root_type,
            one_hot=False,
            relative=True,
            pad=pad,
        )

    else:
        key_vector_length = get_key_change_vector_length(tonic_type, one_hot=False)
        chord_vector_length = len(vector) - 1 - key_vector_length

    chord = decode_chord_vector(
        vector[:chord_vector_length],
        pad=pad,
    )

    is_key_change = vector[-1]
    key_vector = vector[chord_vector_length:-1]
    key = None

    if is_key_change:
        print("Key change:")
        key = decode_key_change_vector(key_vector)
    else:
        print("No key change")
        if np.sum(key_vector) > 0:
            raise ValueError("No key change, but key change vector is not empty.")

    return chord, key
