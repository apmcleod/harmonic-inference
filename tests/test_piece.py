"""Tests for piece.py"""
from fractions import Fraction

import pandas as pd
import numpy as np

from harmonic_inference.data.data_types import KeyMode, PitchType
from harmonic_inference.data.piece import Note, Key, Chord, ScorePiece, get_reduction_mask
import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.rhythmic_utils as ru
import harmonic_inference.utils.harmonic_utils as hu


def test_note_from_series():
    def check_equals(note_dict, note, measures_df, pitch_type):
        assert pitch_type == note.pitch_type
        if pitch_type == PitchType.MIDI:
            assert (note_dict['midi'] % hc.NUM_PITCHES[PitchType.MIDI]) == note.pitch_class
        else:
            assert note.pitch_class == note_dict['tpc'] + hc.TPC_C
        assert note.octave == note_dict['midi'] // hc.NUM_PITCHES[PitchType.MIDI]
        assert note.onset == (note_dict['mc'], note_dict['onset'])
        assert note.offset == (note_dict['offset_mc'], note_dict['offset_beat'])
        assert note.duration == note_dict['duration']
        assert note.onset_level == ru.get_metrical_level(
            note_dict['onset'],
            measures_df.loc[measures_df['mc'] == note_dict['mc']].squeeze(),
        )
        assert note.offset_level == ru.get_metrical_level(
            note_dict['offset_beat'],
            measures_df.loc[measures_df['mc'] == note_dict['offset_mc']].squeeze(),
        )

    note_dict = {
        'midi': 50,
        'tpc': 5,
        'mc': 1,
        'onset': Fraction(1, 2),
        'offset_mc': 2,
        'offset_beat': Fraction(3, 4),
        'duration': Fraction(5, 6),
    }

    key_values = {
        'midi': range(127),
        'tpc': range(-hc.TPC_C, hc.TPC_C),
        'mc': range(3),
        'onset': [i * Fraction(1, 2) for i in range(3)],
        'offset_mc': range(3),
        'offset_beat': [i * Fraction(1, 2) for i in range(3)],
        'duration': [i * Fraction(1, 2) for i in range(3)],
    }

    measures_df = pd.DataFrame({
        'mc': list(range(10)),
        'timesig': '12/8'
    })

    for key, values in key_values.items():
        for value in values:
            note_dict[key] = value
            note_series = pd.Series(note_dict)
            note = Note.from_series(note_series, measures_df, PitchType.MIDI)
            check_equals(note_dict, note, measures_df, PitchType.MIDI)
            note = Note.from_series(note_series, measures_df, PitchType.TPC)
            check_equals(note_dict, note, measures_df, PitchType.TPC)

    note_dict['tpc'] = hc.NUM_PITCHES[PitchType.TPC] - hc.TPC_C
    assert Note.from_series(pd.Series(note_dict), measures_df, PitchType.TPC) is None
    note_dict['tpc'] = 0 - hc.TPC_C - 1
    assert Note.from_series(pd.Series(note_dict), measures_df, PitchType.TPC) is None


def test_chord_from_series():
    def check_equals(chord_dict, chord, measures_df, pitch_type, key):
        assert chord.pitch_type == pitch_type
        assert chord.chord_type == hu.get_chord_type_from_string(chord_dict['chord_type'])
        assert chord.inversion == hu.get_chord_inversion(chord_dict['figbass'])
        assert chord.onset == (chord_dict['mc'], chord_dict['onset'])
        assert chord.offset == (chord_dict['mc_next'], chord_dict['onset_next'])
        assert chord.duration == chord_dict['duration']
        assert chord.onset_level == ru.get_metrical_level(
            chord_dict['onset'],
            measures_df.loc[measures_df["mc"] == chord_dict["mc"]].squeeze(),
        )
        assert chord.offset_level == ru.get_metrical_level(
            chord_dict['onset_next'],
            measures_df.loc[measures_df["mc"] == chord_dict["mc_next"]].squeeze(),
        )

        root = chord_dict['root']
        bass = chord_dict['bass_note']
        if pitch_type == PitchType.MIDI:
            root = hu.tpc_interval_to_midi_interval(root)
            bass = hu.tpc_interval_to_midi_interval(bass)
        assert chord.root == hu.transpose_pitch(key.local_tonic, root, pitch_type)
        assert chord.bass == hu.transpose_pitch(key.local_tonic, bass, pitch_type)

    chord_dict = {
        'numeral': 'III',
        'root': 5,
        'bass_note': 5,
        'chord_type': 'M',
        'figbass': '',
        'globalkey': 'A',
        'globalkey_is_minor': False,
        'localkey': 'iii',
        'localkey_is_minor': True,
        'relativeroot': pd.NA,
        'offset_mc': 2,
        'offset_beat': Fraction(3, 4),
        'duration': Fraction(5, 6),
        'mc': 1,
        'onset': Fraction(1, 2),
        'mc_next': 2,
        'onset_next': Fraction(3, 4),
    }

    key_values = {
        'root': range(-7, 7),
        'bass_note': range(-7, 7),
        'chord_type': hc.STRING_TO_CHORD_TYPE.keys(),
        'figbass': hc.FIGBASS_INVERSIONS.keys(),
        'mc': range(3),
        'onset': [i * Fraction(1, 2) for i in range(3)],
        'mc_next': range(3),
        'onset_next': [i * Fraction(1, 2) for i in range(3)],
        'duration': [i * Fraction(1, 2) for i in range(3)],
    }

    measures_df = pd.DataFrame({
        'mc': list(range(10)),
        'timesig': '12/8'
    })

    for key, values in key_values.items():
        for value in values:
            chord_dict[key] = value
            chord_series = pd.Series(chord_dict)
            for pitch_type in PitchType:
                chord = Chord.from_series(chord_series, measures_df, pitch_type)
                local_key = Key.from_series(chord_series, pitch_type)
                check_equals(chord_dict, chord, measures_df, pitch_type, local_key)

    # @none returns None
    for numeral in ['@none', pd.NA]:
        chord_dict['numeral'] = numeral
        chord_series = pd.Series(chord_dict)
        for pitch_type in PitchType:
            assert Chord.from_series(chord_series, measures_df, pitch_type) is None
    chord_dict['numeral'] = 'III'

    # Bad key returns None
    chord_dict['localkey'] = 'Error'
    chord_series = pd.Series(chord_dict)
    for pitch_type in PitchType:
        assert Chord.from_series(chord_series, measures_df, pitch_type) is None
    chord_dict['localkey'] = 'iii'

    # Bad relativeroot is not ok
    chord_dict['relativeroot'] = 'Error'
    chord_series = pd.Series(chord_dict)
    for pitch_type in PitchType:
        assert Chord.from_series(chord_series, measures_df, pitch_type) is None


def test_key_from_series():
    def get_relative(global_tonic, global_mode, relative_numeral, pitch_type):
        """Get the relative key tonic of a numeral in a given global key."""
        local_interval = hu.get_interval_from_numeral(relative_numeral, global_mode, pitch_type)
        local_tonic = hu.transpose_pitch(global_tonic, local_interval, pitch_type)
        return local_tonic

    def check_equals(key_dict, key, pitch_type):
        assert key.tonic_type == pitch_type

        # Check mode
        if not pd.isnull(key_dict['relativeroot']):
            final_root = key_dict['relativeroot'].split('/')[0]
            assert (
                key.relative_mode == KeyMode.MINOR if final_root[-1].islower() else KeyMode.MAJOR
            )
        else:
            assert key.relative_mode == key.local_mode

        assert key.local_mode == KeyMode.MINOR if key_dict['localkey_is_minor'] else KeyMode.MAJOR

        # Check tonic
        if not pd.isnull(key_dict['relativeroot']):
            # We can rely on this non-relative local key. It is checked below
            key_tonic = key.local_tonic
            key_mode = key.local_mode
            for relative_numeral in reversed(key_dict['relativeroot'].split('/')):
                key_tonic = get_relative(key_mode, relative_numeral, pitch_type)
                key_mode = KeyMode.MINOR if relative_numeral[-1].islower() else KeyMode.MAJOR
            assert key_tonic == key.relative_tonic
            assert key_mode == key.relative_mode
        else:
            assert key.relative_tonic == key.local_tonic
            assert key.relative_mode == key.local_mode

        global_key_tonic = hu.get_pitch_from_string(key_dict['globalkey'], pitch_type)
        global_mode = KeyMode.MINOR if key_dict['globalkey_is_minor'] else KeyMode.MAJOR
        local_key_tonic = get_relative(
            global_key_tonic, global_mode, key_dict['localkey'], pitch_type
        )
        local_key_mode = KeyMode.MINOR if key_dict['localkey_is_minor'] else KeyMode.MAJOR
        assert key.local_tonic == local_key_tonic
        assert key.local_mode == local_key_mode

    key_dict = {
        'globalkey': 'A',
        'globalkey_is_minor': False,
        'localkey': 'iii',
        'localkey_is_minor': True,
        'relativeroot': pd.NA,
    }

    # A few ad-hoc
    key_tpc = Key.from_series(pd.Series(key_dict), PitchType.TPC)
    key_midi = Key.from_series(pd.Series(key_dict), PitchType.MIDI)
    assert key_tpc.local_mode == KeyMode.MINOR == key_midi.local_mode
    assert key_tpc.local_tonic == hc.TPC_C + hc.ACCIDENTAL_ADJUSTMENT[PitchType.TPC]
    assert key_midi.local_tonic == 1

    key_dict['globalkey_is_minor'] = True
    key_tpc = Key.from_series(pd.Series(key_dict), PitchType.TPC)
    key_midi = Key.from_series(pd.Series(key_dict), PitchType.MIDI)
    assert key_tpc.local_mode == KeyMode.MINOR == key_midi.local_mode
    assert key_tpc.local_tonic == hc.TPC_C
    assert key_midi.local_tonic == 0

    key_dict['localkey_is_minor'] = False
    key_tpc = Key.from_series(pd.Series(key_dict), PitchType.TPC)
    key_midi = Key.from_series(pd.Series(key_dict), PitchType.MIDI)
    assert key_tpc.local_mode == KeyMode.MAJOR == key_midi.local_mode
    assert key_tpc.local_tonic == hc.TPC_C
    assert key_midi.local_tonic == 0

    key_dict['localkey'] = 'ii'
    key_tpc = Key.from_series(pd.Series(key_dict), PitchType.TPC)
    key_midi = Key.from_series(pd.Series(key_dict), PitchType.MIDI)
    assert key_tpc.local_mode == KeyMode.MAJOR == key_midi.local_mode
    assert key_tpc.local_tonic == hc.TPC_C + 5
    assert key_midi.local_tonic == 11

    key_dict['globalkey'] = 'C'
    key_tpc = Key.from_series(pd.Series(key_dict), PitchType.TPC)
    key_midi = Key.from_series(pd.Series(key_dict), PitchType.MIDI)
    assert key_tpc.local_mode == KeyMode.MAJOR == key_midi.local_mode
    assert key_tpc.local_tonic == hc.TPC_C + 2
    assert key_midi.local_tonic == 2

    key_values = {
        'globalkey': ['A', 'B#', 'Bb', 'C'],
        'globalkey_is_minor': [False, True],
        'localkey': ['iii', 'i', 'bV'],
        'localkey_is_minor': [False, True],
    }

    for key, values in key_values.items():
        for value in values:
            key_dict[key] = value
            key_series = pd.Series(key_dict)
            for pitch_type in PitchType:
                key_obj = Key.from_series(key_series, pitch_type)
                check_equals(key_dict, key_obj, pitch_type)

    # Try with localkey minor, relatives major
    key_dict['localkey_is_minor'] = True
    for root_symbol in ['I', 'bII', '#III', 'V', 'bVI', 'bVII']:
        initial_interval_tpc = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MINOR, PitchType.TPC
        )
        initial_interval_midi = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MINOR, PitchType.MIDI
        )
        relative_interval_tpc = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MAJOR, PitchType.TPC
        )
        relative_interval_midi = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MAJOR, PitchType.MIDI
        )
        for repeats in range(1, 4):
            if repeats == 1:
                relative_root = root_symbol
            else:
                relative_root = '/'.join([root_symbol] * repeats)
            interval_tpc = initial_interval_tpc + (repeats - 1) * relative_interval_tpc
            interval_midi = initial_interval_midi + (repeats - 1) * relative_interval_midi
            key_series = pd.Series(key_dict)
            key_series['relativeroot'] = pd.NA
            old_key_tpc = Key.from_series(key_series, PitchType.TPC)
            old_key_midi = Key.from_series(key_series, PitchType.MIDI)
            key_series['relativeroot'] = relative_root
            key_tpc = Key.from_series(key_series, PitchType.TPC)
            key_midi = Key.from_series(key_series, PitchType.MIDI)
            target_tpc = old_key_tpc.relative_tonic + interval_tpc
            if target_tpc < 0 or target_tpc >= hc.NUM_PITCHES[PitchType.TPC]:
                assert key_tpc is None
            else:
                assert key_tpc.relative_tonic == target_tpc
            target_midi = (
                (old_key_midi.relative_tonic + interval_midi) % hc.NUM_PITCHES[PitchType.MIDI]
            )
            assert key_midi.relative_tonic == target_midi

    # Try with localkey major, relatives minor
    key_dict['localkey_is_minor'] = False
    for root_symbol in ['i', 'bii', '#iii', 'v', 'bvi', 'bvii']:
        initial_interval_tpc = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MAJOR, PitchType.TPC
        )
        initial_interval_midi = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MAJOR, PitchType.MIDI
        )
        relative_interval_tpc = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MINOR, PitchType.TPC
        )
        relative_interval_midi = hu.get_interval_from_numeral(
            root_symbol, KeyMode.MINOR, PitchType.MIDI
        )
        for repeats in range(1, 4):
            if repeats == 1:
                relative_root = root_symbol
            else:
                relative_root = '/'.join([root_symbol] * repeats)
            interval_tpc = initial_interval_tpc + (repeats - 1) * relative_interval_tpc
            interval_midi = initial_interval_midi + (repeats - 1) * relative_interval_midi
            key_series = pd.Series(key_dict)
            key_series['relativeroot'] = pd.NA
            old_key_tpc = Key.from_series(key_series, PitchType.TPC)
            old_key_midi = Key.from_series(key_series, PitchType.MIDI)
            key_series['relativeroot'] = relative_root
            key_tpc = Key.from_series(key_series, PitchType.TPC)
            key_midi = Key.from_series(key_series, PitchType.MIDI)
            target_tpc = old_key_tpc.relative_tonic + interval_tpc
            if target_tpc < 0 or target_tpc >= hc.NUM_PITCHES[PitchType.TPC]:
                assert key_tpc is None
            else:
                assert key_tpc.relative_tonic == target_tpc
            target_midi = (
                (old_key_midi.relative_tonic + interval_midi) % hc.NUM_PITCHES[PitchType.MIDI]
            )
            assert key_midi.relative_tonic == target_midi


def test_score_piece():
    measures_df = pd.DataFrame({
        'mc': list(range(20)),
        'timesig': '12/8',
        'act_dur': Fraction(12, 8),
        'offset': Fraction(0),
        'next': list(range(1, 20)) + [-1],
    })

    note_dict = {
        'midi': 50,
        'tpc': 5,
        'mc': range(10),
        'onset': Fraction(1, 2),
        'offset_mc': range(1, 11),
        'offset_beat': Fraction(1, 2),
        'duration': Fraction(1),
    }
    note_df = pd.DataFrame(note_dict)
    notes = [
        Note.from_series(note_row, measures_df, PitchType.TPC)
        for _, note_row in note_df.iterrows()
    ]

    # TODO: Some of these have repeated chords crossing key boundaries.
    # Check if this happens and remove.
    chord_dict = {
        'numeral': ['III', 'III', '@none', 'III', 'IV', 'IV', '@none'],
        'root': 5,
        'bass_note': 5,
        'chord_type': 'M',
        'figbass': '',
        'globalkey': 'A',
        'globalkey_is_minor': False,
        'localkey': ['iii', 'iii', pd.NA, 'III', 'III', 'I', pd.NA],
        'localkey_is_minor': [True, True, pd.NA, False, False, False, pd.NA],
        'relativeroot': [pd.NA, pd.NA, pd.NA, pd.NA, 'V', 'V', pd.NA],
        'duration': Fraction(2),
        'mc': [0, 2, 4, 5, 6, 8, 10],
        'onset': Fraction(1, 2),
        'mc_next': [2, 5, 5, 6, 8, 10, 12],
        'onset_next': Fraction(1, 2),
    }
    chord_df = pd.DataFrame(chord_dict)
    chord_df.drop(labels=2, axis='index')
    chords = [
        Chord.from_series(chord_row, measures_df, PitchType.TPC)
        for _, chord_row in chord_df.iterrows()
    ]
    not_none_chords = np.array([c for c in chords if c is not None])
    mask = get_reduction_mask(not_none_chords)
    correct_chords = []
    for m, chord in zip(mask, not_none_chords):
        if m:
            correct_chords.append(chord)
        else:
            correct_chords[-1].merge_with(chord)
    not_none_chords = correct_chords

    keys = [Key.from_series(chord_row, PitchType.TPC) for _, chord_row in chord_df.iterrows()]
    not_none_keys = [k for k in keys if k is not None]
    unique_keys = [not_none_keys[0]] + [
        k for k, k_prev in zip(not_none_keys[1:], not_none_keys[:-1]) if k != k_prev
    ]
    unique_keys = [unique_keys[0], unique_keys[3]]

    piece = ScorePiece(note_df, chord_df, measures_df)

    assert all(piece.get_inputs() == notes)
    assert all(piece.get_chords() == not_none_chords)
    assert all(piece.get_keys() == unique_keys)
    assert all(piece.get_chord_change_indices() == [0, 8])
    assert all(piece.get_key_change_indices() == [0, 1])

    inputs = piece.get_chord_note_inputs(window=2)
    assert np.sum(inputs[0][:2]) == 0
    assert all(
        inputs[0][2] ==
        notes[0].to_vec(
            not_none_chords[0].onset,
            not_none_chords[0].offset,
            not_none_chords[0].duration,
            measures_df,
            (notes[0].octave, notes[0].pitch_class)
        )
    )
    assert np.sum(inputs[-1][-2:]) == 0
    assert all(
        inputs[-1][-3] ==
        notes[-1].to_vec(
            not_none_chords[-1].onset,
            not_none_chords[-1].offset,
            not_none_chords[-1].duration,
            measures_df,
            (notes[-1].octave, notes[-1].pitch_class),
        )
    )
    assert all(
        inputs[1][2] ==
        notes[8].to_vec(
            not_none_chords[1].onset,
            not_none_chords[1].offset,
            not_none_chords[1].duration,
            measures_df,
            (notes[8].octave, notes[8].pitch_class)
        )
    )
