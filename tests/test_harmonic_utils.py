"""Tests for harmonic_utils.py"""
import itertools
import pytest

import pandas as pd

from harmonic_inference.data.data_types import KeyMode, ChordType, PitchType
import harmonic_inference.utils.harmonic_utils as hu
import harmonic_inference.utils.harmonic_constants as hc


def test_get_accidental_adjustment():
    for accidental, diff in zip(['#', 'b'], [1, -1]):
        for count in range(10):
            for root in ['b', 'B', 'V', '1']:
                # Front
                input_string = accidental * count + root
                adj_out, out_string = hu.get_accidental_adjustment(input_string, in_front=True)

                assert adj_out == count * diff, (
                    f"Incorrect adjustment ({adj_out}) for input {input_string}"
                )
                assert out_string == root, (
                    f"Incorrect string out ({out_string}) for input {input_string}"
                )

                # Back
                input_string = root + accidental * count
                adj_out, out_string = hu.get_accidental_adjustment(input_string, in_front=False)

                assert adj_out == count * diff, (
                    f"Incorrect adjustment ({adj_out}) for input {input_string}"
                )
                assert out_string == root, (
                    f"Incorrect string out ({out_string}) for input {input_string}"
                )


def test_transpose_chord_vector():
    midi_vector = [0] * hc.NUM_PITCHES[PitchType.MIDI]
    tpc_vector = [0] * hc.NUM_PITCHES[PitchType.TPC]
    for idx in range(0, len(midi_vector), 2):
        midi_vector[idx] = 1
    for idx in range(0, len(tpc_vector), 3):
        tpc_vector[idx] = 1

    for interval in range(-50, 50):
        midi_output = hu.transpose_chord_vector(midi_vector, interval, PitchType.MIDI)
        tpc_output = hu.transpose_chord_vector(tpc_vector, interval, PitchType.TPC)

        for idx, val in enumerate(midi_vector):
            assert midi_output[(idx + interval) % hc.NUM_PITCHES[PitchType.MIDI]] == val

        checked = []
        for idx, val in enumerate(tpc_vector):
            index = idx + interval
            if 0 <= index < hc.NUM_PITCHES[PitchType.TPC]:
                checked.append(index)
                assert tpc_output[index] == val
        for idx, val in enumerate(tpc_output):
            if idx not in checked:
                assert val == 0


def test_get_vector_from_chord_type():
    chord_types = [ChordType.MAJOR,
                   ChordType.MINOR,
                   ChordType.DIMINISHED,
                   ChordType.AUGMENTED,
                   ChordType.MIN_MIN7,
                   ChordType.MAJ_MIN7,
                   ChordType.MAJ_MAJ7,
                   ChordType.MIN_MAJ7,
                   ChordType.DIM7,
                   ChordType.HALF_DIM7,
                   ChordType.AUG_MIN7,
                   ChordType.AUG_MAJ7]
    chord_vectors_midi = [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]
    chord_vectors_tpc = [['C', 'E', 'G'],
                         ['C', 'Eb', 'G'],
                         ['C', 'Eb', 'Gb'],
                         ['C', 'E', 'G#'],
                         ['C', 'Eb', 'G', 'Bb'],
                         ['C', 'E', 'G', 'Bb'],
                         ['C', 'E', 'G', 'B'],
                         ['C', 'Eb', 'G', 'B'],
                         ['C', 'Eb', 'Gb', 'Bbb'],
                         ['C', 'Eb', 'Gb', 'Bb'],
                         ['C', 'E', 'G#', 'Bb'],
                         ['C', 'E', 'G#', 'B']]
    for chord_type, midi_vector, tpc_vector in zip(chord_types, chord_vectors_midi,
                                                   chord_vectors_tpc):
        out_vector = hu.get_vector_from_chord_type(chord_type, PitchType.MIDI)
        assert out_vector.dtype == int
        assert all(midi_vector == out_vector), (
            f"Chord vector incorrect for MIDI and chord type {chord_type}"
        )

        out_vector_tpc = hu.get_vector_from_chord_type(chord_type, PitchType.TPC)
        assert out_vector_tpc.dtype == int
        tpc_vector_one_hot = [0] * hc.NUM_PITCHES[PitchType.TPC]
        for pitch_string in tpc_vector:
            tpc_vector_one_hot[hu.get_pitch_from_string(pitch_string, PitchType.TPC)] = 1
        assert all(tpc_vector_one_hot == out_vector_tpc), (
            f"Chord vector incorrect for TPC and chord type {chord_type}"
        )


def test_get_interval_from_numeral():
    for acc, adj in zip(['bbb', 'bb', 'b', '', '#', '##', '###'], [-3, -2, -1, 0, 1, 2, 3]):
        for key_mode, semitones, tpc in zip([KeyMode.MAJOR, KeyMode.MINOR],
                                            [[0, 2, 4, 5, 7, 9, 11],
                                             [0, 2, 3, 5, 7, 8, 10]],
                                            [[0, 2, 4, -1, 1, 3, 5],
                                             [0, 2, -3, -1, 1, -4, -2]]):
            for numeral, index in zip(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], range(7)):
                numeral = acc + numeral
                out_semis = hu.get_interval_from_numeral(numeral, key_mode, PitchType.MIDI)
                assert out_semis == semitones[index] + adj, (
                    f"Output semitones incorrect for inputs {(numeral, key_mode, PitchType.MIDI)}"
                )
                assert out_semis == hu.get_interval_from_numeral(numeral, key_mode, PitchType.MIDI)

                out_tpc = hu.get_interval_from_numeral(numeral, key_mode, PitchType.TPC)
                assert out_tpc == tpc[index] + adj * hc.ACCIDENTAL_ADJUSTMENT[PitchType.TPC], (
                    f"Output interval incorrect for inputs {(numeral, key_mode, PitchType.TPC)}"
                )
                assert out_tpc == hu.get_interval_from_numeral(numeral, key_mode, PitchType.TPC)


def test_get_interval_from_scale_degree():
    for acc, adj in zip(['bbb', 'bb', 'b', '', '#', '##', '###'], [-3, -2, -1, 0, 1, 2, 3]):
        for key_mode, semitones, tpc in zip([KeyMode.MAJOR, KeyMode.MINOR],
                                            [[0, 2, 4, 5, 7, 9, 11],
                                             [0, 2, 3, 5, 7, 8, 10]],
                                            [[0, 2, 4, -1, 1, 3, 5],
                                             [0, 2, -3, -1, 1, -4, -2]]):
            for numeral, number, index in zip(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'],
                                              ['1', '2', '3', '4', '5', '6', '7'], range(7)):
                for prefixed, degree in itertools.product([True, False], [numeral, number]):
                    if prefixed:
                        degree = acc + degree
                    else:
                        degree = degree + acc
                    out_semis = hu.get_interval_from_scale_degree(degree, prefixed, key_mode,
                                                                  PitchType.MIDI)
                    assert out_semis == semitones[index] + adj, (
                        "Output semitones incorrect for inputs "
                        f"{(degree, key_mode, prefixed, PitchType.MIDI)}"
                    )
                    assert out_semis == hu.get_interval_from_scale_degree(degree, prefixed,
                                                                          key_mode, PitchType.MIDI)

                    out_tpc = hu.get_interval_from_scale_degree(degree, prefixed, key_mode,
                                                                PitchType.TPC)
                    assert out_tpc == tpc[index] + adj * hc.ACCIDENTAL_ADJUSTMENT[PitchType.TPC], (
                        "Output interval incorrect for inputs "
                        f"{(degree, prefixed, key_mode, PitchType.TPC)}"
                    )
                    assert out_tpc == hu.get_interval_from_scale_degree(degree, prefixed, key_mode,
                                                                        PitchType.TPC)


def test_tpc_interval_to_midi_interval():
    for tpc in range(hc.NUM_PITCHES[PitchType.TPC]):
        midi_target = hu.get_pitch_from_string(
            hu.get_pitch_string(tpc, PitchType.TPC),
            PitchType.MIDI
        )
        assert hu.tpc_interval_to_midi_interval(tpc - hc.TPC_C) == midi_target


def test_transpose_pitch():
    for midi, tpc in zip(range(hc.NUM_PITCHES[PitchType.MIDI]),
                         range(hc.NUM_PITCHES[PitchType.TPC])):
        for interval in range(-50, 50):
            correct_midi = (midi + interval) % hc.NUM_PITCHES[PitchType.MIDI]
            res_midi = hu.transpose_pitch(midi, interval, PitchType.MIDI)
            assert res_midi == correct_midi

            correct_tpc = tpc + interval
            if 0 <= correct_tpc < hc.NUM_PITCHES[PitchType.TPC]:
                res_tpc = hu.transpose_pitch(tpc, interval, PitchType.TPC)
                assert res_tpc == correct_tpc
            else:
                with pytest.raises(ValueError):
                    hu.transpose_pitch(tpc, interval, PitchType.TPC)


def test_get_chord_inversion():
    correct_inversions = {
        '7':  0,
        '6':  1,
        '65': 1,
        '43': 2,
        '64': 2,
        '2':  3
    }

    for null_string in pd.NA, None, '':
        assert hu.get_chord_inversion(null_string) == 0

    for figbass, correct in correct_inversions.items():
        assert hu.get_chord_inversion(figbass) == correct

    with pytest.raises(ValueError):
        hu.get_chord_inversion('Some nonsense')


def test_get_chord_string_conversion():
    chord_type_strings = {
        ChordType.MAJOR: 'M',
        ChordType.MINOR: 'm',
        ChordType.DIMINISHED: 'o',
        ChordType.AUGMENTED: '+',
        ChordType.MIN_MIN7: 'mm7',
        ChordType.MAJ_MIN7: 'Mm7',
        ChordType.MAJ_MAJ7: 'MM7',
        ChordType.MIN_MAJ7: 'mM7',
        ChordType.DIM7: 'o7',
        ChordType.HALF_DIM7: '%7',
        ChordType.AUG_MIN7: '+7',
        ChordType.AUG_MAJ7: '+M7'
    }

    for chord_type, string in chord_type_strings.items():
        assert hu.get_chord_type_from_string(string) == chord_type
        assert hu.get_chord_string(chord_type) == string


def test_get_pitch_from_string():
    for acc, adj in zip(['###', '##', '#', '', 'b', 'bb', 'bbb'], [3, 2, 1, 0, -1, -2, -3]):
        for pitch, midi, tpc in zip(['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                                    [0, 2, 4, 5, 7, 9, 11],
                                    [0, 2, 4, -1, 1, 3, 5]):
            tpc += hc.TPC_C
            pitch += acc

            correct_midi = (midi + adj) % hc.NUM_PITCHES[PitchType.MIDI]
            assert hu.get_pitch_from_string(pitch, PitchType.MIDI) == correct_midi

            correct_tpc = tpc + adj * hc.ACCIDENTAL_ADJUSTMENT[PitchType.TPC]
            if 0 <= correct_tpc < hc.NUM_PITCHES[PitchType.TPC]:
                assert hu.get_pitch_from_string(pitch, PitchType.TPC) == correct_tpc
            else:
                with pytest.raises(ValueError):
                    hu.get_pitch_from_string(pitch, PitchType.TPC)


def test_get_pitch_string():
    base_tpc = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    pitch_strings = {
        PitchType.MIDI: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        PitchType.TPC: (
            [base + 'bb' for base in base_tpc] +
            [base + 'b' for base in base_tpc] +
            base_tpc +
            [base + '#' for base in base_tpc] +
            [base + '##' for base in base_tpc]
        )
    }
    for pitch_type in [PitchType.MIDI, PitchType.TPC]:
        for pitch in range(0, hc.NUM_PITCHES[pitch_type]):
            string = hu.get_pitch_string(pitch, pitch_type)
            correct_string = pitch_strings[pitch_type][pitch]

            if pitch_type == PitchType.TPC or ('#' not in correct_string):
                assert string == correct_string
            else:
                assert '/' in string and correct_string in string.split('/')


def test_get_chord_label_list():
    for pitch_type in [PitchType.MIDI, PitchType.TPC]:
        one_hots = hu.get_chord_label_list(pitch_type, use_inversions=False)
        i = 0

        for chord_type in ChordType:
            chord_string = hu.get_chord_string(chord_type)
            for pitch in range(0, hc.NUM_PITCHES[pitch_type]):
                pitch_string = hu.get_pitch_string(pitch, pitch_type)

                assert one_hots[i] == f'{pitch_string}:{chord_string}'
                i += 1

        assert i == len(one_hots)

    for pitch_type in [PitchType.MIDI, PitchType.TPC]:
        one_hots = hu.get_chord_label_list(pitch_type, use_inversions=True)
        i = 0

        for chord_type in ChordType:
            chord_string = hu.get_chord_string(chord_type)
            for pitch in range(0, hc.NUM_PITCHES[pitch_type]):
                pitch_string = hu.get_pitch_string(pitch, pitch_type)
                for inv in range(0, hu.get_chord_inversion_count(chord_type)):
                    assert one_hots[i] == f'{pitch_string}:{chord_string}, inv:{inv}'
                    i += 1

        assert i == len(one_hots)


def test_get_chord_inversion_count():
    for chord_type in ChordType:
        for pitch_type in PitchType:
            vector = hu.get_vector_from_chord_type(chord_type, pitch_type)
            assert hu.get_chord_inversion_count(chord_type) == sum(vector)


def test_get_chord_one_hot_index():
    for chord_type in ChordType:
        for pitch_type in PitchType:
            for root_pitch in range(hc.NUM_PITCHES[pitch_type]):
                for inversion in range(hu.get_chord_inversion_count(chord_type)):
                    pitch_string = hu.get_pitch_string(root_pitch, pitch_type)
                    chord_string = hu.get_chord_string(chord_type)

                    string_no_inv = f'{pitch_string}:{chord_string}'
                    string = f'{pitch_string}:{chord_string}, inv:{inversion}'

                    index_no_inv = hu.get_chord_one_hot_index(
                        chord_type, root_pitch, pitch_type, inversion=inversion,
                        use_inversion=False
                    )
                    assert (
                        hu.get_chord_label_list(pitch_type, use_inversions=False)[index_no_inv]
                        == string_no_inv
                    )

                    index = hu.get_chord_one_hot_index(
                        chord_type, root_pitch, pitch_type, inversion=inversion, use_inversion=True
                    )
                    assert (
                        hu.get_chord_label_list(pitch_type, use_inversions=True)[index] == string
                    )

    with pytest.raises(ValueError):
        hu.get_chord_one_hot_index(
            ChordType.MAJOR, -1, PitchType.MIDI, inversion=0, use_inversion=True
        )
    with pytest.raises(ValueError):
        hu.get_chord_one_hot_index(
            ChordType.MAJOR, hc.NUM_PITCHES[PitchType.MIDI], PitchType.MIDI, inversion=0,
            use_inversion=True
        )

    with pytest.raises(ValueError):
        hu.get_chord_one_hot_index(
            ChordType.MAJOR, -1, PitchType.TPC, inversion=0, use_inversion=True
        )
    with pytest.raises(ValueError):
        hu.get_chord_one_hot_index(
            ChordType.MAJOR, hc.NUM_PITCHES[PitchType.TPC], PitchType.TPC, inversion=0,
            use_inversion=True
        )

    with pytest.raises(ValueError):
        hu.get_chord_one_hot_index(
            ChordType.MAJOR, 0, PitchType.TPC, inversion=3, use_inversion=True
        )
    hu.get_chord_one_hot_index(
        ChordType.MAJOR, 0, PitchType.TPC, inversion=3, use_inversion=False
    )


def test_get_key_label_list():
    for pitch_type in [PitchType.MIDI, PitchType.TPC]:
        one_hots = hu.get_key_label_list(pitch_type)
        i = 0

        for key_mode in KeyMode:
            for pitch in range(0, hc.NUM_PITCHES[pitch_type]):
                pitch_string = hu.get_pitch_string(pitch, pitch_type)
                if key_mode == KeyMode.MINOR:
                    pitch_string = str(pitch_string).lower()
                assert one_hots[i] == f'{pitch_string}:{key_mode}'
                i += 1

        assert i == len(one_hots)


def test_get_key_one_hot_index():
    for key_mode in KeyMode:
        for pitch_type in PitchType:
            for tonic in range(hc.NUM_PITCHES[pitch_type]):
                tonic_string = hu.get_pitch_string(tonic, pitch_type)
                if key_mode == KeyMode.MINOR:
                    tonic_string = str(tonic_string).lower()
                string = f'{tonic_string}:{key_mode}'

                index = hu.get_key_one_hot_index(key_mode, tonic, pitch_type)
                assert hu.get_key_label_list(pitch_type)[index] == string

    with pytest.raises(ValueError):
        hu.get_key_one_hot_index(KeyMode.MAJOR, -1, PitchType.MIDI)
    with pytest.raises(ValueError):
        hu.get_key_one_hot_index(KeyMode.MAJOR, hc.NUM_PITCHES[PitchType.MIDI], PitchType.MIDI)

    with pytest.raises(ValueError):
        hu.get_key_one_hot_index(KeyMode.MAJOR, -1, PitchType.TPC)
    with pytest.raises(ValueError):
        hu.get_key_one_hot_index(KeyMode.MAJOR, hc.NUM_PITCHES[PitchType.TPC], PitchType.TPC)


def test_get_bass_note():
    for pitch_type in PitchType:
        for chord_type in ChordType:
            for inversion in range(hu.get_chord_inversion_count(chord_type)):
                target_bass = hc.CHORD_PITCHES[pitch_type][chord_type][inversion]
                for transposition in range(12):
                    root = transposition + hc.C[pitch_type]
                    bass = hu.get_bass_note(chord_type, root, inversion, pitch_type)
                    assert bass == hu.transpose_pitch(target_bass, transposition, pitch_type)
