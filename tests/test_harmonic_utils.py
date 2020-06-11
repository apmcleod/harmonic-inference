"""Tests for harmonic_utils.py"""
import pytest

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import itertools

import harmonic_utils as hu


def test_get_accidental_adjustment():
    for accidental, diff in zip(['#', 'b'], [1, -1]):
        for count in range(10):
            for root in ['b', 'B', 'V', '1']:
                # Front
                input_string = accidental * count + root
                adj_out, out_string = hu.get_accidental_adjustment(input_string, in_front=True)
                
                assert adj_out == count * diff, f"Incorrect adjustment ({adj_out}) for input {input_string}"
                assert out_string == root, f"Incorrect string out ({out_string}) for input {intpu_string}"
                
                # Back
                input_string = root + accidental * count
                adj_out, out_string = hu.get_accidental_adjustment(input_string, in_front=False)
                
                assert adj_out == count * diff, f"Incorrect adjustment ({adj_out}) for input {input_string}"
                assert out_string == root, f"Incorrect string out ({out_string}) for input {intput_string}"
                
                
                
def test_get_numeral_semitones():
    for acc, adj in zip(['b', '', '#'], [-1, 0, 1]):
        for is_major, semitones in zip([True, False],
                                       [[0, 2, 4, 5, 7, 9, 11],
                                        [0, 2, 3, 5, 7, 8, 10]]):
            for numeral, index in zip(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], range(7)):
                numeral = acc + numeral
                out_semis, out_is_major = hu.get_numeral_semitones(numeral, is_major)
                assert out_semis == semitones[index] + adj, f"Output semitones incorrect for inputs {numeral}, {is_major}"
                assert out_is_major, f"Output is_major incorrect for inputs {numeral}, {is_major}"
                
                numeral_lower = numeral.lower()
                out_semis_lower, out_is_major_lower = hu.get_numeral_semitones(numeral_lower, is_major)
                assert out_semis_lower == out_semis, f"Output semitones different for {numeral} and {numeral_lower}"
                assert not out_is_major_lower, f"Output is_major incorrect for inputs {numeral_lower}, {is_major}"
                
    for numeral in ['GER', 'IT', 'FR']:
        correct_semis = 8
        for is_major in [True, False]:
            out_semis, out_is_major = hu.get_numeral_semitones(numeral, is_major)
            assert out_semis == correct_semis, f"Output semitones incorrect for inputs {numeral}, {is_major}"
            assert not out_is_major, f"Output is_major incorrect for inputs {numeral_lower}, {is_major}"
                
                
def test_get_bass_step_semitones():
    for acc, adj in zip(['b', '', '#'], [-1, 0, 1]):
        for is_major, semitones in zip([True, False],
                                       [[0, 2, 4, 5, 7, 9, 11],
                                        [0, 2, 3, 5, 7, 8, 10]]):
            for step, index in zip(['1', '2', '3', '4', '5', '6', '7'], range(7)):
                step = acc + step
                out_semis = hu.get_bass_step_semitones(step, is_major)
                assert out_semis == semitones[index] + adj, f"Output semitones incorrect for inputs {step}, {is_major}"
                
    for step in ['Error', 'Unclear']:
        for is_major in [True, False]:
            out_semis = hu.get_bass_step_semitones(step, is_major)
            assert out_semis is None, f"Output semitones not None for inputs {numeral}, {is_major}"
            
            
            
def test_get_key():
    for acc, adj in zip(['b', '', '#'], [-1, 0, 1]):
        for tonic, semitones in zip(['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                                    [0, 2, 4, 5, 7, 9, 11]):
            major_key = tonic + acc
            out_semitones, out_is_major = hu.get_key(major_key)
            assert out_semitones == semitones + adj, f"Output semitones incorrect for input {major_key}"
            assert out_is_major, f"Output is minor for key {major_key}"
            
            minor_key = major_key.lower()
            out_semitones, out_is_major = hu.get_key(minor_key)
            assert out_semitones == semitones + adj, f"Output semitones incorrect for input {minor_key}"
            assert not out_is_major, f"Output is major for key {minor_key}"
            
            
            
            
def test_transpose_chord_vector():
    LENGTH = 12
    
    for chord_vector in itertools.product([0, 1], repeat=LENGTH):
        for transposition in range(-LENGTH + 1, LENGTH):
            output = hu.transpose_chord_vector(chord_vector, transposition)
            for index in range(LENGTH):
                out_index = (index + transposition + LENGTH) % LENGTH
                assert chord_vector[index] == output[out_index], (
                    f"Transposed output at index {out_index} does not match input at index {index} "
                    f"with transposition {transposition}"
                )
                
                
                
                
def test_get_vector_from_chord_type():
    for chord_type, vector in zip(['M', 'm', 'o', '+', 'mm7', 'Mm7', 'MM7', 'mM7', 'o7', '%7', '+7'],
                                  [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                                   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]]):
        out_vector = hu.get_vector_from_chord_type(chord_type)
        assert vector == out_vector, f"Chord vector incorrect for chord type {chord_type}"
        
        
        
        
def test_get_chord_type_string():
    for chord_type in ['M', 'm', 'o', '+', 'mm7', 'Mm7', 'MM7', 'mM7', 'o7', '%7', '+7']:
        form = pd.NA
        figbass = pd.NA
        is_major = True
        
        # Triad
        if chord_type[-1] != '7':
            if chord_type in ['o', '+']:
                form = chord_type
            else:
                is_major = chord_type.isupper()
                
            for figbass in [pd.NA, '6', '64']:
                out_type = hu.get_chord_type_string(is_major, form=form, figbass=figbass)
                assert out_type == chord_type, (f"Chord type is {out_type} instead of {chord_type} "
                                                f"for inputs (is_major, form, figbass) {is_major}, "
                                                f"{form}, {figbass}")
            
        # 7th chord
        else:
            # Dim, aug, half-dim
            if len(chord_type) == 2:
                form = chord_type[0]
                for figbass in ['7', '65', '43', '2', '42']:
                    out_type = hu.get_chord_type_string(is_major, form=form, figbass=figbass)
                    assert out_type == chord_type, (f"Chord type is {out_type} instead of {chord_type} "
                                                    f"for inputs (is_major, form, figbass) {is_major}, "
                                                    f"{form}, {figbass}")
                    
            # MM, Mm, mM, mm
            else:
                is_major = chord_type[0].isupper()
                form = chord_type[1]
                for figbass in ['7', '65', '43', '2', '42']:
                        out_type = hu.get_chord_type_string(is_major, form=form, figbass=figbass)
                        assert out_type == chord_type, (f"Chord type is {out_type} instead of {chord_type} "
                                                        f"for inputs (is_major, form, figbass) {is_major}, "
                                                        f"{form}, {figbass}")
