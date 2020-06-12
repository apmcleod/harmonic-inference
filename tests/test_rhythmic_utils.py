"""Tests for rhythmic_utils.py"""
import pytest

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
from fractions import Fraction
import math

import rhythmic_utils as ru


def test_get_range_length():
    NUM_MEASURES = 5
    
    mc = list(range(NUM_MEASURES))
    next = list(range(1, NUM_MEASURES)) + [pd.NA]
    act_dur = [Fraction(3, 2)] * NUM_MEASURES
    
    for start_mc in range(NUM_MEASURES):
        for end_mc in range(start_mc, NUM_MEASURES):
            for start_beat_num in range(7):
                min_end_beat_num = start_beat_num if start_mc == end_mc else 0
                
                for end_beat_num in range(min_end_beat_num, 7):
                    start_beat = Fraction(start_beat_num, 4)
                    end_beat = Fraction(end_beat_num, 4)
                    start = (start_mc, start_beat)
                    end = (end_mc, end_beat)
                    
                    measures = pd.DataFrame({'mc': mc,
                                             'next': next,
                                             'act_dur': act_dur}).set_index('mc')
                    
                    length = ru.get_range_length(start, end, measures)
                    if start_mc == end_mc:
                        correct_length = end_beat - start_beat
                    elif start_mc + 1 == end_mc:
                        correct_length = end_beat + Fraction(3, 2) - start_beat
                    else:
                        correct_length = (Fraction(3, 2) * (end_mc - start_mc - 1) + # Full measures
                                          (Fraction(3, 2) - start_beat) + # Start measure
                                          end_beat) # End measure
                    assert length == correct_length, f"Range length incorrect between {start} and {end}"
                    
                    # Try different measure lengths
                    measures.loc[start_mc, 'act_dur'] = Fraction(7, 2)
                    if end_mc > start_mc:
                        measures.loc[list(range(start_mc + 1, end_mc + 1)), 'act_dur'] = Fraction(9, 2)
                        
                    length = ru.get_range_length(start, end, measures)
                    if start_mc == end_mc:
                        correct_length = end_beat - start_beat
                    elif start_mc + 1 == end_mc:
                        correct_length = end_beat + Fraction(7, 2) - start_beat
                    else:
                        correct_length = (Fraction(9, 2) * (end_mc - start_mc - 1) + # Full measures
                                          (Fraction(7, 2) - start_beat) + # Start measure
                                          end_beat) # End measure
                    assert length == correct_length, f"Range length incorrect between {start} and {end}"
                    
                    # Try weird next list
                    if end_mc > start_mc:
                        measures = pd.DataFrame({'mc': mc,
                                                 'next': next,
                                                 'act_dur': act_dur}).set_index('mc')
                        
                        # One measure
                        measures.loc[start_mc, 'next'] = end_mc
                        length = ru.get_range_length(start, end, measures)
                        correct_length = end_beat + Fraction(3, 2) - start_beat
                        assert length == correct_length, f"Range length incorrect with start.next==end between {start} and {end}"
                        
                        if start_mc != 0:
                            measures.loc[start_mc, 'next'] = 0
                            measures.loc[0, 'next'] = end_mc
                            length = ru.get_range_length(start, end, measures)
                            correct_length = end_beat + 2 * Fraction(3, 2) - start_beat
                            assert length == correct_length, f"Range length incorrect with start.next==0 between {start} and {end}"
            



def get_rhythmic_info_as_proportion_of_range():
    pass



def test_get_metrical_level_lengths():
    for num in range(1, 17):
        is_compound = num > 3 and num % 3 == 0
        
        for denom in [1, 2, 4, 8, 16, 32]:
            time_sig = str(num) + '/' + str(denom)
            measure_correct = Fraction(num, denom)
            
            if is_compound:
                sub_beat_correct = Fraction(1, denom)
                beat_correct = sub_beat_correct * 3
            else:
                beat_correct = Fraction(1, denom)
                sub_beat_correct = beat_correct / 2
                
            measure, beat, sub_beat = ru.get_metrical_level_lengths(time_sig)
            assert measure == measure_correct, f"Measure length incorrect for time_sig {time_sig}"
            assert beat == beat_correct, f"Beat length incorrect for time_sig {time_sig}"
            assert sub_beat == sub_beat_correct, f"Sub beat length incorrect for time_sig {time_sig}"



def test_get_metrical_level():
    # Time signatures are really tested above
    for num in [3, 4, 12]:
        for denom in [4, 8, 16]:
            time_sig = str(num) + '/' + str(denom)
            measure_length, beat_length, sub_beat_length = ru.get_metrical_level_lengths(time_sig)
            
            # This is what is being tested here
            for offset_denom in [1, 2, 4, 6, 8, 16, 32, 64]:
                for offset_num in range(0, 1 + math.floor(offset_denom * measure_length)):
                    offset = Fraction(offset_num, offset_denom)
                    
                    measure = pd.Series([time_sig, offset], index=['timesig', 'offset'])
                    if offset % measure_length == 0:
                        correct_level = 3
                    elif offset % beat_length == 0:
                        correct_level = 2
                    elif offset % sub_beat_length == 0:
                        correct_level = 1
                    else:
                        correct_level = 0
                    
                    beat = 0
                    level = ru.get_metrical_level(beat, measure)
                    assert level == correct_level, f"Level is wrong for measure {measure} and beat {beat}"
                    
                    # Swap offset and beat
                    beat = measure.offset
                    measure.offset = 0
                    
                    level = ru.get_metrical_level(beat, measure)
                    assert level == correct_level, f"Level is wrong for measure {measure} and beat {beat}"
                    
            # Try with offset == measure_length
            level = ru.get_metrical_level(measure_length, pd.Series([time_sig, 0], index=['timesig', 'offset']))
            assert level == 3, f"Next downbeat is not detected correctly"
