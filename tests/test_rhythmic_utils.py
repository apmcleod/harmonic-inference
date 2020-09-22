"""Tests for rhythmic_utils.py"""
from fractions import Fraction
import math

import pandas as pd

import harmonic_inference.utils.rhythmic_utils as ru


def test_get_range_length():
    NUM_MEASURES = 5
    DURATION = Fraction(3, 2)
    mc_current = list(range(NUM_MEASURES))
    next_mc = list(range(1, NUM_MEASURES)) + [pd.NA]
    act_dur = [DURATION] * NUM_MEASURES
    offset = [i * Fraction(1, 50) for i in range(NUM_MEASURES)]

    for start_mc in range(NUM_MEASURES):
        for end_mc in range(start_mc, NUM_MEASURES):
            for start_beat_num in range(7):
                min_end_beat_num = start_beat_num if start_mc == end_mc else 0

                for end_beat_num in range(min_end_beat_num, 6):
                    start_beat = Fraction(start_beat_num, 4) + Fraction(1, 2)
                    end_beat = Fraction(end_beat_num, 4) + Fraction(1, 2)
                    start = (start_mc, start_beat)
                    end = (end_mc, end_beat)

                    measures = pd.DataFrame({'mc': mc_current,
                                             'next': next_mc,
                                             'act_dur': act_dur,
                                             'offset': offset})

                    length = ru.get_range_length(start, end, measures)
                    if start_mc == end_mc:
                        correct_length = end_beat - start_beat
                    elif start_mc + 1 == end_mc:
                        correct_length = ((DURATION - (start_beat - offset[start_mc]))  # Start mc
                                          + (end_beat - offset[end_mc]))  # End measure
                    else:
                        correct_length = (DURATION * (end_mc - start_mc - 1)  # Full measures
                                          + (DURATION - (start_beat - offset[start_mc]))  # Start
                                          + (end_beat - offset[end_mc]))  # End
                    assert length == correct_length, (
                        f"Range length incorrect between {start} and {end}"
                    )
                    length = ru.get_range_length(end, start, measures)
                    assert length == -correct_length

                    # Try different measure lengths
                    measures.loc[start_mc, 'act_dur'] = Fraction(7, 2)
                    if end_mc > start_mc:
                        measures.loc[list(range(start_mc + 1, end_mc + 1)),
                                     'act_dur'] = Fraction(9, 2)

                    length = ru.get_range_length(start, end, measures)
                    if start_mc == end_mc:
                        correct_length = end_beat - start_beat
                    elif start_mc + 1 == end_mc:
                        correct_length = (Fraction(7, 2) - (start_beat - offset[start_mc])  # Start
                                          + (end_beat - offset[end_mc]))  # End measure
                    else:
                        correct_length = (Fraction(9, 2) * (end_mc - start_mc - 1) +  # Full
                                          Fraction(7, 2) - (start_beat - offset[start_mc])  # Start
                                          + (end_beat - offset[end_mc]))  # End
                    assert length == correct_length, (
                        f"Range length incorrect between {start} and {end}"
                    )
                    length = ru.get_range_length(end, start, measures)
                    assert length == -correct_length

                    # Try weird next list
                    if end_mc > start_mc:
                        measures = pd.DataFrame({'mc': mc_current,
                                                 'next': next_mc,
                                                 'act_dur': act_dur,
                                                 'offset': offset})

                        # One measure
                        measures.loc[start_mc, 'next'] = end_mc
                        length = ru.get_range_length(start, end, measures)
                        correct_length = ((DURATION - (start_beat - offset[start_mc]))  # Start mc
                                          + (end_beat - offset[end_mc]))  # End measure
                        assert length == correct_length, (
                            f"Range length incorrect with start.next==end between {start} and "
                            f"{end}"
                        )
                        length = ru.get_range_length(end, start, measures)
                        assert length == -correct_length

                        if start_mc != 0:
                            measures.loc[start_mc, 'next'] = 0
                            measures.loc[0, 'next'] = end_mc
                            length = ru.get_range_length(start, end, measures)
                            correct_length = (DURATION +  # Full measures
                                              (DURATION - (start_beat - offset[start_mc]))  # Start
                                              + (end_beat - offset[end_mc]))  # End
                            assert length == correct_length, (
                                f"Range length incorrect with start.next==0 between {start} and "
                                f"{end}"
                            )
                            length = ru.get_range_length(end, start, measures)
                            assert length == -correct_length


def test_get_rhythmic_info_as_proportion_of_range():
    # note, range_start, range_end, measures, range_len=None
    # note has duration, mc, onset

    # Create measures
    NUM_MEASURES = 5
    DURATION = Fraction(3, 2)
    mc_current = list(range(NUM_MEASURES))
    next_mc = list(range(1, NUM_MEASURES)) + [pd.NA]
    act_dur = [DURATION] * NUM_MEASURES
    offset = [i * Fraction(1, 50) for i in range(NUM_MEASURES)]
    measures = pd.DataFrame({'mc': mc_current,
                             'next': next_mc,
                             'act_dur': act_dur,
                             'offset': offset})

    # Note equal to range
    for note_onset in [Fraction(2, 5), Fraction(1, 2), Fraction(3, 2)]:
        for mc_current in [0, 1]:
            for note_duration in [0, Fraction(1, 2), Fraction(3, 2)]:
                for range_end_mc in range(mc_current + 1, 5):
                    for range_start_mc in range(0, mc_current):
                        range_start = (range_start_mc, Fraction(1, 2))
                        range_end = (range_end_mc, Fraction(1, 2))
                        note = pd.Series([mc_current, note_onset, note_duration],
                                         index=['mc', 'onset', 'duration'])
                        onset, offset, duration = ru.get_rhythmic_info_as_proportion_of_range(
                            note, range_start, range_end, measures
                        )

                        # Tested above
                        range_len = ru.get_range_length(range_start, range_end, measures)
                        correct_onset = ru.get_range_length(range_start, (note.mc, note.onset),
                                                            measures) / range_len
                        correct_duration = note.duration / range_len
                        correct_offset = correct_onset + correct_duration
                        assert onset == correct_onset, (
                            f"Incorrect onset for {note} and range {range_start}-{range_end}"
                        )
                        assert offset == correct_offset, (
                            f"Incorrect offset for {note} and range {range_start}-{range_end}"
                        )
                        assert duration == correct_duration, (
                            f"Incorrect duration with note {note} and range "
                            f"{range_start}-{range_end}"
                        )

                        # Again with range_len given
                        onset, offset, duration = ru.get_rhythmic_info_as_proportion_of_range(
                            note, range_start, range_end, measures, range_len=range_len
                        )
                        assert onset == correct_onset, (
                            f"Incorrect onset with range_len given and note {note} and range "
                            f"{range_start}-{range_end}"
                        )
                        assert offset == correct_offset, (
                            f"Incorrect offset with range_len given and note {note} and range "
                            f"{range_start}-{range_end}"
                        )
                        assert duration == correct_duration, (
                            f"Incorrect duration with range_len given and note {note} and range "
                            f"{range_start}-{range_end}"
                        )


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
            assert sub_beat == sub_beat_correct, (
                f"Sub beat length incorrect for time_sig {time_sig}"
            )


def test_get_metrical_level():
    # Time signatures are really tested above
    for num in [3, 4, 12]:
        for denom in [4, 8, 16]:
            time_sig = f"{num}/{denom}"
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
                    assert level == 3

                    # Swap offset and beat
                    beat = measure.offset
                    measure.offset = 0

                    level = ru.get_metrical_level(beat, measure)
                    assert level == correct_level, (
                        f"Level is wrong for measure {measure} and beat {beat}"
                    )

            # Try with offset == measure_length
            level = ru.get_metrical_level(measure_length,
                                          pd.Series([time_sig, 0], index=['timesig', 'offset']))
            assert level == 3, "Next downbeat is not detected correctly"
