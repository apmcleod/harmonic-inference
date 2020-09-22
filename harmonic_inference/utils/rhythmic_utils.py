"""Utility functions for getting rhythmic or metrical information from the corpus DataFrames."""
from typing import Tuple
from fractions import Fraction

import pandas as pd


def get_range_length(
    range_start: Tuple[int, Fraction],
    range_end: Tuple[int, Fraction],
    measures: pd.DataFrame
) -> Fraction:
    """
    Get the length of a range in whole notes.

    Parameters
    ----------
    range_start : tuple(int, Fraction)
        A tuple of (mc, beat) values of the start of the range.

    range_end : tuple(int, Fraction)
        A tuple of (mc, beat) values of the end of the range.

    measures : pd.DataFrame
        A DataFrame containing the measures info for this particular piece.

    Returns
    -------
    length : Fraction
        The length of the given range, in whole notes.
    """
    factor = 1
    if range_start > range_end:
        factor = -1
        tmp = range_start
        range_start = range_end
        range_end = tmp

    start_mc, start_beat = range_start
    end_mc, end_beat = range_end

    if start_mc == end_mc:
        return factor * (end_beat - start_beat)

    # Start looping at end of start_mc
    act_dur, offset, current_mc = measures.loc[measures.mc == start_mc,
                                               ['act_dur', 'offset', 'next']].values[0]
    length = act_dur + offset - start_beat

    # Loop until reaching end_mc
    while current_mc != end_mc and current_mc is not None:
        act_dur, current_mc = measures.loc[measures.mc == current_mc,
                                           ['act_dur', 'next']].values[0]
        length += act_dur

    # Add remainder
    final_offset = measures.loc[measures.mc == current_mc, 'offset'].values[0]
    length += end_beat - final_offset

    return factor * length


def get_rhythmic_info_as_proportion_of_range(
    note: pd.Series,
    range_start: Tuple[int, Fraction],
    range_end: Tuple[int, Fraction],
    measures: pd.DataFrame,
    range_len: Fraction = None
) -> Tuple[Fraction, Fraction, Fraction]:
    """
    Get a note's onset, offset, and duration as a proportion of the given range.

    Parameters
    ----------
    note : pd.Series
        The note whose onset offset and duration this will return.

    range_start : tuple(int, Fraction)
        A tuple of (mc, beat) values of the start of the range.

    range_end : tuple(int, Fraction)
        A tuple of (mc, beat) values of the end of the range.

    measures : pd.DataFrame
        A DataFrame containing the measures info for the the corpus.

    range_len : Fraction
        The total duration of the given range, if it is known.

    Returns
    -------
    onset : Fraction
        The onset of the note, as a proportion of the given range.

    offset : Fraction
        The offset of the note, as a proportion of the given range.

    duration : Fraction
        The duration of the note, as a proportion of the given range.
    """
    if range_len is None:
        range_len = get_range_length(range_start, range_end, measures)

    duration = note.duration / range_len

    onset_to_start = abs(note.mc - range_start[0])
    onset_to_end = abs(note.mc - range_end[0])
    if onset_to_start <= onset_to_end:
        onset = get_range_length(range_start, (note.mc, note.onset), measures) / range_len
    else:
        onset = 1 - get_range_length((note.mc, note.onset), range_end, measures) / range_len
    offset = onset + duration

    return onset, offset, duration


def get_metrical_level_lengths(timesig: str) -> Tuple[Fraction, Fraction, Fraction]:
    """
    Get the lengths of the beat and subbeat levels of the given time signature.

    Parameters
    ----------
    timesig : string
        A string representation of the time signature as "numerator/denominator".

    Returns
    -------
    measure_length : Fraction
        The length of a measure in the given time signature, where 1 is a whole note.

    beat_length : Fraction
        The length of a beat in the given time signature, where 1 is a whole note.

    sub_beat_length : Fraction
        The length of a sub_beat in the given time signature, where 1 is a whole note.
    """
    numerator, denominator = [int(val) for val in timesig.split('/')]

    if (numerator > 3) and (numerator % 3 == 0):
        # Compound meter
        sub_beat_length = Fraction(1, denominator)
        beat_length = sub_beat_length * 3
    else:
        # Simple meter
        beat_length = Fraction(1, denominator)
        sub_beat_length = beat_length / 2

    return Fraction(numerator, denominator), beat_length, sub_beat_length


def get_metrical_level(beat: Fraction, measure: pd.Series) -> int:
    """
    Get the metrical level of a given beat.

    Parameters
    ----------
    beat : Fraction
        The beat we are interested in within the measure. 1 corresponds to a whole
        note after the downbeat.

    measure : pd.Series
        The measures_df row of the corresponding mc.

    Returns
    -------
    level : int
        An int representing the metrical level of the given beat:
        3: downbeat
        2: beat
        1: sub-beat
        0: lower
    """
    measure_length, beat_length, sub_beat_length = get_metrical_level_lengths(measure.timesig)

    if beat % measure_length == 0:
        return 3
    if beat % beat_length == 0:
        return 2
    if beat % sub_beat_length == 0:
        return 1
    return 0
