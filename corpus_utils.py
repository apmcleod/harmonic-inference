"""Utility functions for working with the corpus data."""

import pandas as pd
import numpy as np
from fractions import Fraction


def get_offsets(notes):
    """
    Get the offset times (measure and beat) for each note.
    
    Parameters
    ----------
    notes : pd.DataFrame
        The notes whose offsets we want. Must have at least columns:
        duration, onset, mc, and timesig. These notes MUST NOT cross bar lines
        (e.g., as a result of merging ties), or results will be inaccurate
        (since we may only know the time signature of the bar with the note onset).
        Notes that continue to the end of a bar have their offset set to the downbeat
        of the following bar.
        
    Returns
    -------
    offset_mc : list(int)
        A list of the measure count of the offset for each of the given notes.
        
    offset_beat : list(Fraction)
        A list of the beat times of the offset for each of the given notes.
    """
    frac_timesigs = np.array([Fraction(ts) for ts in notes.timesig])
    
    offset_beat = (notes.duration + notes.onset).to_numpy()
    new_measure = offset_beat >= frac_timesigs
    offset_beat[new_measure] = Fraction(0)
    offset_mc = notes.mc.to_numpy(dtype=int) + new_measure
    
    return offset_mc, offset_beat
