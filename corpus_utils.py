"""Utility functions for working with the corpus data."""

import pandas as pd
import numpy as np
from fractions import Fraction


def get_offsets(notes, measures):
    """
    Get the offset positions ('mc' measure count and beat) for each note. If the offset is
    on a downbeat, the returned offset is at the following measure at beat 0; UNLESS the note
    is in the last measure of a piece, in which case the returned offset is in that measure
    at a beat equal to the duration of that bar.
    
    Parameters
    ----------
    notes : pd.DataFrame
        The notes whose offset positions we want. Must have at least these columns:
            'id' (index, int): The piece id from which this note comes.
            'mc' (int): The 'measure count' index of the onset of each note. This is used
                to index into the given measures DataFrame.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of the given mc.
            'duration' (Fraction): The duration of each note, in whole notes. Notes whose
                duration goes beyond the end of their mc (e.g., after merging ties) are
                handled correctly.
        
    measures : pd.DataFrame
        A DataFrame containing information about each measure. Must have at least these columns:
            'id' (index, int): The piece id from which this measure comes.
            'mc' (index, int): The 'measure count' index of this measure. Used by notes to index into
                this DataFrame.
            'act_dur' (Fraction): The duration of this measure, in whole notes. Note that this
                can be different from 'timesig', because of, e.g., partial measures near repeats.
            'next' (list(int)): The 'mc' of the measure that follows this one. This may contain
                multiple 'mc's in the case of a repeat, but it is recommended to either
                unroll or eliminate repeats before running get_offsets, which will result in
                only 1- or 0-length lists in this column. In the case of a longer list, the last
                'mc' in the list (measures['next'][-1]) is treated as the next mc and a warning is
                printed. This functionality is similar to eliminating repeats, although the underlying
                measures DataFrame is not changed.
        
    Returns
    -------
    offset_mc : list(int)
        A list of the 'mc' measure count of the offset for each of the notes in the given notes
        DataFrame.
        
    offset_beat : list(Fraction)
        A list of the beat times of the offset of each of the notes in the given notes DataFrame,
        measured in whole notes after the beginning of the corresponding offset_mc.
    """
    # Index measures in the order of notes
    note_measures = measures.loc[pd.MultiIndex.from_arrays((notes.index.get_level_values('id'), notes.mc))]
    last_measures = note_measures.next.apply(len) == 0
    
    # Simple offset position calculation
    offset_mc = notes.mc.to_numpy()
    offset_beat = (notes.onset + notes.duration).to_numpy()
    
    # Fix offsets which go beyond the end of their current measure, (if that isn't the last measure)
    new_measures = ((offset_beat >= note_measures.act_dur) & ~last_measures).to_numpy()
    while new_measures.any():
        # Update offset position (and save indexed list for speed later)
        # First 3 lines of code: save indexed lists for faster computation
        note_measures = note_measures.loc[new_measures]
        new_offset_beats = (offset_beat[new_measures] - note_measures.act_dur).to_numpy()
        new_offset_mcs = [mc[-1] for mc in note_measures.next] # Get the last value in each 'next' list
        offset_beat[new_measures] = new_offset_beats
        offset_mc[new_measures] = new_offset_mcs
        
        # Update indexed measure info with new values based on new note 'mc's
        note_measures = measures.loc[pd.MultiIndex.from_arrays((note_measures.index.get_level_values('id'),
                                                                new_offset_mcs), names=['id', 'mc'])]
        last_measures = note_measures.next.apply(len) == 0
        
        # Check for any notes which still go beyond the end of a measure
        new_measures[new_measures] = ((new_offset_beats >= note_measures.act_dur) & ~last_measures).to_numpy()
    
    return offset_mc, offset_beat
