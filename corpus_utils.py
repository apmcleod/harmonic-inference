"""Utility functions for working with the corpus data."""

import warnings
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
    
    # Find the last measures of each piece
    next_lengths = measures.next.apply(len)
    last_measures = next_lengths == 0 # Default case
    
    # Check for multi-valued next lists
    if next_lengths.max() > 1:
        warnings.warn("Repeats have not been unrolled or removed. Calculating offsets as if they "
                      "were removed (by using only the last 'next' pointer for each measure).")
        last_measures |= (np.roll(measures.index.get_level_values('id').to_numpy(), -1) !=
                          measures.index.get_level_values('id'))
        
    # Get last_measures for each of our notes
    note_last_measures = last_measures.loc[note_measures.index]
    
    # Simple offset position calculation
    offset_mc = notes.mc.to_numpy()
    offset_beat = (notes.onset + notes.duration).to_numpy()
    
    # Find which offsets go beyond the end of their current measure, (if it isn't the last measure)
    new_measures = ((offset_beat >= note_measures.act_dur) & ~note_last_measures).to_numpy()
    to_change_note_measures = note_measures.loc[new_measures]
    
    # Loop through, fixing those notes that still go beyond the end of their measure
    while new_measures.any():
        # Update offset position in 2 steps
        # First: save lists of only the changed values for faster computation
        changed_offset_beats = (offset_beat[new_measures] - to_change_note_measures.act_dur).to_numpy()
        changed_offset_mcs = [mc[-1] for mc in to_change_note_measures.next] # Get the last value in each 'next' list
        
        # Second: Update the global offset lists with the changed values
        offset_beat[new_measures] = changed_offset_beats
        offset_mc[new_measures] = changed_offset_mcs
        
        # Updated measure info for changed note 'mc's
        changed_note_measures = measures.loc[pd.MultiIndex.from_arrays((to_change_note_measures.index.get_level_values('id'),
                                                                        changed_offset_mcs), names=['id', 'mc'])]
        note_last_measures = last_measures.loc[changed_note_measures.index]
        
        # Check for any notes which still go beyond the end of a measure
        changed_new_measures = ((changed_offset_beats >= changed_note_measures.act_dur) & ~note_last_measures).to_numpy()
        new_measures[new_measures] = changed_new_measures
        to_change_note_measures = changed_note_measures.loc[changed_new_measures]
    
    return offset_mc, offset_beat



def find_matching_tie(note, tied_in_notes, prefiltered=False):
    """
    Find the note which a given note is tied to. The matching note must have an onset beat
    and mc equal to the given note's offset_beat and offset_mc, as well as equal midi pitch.
    If multiple matching notes are found, they are disambiguated by using the notes' voice.
    
    By default, the matching note must also match the given note's id and section (the first
    two values of its multi-index tuple). If prefiltered is given as True, the given
    tied_in_notes DataFrame is assumed to be already filtered by id and section, and this
    requirement is not checked.
    
    Parameters
    ----------
    note : pd.Series
        A note that is tied out of. The function will return the note which this note is
        tied into. The series must have at least the columns:
            'midi' (int): The MIDI pitch of the note.
            'voice' (int): The voice of the note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time (and might potentially be
                the resulting tied note).
            'offset_beat' (Fraction): The offset beat of the note (see get_offsets).
            'offset_mc' (int): The offset 'mc' of the note (see get_offsets).
        Additionally, if prefiltered is False, the note's name attribute must be a tuple
        where the first and second values correspond to the note's id and section values.
    
    tied_in_notes : pd.DataFrame
        A DataFrame in which to search for the matching tied note of the given note.
        It must have at least the following columns:
            'midi' (int): The MIDI pitch of each note.
            'voice' (int): The voice of each note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time (and might potentially be
                the resulting tied note).
            'mc' (int): The 'measure count' index of the onset of each note.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of the given mc.
        Additionally, if prefiltered is False, its first two index columns must be:
            'id' (index, int): The piece id from which each note comes; and
            'section' (index, int): The section of the piece from which each note comes.
            
    prefiltered : boolean
        False if the given tied_in_notes DataFrame has not yet been filtered by id and section.
        In this case, the note's name attribute must be a tuple whose first two values
        correspond to its id and section, and tied_in_notes must use a MultiIndex whose first
        two values correspond to its id and section. If True, the id and section values
        are not checked, and are not even required to be present.
        
    Returns
    -------
    matched_note : pd.Series
        The matching tied note to the given one, or None, if none was found. It's name attribute
        will correspond to the MultiIndex columns that are in the given tied_in_notes DataFrame.
        
        If multiple matches are found in the basic search (across midi pitch, onset/offset beat
        and measure, as well as id and section if prefiltered is False), the matches are then
        filtered by voice. If one remains, it is returned as the match. If multiple remain,
        the first one (in the order of the given tied_in_notes DataFrame) is returned as the match.
        If none remain, the first one returned by the basic search is returned.
    """
    if not prefiltered:
        unfiltered_tied_in_notes = tied_in_notes
        tied_in_notes = tied_in_notes.loc[(note.name[0], note.name[1])]
        
    matching_notes_mask = (
        (tied_in_notes.mc == note.offset_mc) &
        (tied_in_notes.onset == note.offset_beat) &
        (tied_in_notes.midi == note.midi)
    )
    matching_notes = tied_in_notes.loc[matching_notes_mask]
    
    if len(matching_notes) > 1:
        # More than 1 match -- filter by voice
        voice_matching_notes = tied_in_notes.loc[matching_notes_mask & tied_in_notes.voice == note.voice]
        
        # Replace matches if voice filtering was successful (or at least, didn't remove all matches)
        if len(voice_matching_notes) != 0:
            matching_notes = voice_matching_notes
    
    # Error -- no match found
    if len(matching_notes) == 0:
        warnings.warn(f"No matching tied note ({matching_notes.index}) found for note index "
                      f"{note.name} and duration {note.duration}. Returning the first one.")
        return None
    
    # Error -- multiple matches found
    if len(matching_notes) > 1:
        warnings.warn(f"Multiple matching tied notes ({matching_notes.index}) found for note index "
                      f"{note.name} and duration {note.duration}. Returning the first one.")
        
    # Return the first note on success or matches > 1
    if prefiltered:
        return matching_notes.iloc[0]
    else:
        return unfiltered_tied_in_notes.loc[(note.name[0], note.name[1], matching_notes.iloc[0].name)]



def merge_ties(notes, measures=None):
    """
    Return a new notes DataFrame, with tied notes removed and replaced by a single note with
    longer duration. If 'offset_beat' and 'offset_mc' columns are not in the given notes DataFrame,
    they will be calculated, so measure must be given. Grace notes are ignored during ties and
    are just returned as is.
    
    Parameters
    ----------
    notes : pd.DataFrame
        A pandas DataFrame containing the notes to be merged together. This should include at least
        the following columns:
            'id' (index, int): The piece id from which each note comes.
            'section' (index, int): The section of the piece from which each note comes.
            'mc' (int): The 'measure count' index of the onset of each note.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of the given mc.
            'duration' (Fraction): The duration of each note, in whole notes. Notes whose
                duration goes beyond the end of their mc (e.g., after merging ties) are
                handled correctly.
            'midi' (int): The MIDI pitch of each note.
            'voice' (int): The voice of each note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time.
            'gracenote' (string): What type of grace-note each note is, or pd.NA if it is not a
                gracenote. This is used because we ignore grace notes and simply return them as is
                during note merging.
            'tied' (int): The tied status of each note:
                pd.nan if the note is not tied.
                1 if the note is tied out of (i.e., it is an onset).
                -1 if the note is tied into (i.e., it is an offset).
                0 if the note is tied into and out of (i.e., it is neither an onset nor an offset).
        The following columns will be calculated with get_offsets if not present (so measure must
        be given):
            'offset_beat' (Fraction): The offset beat of each note (see get_offsets).
            'offset_mc' (int): The offset 'mc' of each note (see get_offsets).
            
    measures : pd.DataFrame
        Data about the measures in the corpus. Required if notes does not contain the columns
        'offset_beat' and 'offset_mc'. See get_offsets for more information.
        
    Returns
    -------
    merged_notes : pd.DataFrame
        A pandas DataFrame containing all of the notes from the input DataFrame, but with the
        merged notes removed and replaced by a single note, spanning their entire duration,
        with tied = 1.
    """
    # First, check for offset information, and calculate it if necessary
    if not all([column in notes.columns for column in ['offset_beat', 'offset_mc']]):
        assert measures is not None, ("measures must be given if offset_beat and offset_mc "
                                      "are not in notes")
        offset_mc, offset_beat = get_offsets(notes, measures)
        notes = notes.assign(offset_mc=offset_mc, offset_beat=offset_beat)
    
    # Tied in and out notes
    tied_out_mask = notes.tied == 1
    tied_out_notes = notes.loc[tied_out_mask & notes.gracenote.isna()]
    tied_in_notes = notes.loc[notes.tied.isin([-1, 0]) & notes.gracenote.isna()]
    
    # For all initial gracenote ties, add the notes they are tied into to the tied_out_notes
    to_add = [] # Add to tied_out_notes
    to_remove = [] # Remove from tied_in_notes
    for idx, note in notes.loc[tied_out_mask & ~notes.gracenote.isna()].iterrows():
        tied_note = find_matching_tie(note, tied_in_notes)
        
        if tied_note is not None:
            to_remove.append(tied_note.name)
            if tied_note.tied == 0:
                to_add.append(tied_note.name)
    
    # Move found notes from tied_in to tied_out
    tied_out_notes = tied_out_notes.append(notes.loc[to_add])
    tied_in_notes = tied_in_notes.drop(index=to_remove)
    
    # This is all of the notes that will be returned
    merged_notes = pd.DataFrame(notes.loc[notes.tied.isna() | (notes.index.isin(tied_out_notes.index))])
    
    # Loop through and fix the duration and offset every tied out note
    prev_idx = -1
    max_index = tied_out_notes.index.get_level_values('id').max()
    for idx, _ in tied_out_notes.iterrows():
        if prev_idx != idx[0]:
            prev_idx = idx[0]
            print(f"Index {idx[0]} / {max_index}")
            tied_in_notes_filtered = tied_in_notes.loc[(idx[0])]
        
        # Add new notes until an end tie is reached (where tied == -1)
        while True:
            tied_note = find_matching_tie(merged_notes.loc[idx], tied_in_notes_filtered, prefiltered=True)
            
            # Error -- no matching tie found.
            if tied_note is None:
                break
                
            # Update duration and break if the tie has ended
            merged_notes.at[idx, ['duration', 'offset_mc', 'offset_beat']] = [
                merged_notes.loc[idx, 'duration'] + tied_note.duration,
                tied_note.offset_mc,
                tied_note.offset_beat
            ]
            if tied_note.tied == -1:
                break
    
    return merged_notes
