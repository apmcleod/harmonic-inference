"""Utility functions for working with the corpus data."""

import pandas as pd


def get_notes_in_chord(chords_df, notes_df, chord_id, combine_ties=True):
    """
    Get the all of the notes that lie within some chord.
    
    Parameters
    ----------
    chords_df : pd.DataFrame
        The chords data.
        
    notes_df : pd.DataFrame
        The notes data.
        
    chord_id : int
        The id of the chord whose notes this will return.
        
    combine_ties : boolean
        Combine tied notes into a single note in the returned DataFrame.
        
    Returns
    -------
    selected_notes : pd.DataFrame
        The notes which occur during the chord with the given chord_id.
    """
    notes = notes_df.loc[notes_df.chord_id == chord_id]
    
    if combine_ties:
        pass
    
    return notes
    
    
def get_all_tied_notes(notes_df, note):
    """
    Get a DataFrame containing all of the notes tied to or from a given note. This
    will chain to give the full length of the tie.
    
    Parameters
    ----------
    notes_df : pd.DataFrame
        The notes data.
        
    note : pd.Series
        A single note whose ties we want.
        
    Return
    ------
    tied_notes : pd.DataFrame
        A DataFrame containing all of the notes which are tied to or from the given note.
    """
    tied_notes = pd.DataFrame(note)
    
    if note.tied.isna():
        # Note is untied
        return tied_notes
    
    # Check for all ties
    for tie_type in ['to', 'from']:
        current_note = note
        while current_note is not None:
            tied_notes.append(current_note)
            current_note = get_tied_note(notes_df, current_note, tie_type='to')
            
    return tied_notes
    
    
def get_tied_note(notes_df, note, tie_type='to'):
    """
    Get the note tied into or out of the given note (depending on the parameter tie_type).
    
    Parameters
    ----------
    notes_df : pd.DataFrame
        The notes data.
        
    note : pd.Series
        A single note whose ties we want.
        
    tie_type : String
        Either 'to', to get any note which is tied into the given one;
        or 'from', to get any note which the given one is tied into.
        
    Returns
    -------
    note : pd.Series
        The selected note, or None if no note exists.
    """
    assert tie in ['to', 'from']
    
    # Constant for column value meanings
    tied_to = [-1, 0]
    tied_from = [0, 1]
    
    # Constant for onset and offset times
    onset_beat = notes_df.onset
    onset_mc = notes_df.mc
    offset_beat = 
    
    # Set vars to allow for same search to be used for both tie types
    if tie_type == 'to':
        required_value = tied_to
        search_for_value = tied_from
        this_note_time = onset
        search_for_time = offset
    else:
        required_value = tied_from
        search_for_value = tied_to
        this_note_time = offset
        search_for_time = onset
        
    # Quick exit for note not tied as wanted
    if not note.tied in required_value:
        return None
    
    for tied_note in notes_df.loc[notes_df.tied.isin(search_for_value)].iterrows():
        
    