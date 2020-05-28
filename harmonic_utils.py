"""Utility functions for getting harmonic and pitch information from the corpus DataFrames."""

import pandas as pd


MAX_PITCH_DEFAULT = 127
PITCHES_PER_OCTAVE = 12

# Scale tone semitone difference from root
MAJOR_SCALE = [0, 0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 0, 2, 3, 5, 7, 8, 10]

NUMERAL_TO_NUMBER = {
    'I':   1,
    'II':  2,
    'III': 3,
    'IV':  4,
    'V':   5,
    'VI':  6,
    'VII': 7
}

NOTE_TO_INDEX = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11
}



def get_accidental_adjustment(string, in_front=True):
    """
    Get the accidental adjustment of the accidentals at the beginning of a string.
    
    Parameters
    ----------
    string : string
        The string whose accidentals we want. It should begin with some number of either 'b'
        or '#', and then anything else.
        
    in_front : boolean
        True if the accidentals come at the beginning of the string. False for the end.
        
    Returns
    -------
    adjustment : int
        -1 for each 'b' at the beginning of the string. +1 for each '#'.
        
    new_string : string
        The given string without the accidentals.
    """
    adjustment = 0
    
    if in_front:
        while string[0] == 'b' and len(string) > 1:
            string = string[1:]
            adjustment -= 1

        while string[0] == '#':
            string = string[1:]
            adjustment += 1
            
    else:
        while string[-1] == 'b' and len(string) > 1:
            string = string[:-1]
            adjustment -= 1

        while string[-1] == '#':
            string = string[:-1]
            adjustment += 1
        
    return adjustment, string



def get_numeral_semitones(numeral, is_major):
    """
    Convert the numeral of a chord tonic to a semitone offset, and return whether it is major
    or minor.
    
    Parameters
    ----------
    numeral : string
        The numeral of a chord, like I, bii, etc.
        
    is_major : boolean
        True if the current key is major. False otherwise.
        
    Returns
    -------
    semitones : int
         The number of semitones above the key tonic the given chord is.
         
    is_major : boolean
        True if the chord is major (upper-case). False otherwise.
    """
    adjustment, numeral = get_accidental_adjustment(numeral)
        
    is_major = numeral.isupper()
    if is_major:
        semitones = MAJOR_SCALE[NUMERAL_TO_NUMBER[numeral]]
    else:
        semitones = MINOR_SCALE[NUMERAL_TO_NUMBER[numeral.upper()]]
        
    return semitones + adjustment, is_major



def get_bass_step_semitones(bass_step, is_major):
    """
    Get the given bass step in semitones.
    
    Parameters
    ----------
    bass_step : string
        The bass step of a chord, 1, b7, etc.
        
    is_major : boolean
        True if the current key is major. False otherwise.
        
    Returns
    -------
    semitones : int
        The number of semitones above the chord root the given bass step is.
        None if the data is malformed ("Error" or "Unclear").
    """
    adjustment, bass_step = get_accidental_adjustment(bass_step)
    
    try:
        return int(bass_step) + adjustment
    except:
        return None



def get_key(key):
    """
    Get the tonic index of a given key string.
    
    Parameters
    ----------
    key : string
        The key, C, db, etc.
        
    Returns
    -------
    tonic_index : int
        The tonic index of the key, with 0 = C, 1 = C#/Db, etc.
        
    is_major : boolean
        True if the given key is major (the tonic is upper-case). False otherwise.
    """
    adjustment, key = get_accidental_adjustment(key, in_front=False)
    
    is_major = numeral.isupper()
    if is_major:
        tonic_index = NOTE_TO_INDEX[key]
    else:
        tonic_index = NOTE_TO_INDEX[key.upper()]
        
    return tonic_index + adjustment, is_major
