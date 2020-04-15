"""Utilities for parsing corpus tsv files into pandas DataFrames."""

import pandas as pd
from fractions import Fraction as frac

# Data types for the different columns in the tsvs
DTYPES = {'bass_step': 'string',
          'barline': 'string',
          'beat': 'Int64',
          'beats': 'string',
          'changes': 'string',
          'chord': 'string',
          'chords': 'string',
          'dont_count': 'Int64',
          'figbass': 'string',
          'form': 'string',
          'globalkey': 'string',
          'gracenote': 'string',
          'key': 'string',
          'keysig': 'Int64',
          'marker': 'string',
          'mc': 'Int64',
          'mc_next': 'Int64',
          'midi': 'Int64',
          'mn': 'Int64',
          'next_chord_id': 'Int64',
          'note_names': 'string',
          'numbering_offset': 'Int64',
          'numeral': 'string',
          'octaves': 'Int64',
          'overlapping': 'Int64',
          'pedal': 'string',
          'phraseend': 'string',
          'relativeroot': 'string',
          'repeats': 'string',
          'section': 'Int64',
          'staff': 'Int64',
          'tied': 'Int64',
          'timesig': 'string',
          'tpc': 'Int64',
          'voice': 'Int64',
          'voices': 'Int64',
          'volta': 'Int64'}


def frac_or_empty(val):
    """
    Helper function to convert Fractional data types into Fractions while handling Nans
    and empty strings.
    
    Parameters
    ----------
    val : string
        A String to convert into a fraction (if it is not empty).
        
    Returns
    -------
    fraction : Fraction or string
        The empty string, if the input is the empty string. Or, a Fraction parsed from
        the input.
    """
    return '' if val == '' else frac(val)

# Converters for different data types
CONVERTERS = {'act_dur': frac_or_empty,
              'beatsize': frac_or_empty,
              'cn': parse_lists_of_int_tuples,
              'ncn': parse_lists_of_int_tuples,
              'chord_length':frac_or_empty,
              'chord_tones': parse_lists_of_int,
              'duration':frac_or_empty,
              'next': parse_lists_of_int,
              'nominal_duration': frac_or_empty,
              'onset':frac_or_empty,
              'onset_next':frac_or_empty,
              'scalar':frac_or_empty,
              'subbeat': frac_or_empty,}



def read_dump(file, index_col=[0,1], converters={}, dtypes={}, **kwargs):
    """
    Read a corpus tsv file into a pandas DataFrame.
    
    Parameters
    ----------
    file : string
        The tsv file to parse.
        
    index_col : int or list(int)
        The index (or indices) of column(s) to use as the index.
        
    converters : dict
        Converters which will be passed to the pandas read_csv function. These will
        overwrite/be added to the default list of CONVERTERS.
        
    dtypes : dict
        Dtypes which will be passed to the pandas read_csv function. These will
        overwrite/be added to the default list of DTYPES.
        
    Returns
    -------
    df : pd.DataFrame
        The pandas DataFrame, parsed from the given tsv file.
    """
    conv = CONVERTERS
    types = DTYPES
    types.update(dtypes)
    conv.update(converters)
    return pd.read_csv(file, sep='\t', index_col=index_col, dtype=types, converters=conv, **kwargs)