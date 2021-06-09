"""Defines constants used for the DCML corpus tsv files."""
from fractions import Fraction
from typing import Tuple

import pandas as pd

MEASURE_OFFSET = "mc_offset"
CHORD_ONSET_BEAT = "mn_onset"
NOTE_ONSET_BEAT = "mn_onset"


# Helper functions to be used as converters, handling empty strings
def str2inttuple(string: str) -> Tuple[int]:
    """
    Convert the given string (with ", " separating ints) into a tuple of integers.

    Parameters
    ----------
    string : str
        A string of the format "i1, i2, i3...".

    Returns
    -------
    int_tuple : Tuple[int]
        A tuple of the integers contained in the string. An empty string will return
        an empty tuple.
    """
    return tuple() if string == "" else tuple(int(s) for s in string.split(", "))


def int2bool(s: str) -> bool:
    """
    Convert the given int string into a boolean. Useful for the error case which
    returns pd.NA.

    Parameters
    ----------
    s : str
        A string of an integer, which represents a boolean ("0" or "1").

    Returns
    -------
    boolean : bool
        False if the string is "0", True if the string is some other integer.
        pd.NA for an error.
    """
    try:
        return bool(int(s))
    except Exception:
        return pd.NA


def str2frac(s: str) -> Fraction:
    """
    Convert the given string into a Fraction. Useful for the error case which returns pd.NA.

    Parameters
    ----------
    s : str
        A string of a fraction.

    Returns
    -------
    frac : Fraction
        The given string converted to a fraction, or pd.NA on any error.
    """
    try:
        return Fraction(s)
    except Exception:
        return pd.NA


CONVERTERS = {
    "added_tones": str2inttuple,
    "act_dur": str2frac,
    "chord_tones": str2inttuple,
    "globalkey_is_minor": int2bool,
    "localkey_is_minor": int2bool,
    "mc_offset": str2frac,
    "mc_onset": str2frac,
    "mn_onset": str2frac,
    "next": str2inttuple,
    "nominal_duration": str2frac,
    "offset": str2frac,
    "onset": str2frac,
    "duration": str2frac,
    "scalar": str2frac,
}


DTYPES = {
    "alt_label": str,
    "barline": str,
    "bass_note": "Int64",
    "cadence": str,
    "cadences_id": "Int64",
    "changes": str,
    "chord": str,
    "chord_type": str,
    "dont_count": "Int64",
    "figbass": str,
    "form": str,
    "globalkey": str,
    "gracenote": str,
    "harmonies_id": "Int64",
    "keysig": "Int64",
    "label": str,
    "localkey": str,
    "mc": "Int64",
    "midi": "Int64",
    "mn": "Int64",
    "notes_id": "Int64",
    "numbering_offset": "Int64",
    "numeral": str,
    "pedal": str,
    "playthrough": "Int64",
    "phraseend": str,
    "relativeroot": str,
    "repeats": str,
    "root": "Int64",
    "special": str,
    "staff": "Int64",
    "tied": "Int64",
    "timesig": str,
    "tpc": "Int64",
    "voice": "Int64",
    "voices": "Int64",
    "volta": "Int64",
}
