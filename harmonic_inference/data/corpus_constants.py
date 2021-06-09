"""Defines constants used for the DCML corpus tsv files."""
from fractions import Fraction
from typing import Iterable, Tuple

import pandas as pd

MEASURE_OFFSET = "mc_offset"
CHORD_ONSET_BEAT = "mn_onset"
NOTE_ONSET_BEAT = "mn_onset"


# Helper functions to be used as converters, handling empty strings
def str2inttuple(string: str) -> Tuple[int]:
    return tuple() if string == "" else tuple(int(s) for s in string.split(", "))


def str2strtuple(string: str) -> Tuple[str]:
    return tuple() if string == "" else tuple(string.split(", "))


def iterable2str(iterable: Iterable) -> str:
    return ", ".join(str(s) for s in iterable)


def int2bool(s: str) -> bool:
    try:
        return bool(int(s))
    except Exception:
        return pd.NA


def str2frac(s: str) -> Fraction:
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
