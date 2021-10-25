import bisect
import re
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from ms3 import Score

from harmonic_inference.data.data_types import KeyMode, PitchType
from harmonic_inference.data.piece import ScorePiece
from harmonic_inference.utils.harmonic_constants import CHORD_TYPE_TO_STRING, STRING_TO_CHORD_TYPE
from harmonic_inference.utils.harmonic_utils import (
    get_chord_one_hot_index,
    get_key_one_hot_index,
    get_pitch_from_string,
)

NO_CHORD_CHANGE_REGEX = r"^\=C"
NO_KEY_CHANGE_REGEX = r"\=K"
CHORD_CHANGE_REGEX = r"\!C"
KEY_CHANGE_REGEX = r"\!K"

PITCH_REGEX = r"[A-Ga-g](#{0,2}|b{0,2})"

CHORD_REGEX = re.compile(
    f"C=({PITCH_REGEX})"  # Root
    "(" + "|".join(list(CHORD_TYPE_TO_STRING.values())).replace("+", "\\+") + ")"  # Chord type
    r"(_[0-3])?"  # Inversion
)
KEY_REGEX = re.compile(f"K={PITCH_REGEX}")


def convert_score_positions_to_note_indexes(
    forces: Union[List[Tuple[int, Fraction]], List[Tuple[int, Fraction, int]]],
    piece: ScorePiece,
) -> Union[List[int], List[Tuple[int, int]]]:
    """
    Convert a list of forces whose positions are encoded as (mc, mn_onset) into
    one with positions encoded as note_indexes into the given piece.

    Parameters
    ----------
    forces : Union[List[Tuple[int, Fraction]], List[Tuple[int, Fraction, int]]]
        A list of forces, either (mc, mn_onset) tuples, or (mc, mn_onset, id) tuples.

    piece : ScorePiece
        A score in which to extract note indexes.

    Returns
    -------
    forces : Union[List[int], List[Tuple[int, int]]]
        A list of forces, where the (mc, mn_onset) position is converted into a note index.
    """
    note_positions = [note.onset for note in piece.get_inputs()]

    new_forces = [0] * len(forces)
    for i, force in enumerate(forces):
        index = bisect.bisect_left(note_positions, force[:2])

        if note_positions[index] != force[:2]:
            raise ValueError(
                f"Position {force[:2]} is not a note onset. Closest is {note_positions[index]}"
            )

        new_forces[i] = index if len(force) == 2 else (index, force[-1])

    return new_forces


def extract_forces_from_musescore(
    score_path: Union[str, Path]
) -> Tuple[
    Tuple[int, Fraction],
    Tuple[int, Fraction],
    Tuple[int, Fraction],
    Tuple[int, Fraction],
    Tuple[int, Fraction, int],
    Tuple[int, Fraction, int],
]:
    """
    Extract forced labels, changes, and non-changes from a Musescore3 file.

    Parameters
    ----------
    score_path : Union[str, Path]
        The path to the Musescore3 file which contains the labels.

    Returns
    -------
    chord_changes : Tuple[int, Fraction]
        Tuples of (mc, mn_onset) indicating positions at which there must be a chord change.

    chord_non_changes : Tuple[int, Fraction]
        Tuples of (mc, mn_onset) indicating positions at which there must NOT be a chord change.

    key_changes : Tuple[int, Fraction]
        Tuples of (mc, mn_onset) indicating positions at which there must be a key change.

    key_non_changes : Tuple[int, Fraction]
        Tuples of (mc, mn_onset) indicating positions at which there must NOT be a key change.

    chords : Tuple[int, Fraction, int]
        Tuples of (mc, mn_onset, chord_id) indicating positions at which a given chord label is
        forced.

    keys : Tuple[int, Fraction, int]
        Tuples of (mc, mn_onset, chord_id) indicating positions at which a given key label is
        forced.
    """
    score = Score(score_path)

    labels = score.annotations.get_labels()

    chord_labels = score.mscx.get_chords(lyrics=True)
    chord_labels = chord_labels.loc[~chord_labels["lyrics"].isnull()]
    chord_labels["label"] = chord_labels["lyrics"]

    chord_changes = pd.concat(
        [
            label_df.loc[label_df["label"].str.contains(CHORD_CHANGE_REGEX), ["mc", "mn_onset"]]
            for label_df in [labels, chord_labels]
        ]
    )
    chord_changes = [
        (mc, mn_onset) for mc, mn_onset in zip(chord_changes["mc"], chord_changes["mn_onset"])
    ]

    chord_non_changes = pd.concat(
        [
            label_df.loc[label_df["label"].str.contains(NO_CHORD_CHANGE_REGEX), ["mc", "mn_onset"]]
            for label_df in [labels, chord_labels]
        ]
    )
    chord_non_changes = [
        (mc, mn_onset)
        for mc, mn_onset in zip(chord_non_changes["mc"], chord_non_changes["mn_onset"])
    ]

    key_changes = pd.concat(
        [
            label_df.loc[label_df["label"].str.contains(KEY_CHANGE_REGEX), ["mc", "mn_onset"]]
            for label_df in [labels, chord_labels]
        ]
    )
    key_changes = [
        (mc, mn_onset) for mc, mn_onset in zip(key_changes["mc"], key_changes["mn_onset"])
    ]

    key_non_changes = pd.concat(
        [
            label_df.loc[label_df["label"].str.contains(NO_KEY_CHANGE_REGEX), ["mc", "mn_onset"]]
            for label_df in [labels, chord_labels]
        ]
    )
    key_non_changes = [
        (mc, mn_onset) for mc, mn_onset in zip(key_non_changes["mc"], key_non_changes["mn_onset"])
    ]

    chords = pd.concat(
        [
            label_df.loc[label_df["label"].str.contains(CHORD_REGEX), ["mc", "mn_onset", "label"]]
            for label_df in [labels, chord_labels]
        ]
    )
    chords = [
        (
            mc,
            mn_onset,
            get_chord_one_hot_index(
                STRING_TO_CHORD_TYPE[label_df[2]],
                get_pitch_from_string(label_df[0], PitchType.TPC),
                PitchType.TPC,
                inversion=0 if len(label_df[3]) == 0 else int(label_df[3][-1]),
            ),
        )
        for mc, mn_onset, (_, label_df) in zip(
            chords["mc"], chords["mn_onset"], chords["label"].str.extract(CHORD_REGEX).iterrows()
        )
    ]

    keys = pd.concat(
        [
            label_df.loc[label_df["label"].str.contains(KEY_REGEX), ["mc", "mn_onset", "label"]]
            for label_df in [labels, chord_labels]
        ]
    )
    keys = [
        (
            mc,
            mn_onset,
            get_key_one_hot_index(
                KeyMode.MAJOR if label[2].isupper() else KeyMode.MINOR,
                get_pitch_from_string(label[2:], PitchType.TPC),
                PitchType.TPC,
            ),
        )
        for mc, mn_onset, label in zip(keys["mc"], keys["mn_onset"], keys["label"])
    ]

    return (
        chord_changes,
        chord_non_changes,
        key_changes,
        key_non_changes,
        chords,
        keys,
    )
