"""A module containing functions to load forced labels from Musescore3 files."""
import bisect
import logging
import re
from fractions import Fraction
from glob import glob
from pathlib import Path
from typing import Set, Tuple, Union

import pandas as pd
from ms3 import Score

from harmonic_inference.data.data_types import ChordType, KeyMode, PitchType
from harmonic_inference.data.piece import ScorePiece
from harmonic_inference.utils.harmonic_utils import (
    decode_relative_keys,
    get_chord_inversion,
    get_chord_one_hot_index,
    get_key_from_one_hot_index,
    get_key_one_hot_index,
    get_pitch_from_string,
)

NO_CHORD_CHANGE_REGEX = r"^\=C"
NO_KEY_CHANGE_REGEX = r"\=K"
CHORD_CHANGE_REGEX = r"\!C"
KEY_CHANGE_REGEX = r"\!K"

ACCIDENTAL_REGEX_STRING = "(#{1,2}|b{1,2})?"
ABS_PITCH_REGEX_STRING = f"[A-Ga-g]{ACCIDENTAL_REGEX_STRING}"
REL_PITCH_REGEX_STRING = (
    f"{ACCIDENTAL_REGEX_STRING}(I|II|III|IV|V|VI|VII|i|ii|iii|iv|v|vi|vii|Ger|It|Fr)"
)

CHORD_REGEX_STRING = (
    f"({ABS_PITCH_REGEX_STRING}|{REL_PITCH_REGEX_STRING})"  # Root
    r"(%|o|M|\+|\+M)?"  # Chord type
    r"(7|65|43|42|2|64|6)?"  # Fig bass (inversion)
    r"(\((((\+|-|\^|v)?(#{1,2}|b{1,2})?\d)+)\))?"  # Chord pitches
    r"(/((((#{1,2}|b{1,2})?)(I|II|III|IV|V|VI|VII|i|ii|iii|iv|v|vi|vii)/?)*))?"  # Applied root
)
CHORD_REGEX = re.compile(CHORD_REGEX_STRING)
KEY_REGEX = re.compile(f"Key: ({ABS_PITCH_REGEX_STRING}|{REL_PITCH_REGEX_STRING})")

DCML_LABEL_REGEX = re.compile(
    f"(({ABS_PITCH_REGEX_STRING}|{REL_PITCH_REGEX_STRING}).)?{CHORD_REGEX_STRING}"
)


def convert_score_position_to_note_index(mc: int, mn_onset: Fraction, piece: ScorePiece) -> int:
    """
    Convert a score position (given as an mc and an mn_onset) into a note-index for the given
    piece.

    Parameters
    ----------
    mc : int
        The score position's mc.

    mn_onset : Fraction
        The score position's mn_onset.

    piece : ScorePiece
        A score in which to extract note indexes.

    Returns
    -------
    note_index : int
        The note index of the given score position.
    """
    note_positions = [note.onset for note in piece.get_inputs()]

    index = bisect.bisect_left(note_positions, (mc, mn_onset))

    if note_positions[index] != (mc, mn_onset):
        raise ValueError(
            f"Position ({mc}, {mn_onset}) is not a note onset. Closest is {note_positions[index]}"
        )

    return index


def find_forces_musescore_file_for_piece(piece: ScorePiece, forces_dir: Path) -> Path:
    """
    Given a piece and a directory containing labeled musescore3 files, find the musescore3 file
    corresponding to the given piece.

    Parameters
    ----------
    piece : ScorePiece
        The piece whose corresponding musescore3 file to find and return.

    forces_dir : Path
        The directory containing potentially matching files.

    Returns
    -------
    score_path : Path
        The Path to the corresponding score file, if found. Otherwise, None.
    """
    piece_name = Path(piece.name.split(" ")[-1])
    matches = glob(str(forces_dir / "**" / piece_name.stem) + "*.mscx", recursive=True)
    if len(matches) == 0:
        return None

    if len(matches) == 1:
        return matches[0]

    logging.info("Multiple matches found. Searching for the best match.")
    for match in matches:
        if str(piece_name.parent) in match:
            return match

    logging.info("No great match found. Returning the first match.")
    return matches[0]


def extract_forces_from_musescore(
    score_path: Union[str, Path], piece: ScorePiece
) -> Tuple[
    Set[int],
    Set[int],
    Set[int],
    Set[int],
    Tuple[int, Union[Tuple[int, str], Tuple[str, ChordType, int, str]], str],
    Tuple[int, Union[int, str], str],
]:
    """
    Extract forced labels, changes, and non-changes from a Musescore3 file.

    Parameters
    ----------
    score_path : Union[str, Path]
        The path to the Musescore3 file which contains the labels.

    piece : ScorePiece
        The piece whose forces we are loading. To convert (mc, mn_onset) positions into
        note indexes.

    Returns
    -------
    chord_changes : Set[int]
       A set of note indexes indicating positions at which there must be a chord change.

    chord_non_changes : Set[int]
       A set of note indexes indicating positions at which there must NOT be a chord change.

    key_changes : Set[int]
       A set of note indexes indicating positions at which there must be a key change.

    key_non_changes : Set[int]
       A set of note indexes indicating positions at which there must NOT be a key change.

    chords : Tuple[int, Union[Tuple[int, str], Tuple[str, ChordType, int, str]], str]
        Tuples of (note_index, chord_id, type) indicating positions at which a given chord label
        is forced. Type may be either "abs" or "rel", denoting the type of chord_id used
        If abs, chord_id is a tuple containing the one-hot chord id and a string of the changes.
        If rel, chord_id is a tuple containing the (string) relative root, the chord type,
        the inversion, and a string of the changes.

    keys : Tuple[int, Union[int, str], str]
        Tuples of (note_index, key_id, type) indicating positions at which a given key label
        is forced. Type may be either "abs" or "rel", denoting the type of key_id used.
        If abs, the key_id is a one-hot key id. If rel, a label string is given in that slot
        instead (since RN label intervals are dependant on the local mode). Such label strings
        are formatted like relativeroots (slash-separated Roman numerals).
    """
    score = Score(score_path)

    labels: pd.DataFrame = score.annotations.get_labels()

    chord_changes = pd.concat(
        [labels.loc[labels["label"].str.fullmatch(CHORD_CHANGE_REGEX), ["mc", "mn_onset"]]]
    )
    chord_changes = [
        convert_score_position_to_note_index(mc, mn_onset, piece)
        for mc, mn_onset in zip(chord_changes["mc"], chord_changes["mn_onset"])
    ]

    chord_non_changes = pd.concat(
        [labels.loc[labels["label"].str.fullmatch(NO_CHORD_CHANGE_REGEX), ["mc", "mn_onset"]]]
    )
    chord_non_changes = [
        convert_score_position_to_note_index(mc, mn_onset, piece)
        for mc, mn_onset in zip(chord_non_changes["mc"], chord_non_changes["mn_onset"])
    ]

    key_changes = pd.concat(
        [labels.loc[labels["label"].str.fullmatch(KEY_CHANGE_REGEX), ["mc", "mn_onset"]]]
    )
    key_changes = [
        convert_score_position_to_note_index(mc, mn_onset, piece)
        for mc, mn_onset in zip(key_changes["mc"], key_changes["mn_onset"])
    ]

    key_non_changes = pd.concat(
        [labels.loc[labels["label"].str.fullmatch(NO_KEY_CHANGE_REGEX), ["mc", "mn_onset"]]]
    )
    key_non_changes = [
        convert_score_position_to_note_index(mc, mn_onset, piece)
        for mc, mn_onset in zip(key_non_changes["mc"], key_non_changes["mn_onset"])
    ]

    chord_ids = []
    key_ids = []

    keys = pd.concat(
        [labels.loc[labels["label"].str.fullmatch(KEY_REGEX), ["mc", "mn_onset", "label"]]]
    )
    for mc, mn_onset, label in zip(keys["mc"], keys["mn_onset"], keys["label"]):
        tonic_str = KEY_REGEX.match(label).group(1)
        mode = KeyMode.MINOR if tonic_str.islower() else KeyMode.MAJOR

        if any([numeral in tonic_str for numeral in ["v", "V", "i", "I"]]):
            id_type = "rel"
            key_id = tonic_str

        else:
            id_type = "abs"
            key_id = get_key_one_hot_index(
                mode, get_pitch_from_string(tonic_str, PitchType.TPC), PitchType.TPC
            )

        key_ids.append((convert_score_position_to_note_index(mc, mn_onset, piece), key_id, id_type))

    # Can include key and chord, plus relative roots (also modeled as key changes)
    dcml_labels = pd.concat(
        [labels.loc[labels["label"].str.fullmatch(DCML_LABEL_REGEX), ["mc", "mn_onset", "label"]]]
    )
    for mc, mn_onset, label in zip(
        dcml_labels["mc"], dcml_labels["mn_onset"], dcml_labels["label"]
    ):
        if "." in label:
            # Label has chord and key: handle the key here and save only the chord label
            idx = label.index(".")
            tonic_str = label[:idx]
            label = label[idx + 1 :]

            # Handle key label
            if any([numeral in tonic_str for numeral in ["v", "V", "i", "I"]]):
                id_type = "rel"
                key_id = tonic_str

            else:
                id_type = "abs"
                key_id = get_key_one_hot_index(
                    KeyMode.MINOR if tonic_str[0].islower() else KeyMode.MAJOR,
                    get_pitch_from_string(tonic_str, PitchType.TPC),
                    PitchType.TPC,
                )

            key_ids.append(
                (convert_score_position_to_note_index(mc, mn_onset, piece), key_id, id_type)
            )

        # Label is now only a chord label. We can match it to get groups.
        chord_match = CHORD_REGEX.match(label)
        root_string = chord_match.group(1)
        type_string = chord_match.group(5)
        figbass_string = chord_match.group(6)
        changes_string = chord_match.group(7)
        relroot_string = chord_match.group(13)

        # Get chord features
        is_minor = root_string.islower()
        inversion = get_chord_inversion(figbass_string)
        if figbass_string in ["7", "65", "43", "2"]:
            # 7th chord
            chord_type = {
                "o": ChordType.DIM7,
                "%": ChordType.HALF_DIM7,
                "+": ChordType.AUG_MIN7,
                "+M": ChordType.AUG_MAJ7,
                "M": ChordType.MIN_MAJ7 if is_minor else ChordType.MAJ_MAJ7,
                None: ChordType.MIN_MIN7 if is_minor else ChordType.MAJ_MIN7,
            }[type_string]
        else:
            # Triad
            chord_type = {
                "o": ChordType.DIMINISHED,
                "+": ChordType.AUGMENTED,
                None: ChordType.MINOR if is_minor else ChordType.MAJOR,
            }[type_string]

        if any([numeral in root_string for numeral in ["v", "V", "i", "I"]]):
            id_type = "rel"
            chord_id = (root_string, chord_type, inversion, changes_string)

        else:
            id_type = "abs"
            chord_id = (
                get_chord_one_hot_index(
                    chord_type,
                    get_pitch_from_string(root_string, PitchType.TPC),
                    PitchType.TPC,
                    inversion=inversion,
                ),
                changes_string,
            )

        chord_ids.append(
            (convert_score_position_to_note_index(mc, mn_onset, piece), chord_id, id_type)
        )

        # Handle relroot_string (add to existing key force)
        if relroot_string is not None:
            found = False
            for i, (key_mc, key_mn_onset, key_id, key_id_type) in enumerate(key_ids):
                if key_mc == mc and key_mn_onset == mn_onset:
                    found = True
                    if key_id_type == "abs":
                        tonic, mode = get_key_from_one_hot_index(key_id, PitchType.TPC)
                        tonic, mode = decode_relative_keys(
                            relroot_string, tonic, mode, PitchType.TPC
                        )
                        key_ids[i] = (
                            key_mc,
                            key_mn_onset,
                            get_key_one_hot_index(mode, tonic, PitchType.TPC),
                            key_id_type,
                        )

                    else:
                        # Relative: Just append relroot to previous relative key
                        key_ids[i] = (
                            key_mc,
                            key_mn_onset,
                            f"{relroot_string}/{key_id}",
                            key_id_type,
                        )

            if not found:
                # Here, there is no real way to represent this during the search, so it is ignored
                logging.warning(
                    "Ignoring relative root of forced %s (relative roots are ignored for forces)",
                    label,
                )

    return (
        chord_changes,
        chord_non_changes,
        key_changes,
        key_non_changes,
        chord_ids,
        key_ids,
    )
