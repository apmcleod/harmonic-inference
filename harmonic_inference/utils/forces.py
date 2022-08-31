"""A module containing functions to load forced labels from Musescore3 files."""
import bisect
import logging
import re
from fractions import Fraction
from glob import glob
from pathlib import Path
from typing import Dict, Set, Tuple, Union

import pandas as pd
from ms3 import Score

from harmonic_inference.data.data_types import ChordType, KeyMode, PitchType
from harmonic_inference.data.piece import ScorePiece
from harmonic_inference.utils.harmonic_utils import (
    decode_relative_keys,
    get_chord_inversion,
    get_chord_one_hot_index,
    get_key_one_hot_index,
    get_pitch_from_string,
)

NO_CHORD_CHANGE_REGEX = re.compile(r"^\=C")
NO_KEY_CHANGE_REGEX = re.compile(r"\=K")
CHORD_CHANGE_REGEX = re.compile(r"\!C")
KEY_CHANGE_REGEX = re.compile(r"\!K")

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
    r"(/(((((#{1,2}|b{1,2})?)(I|II|III|IV|V|VI|VII|i|ii|iii|iv|v|vi|vii)|"  # Applied root
    f"{ABS_PITCH_REGEX_STRING})/?)*))?$"  # Applied root (con't)
)
CHORD_REGEX = re.compile(CHORD_REGEX_STRING)
KEY_REGEX = re.compile(f"Key=({ABS_PITCH_REGEX_STRING}|{REL_PITCH_REGEX_STRING})")

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

    return bisect.bisect_left(note_positions, (mc, mn_onset))


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
    Dict[int, Tuple[Union[Tuple[int, str], Tuple[str, ChordType, int, str]], str]],
    Dict[int, Tuple[Union[int, str], str]],
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

    chords : Dict[int, Tuple[Union[Tuple[int, str], Tuple[str, ChordType, int, str]], str]]
        A dictionary mapping a note index to a tuple containing a chord label, and a
        chord label type (either "abs" or "rel").
        If "abs", the chord label is a tuple of the absolute chord's one-hot index and
        the changes string.
        If "rel", the chord label is a tuple containing the chord root (string of a Roman
        numeral), the chord type, and the chord's inversion, plus the chord's changes
        string.

    keys : Dict[int, Tuple[Union[int, str], str]]
        A dictional mapping a note index to a tuple containing a key label and a label type
        string (either "abs" or "rel").
        If abs, the key label is an absolute key one-hot index. If rel, the key label
        is a Roman numeral string.
    """

    def decode_key(
        tonic_str: str,
        global_tonic: Union[str, int],
        global_mode: KeyMode,
        mc: int,
        mn_onset: Fraction,
    ) -> Tuple[Union[str, int], KeyMode, str]:
        """
        Decode the given (absolute or relative) key into a tonic and mode.

        Parameters
        ----------
        tonic_str : str
            An absolute or relative key string. Can also be a relative root with slashes.
        global_tonic : Union[str, int]
            The current surrounding tonic to which the given key is relative.
        global_mode : KeyMode
            The current surrounding mode to which the given key is relative.
        mc : int
            The mc of this key label, for error printing.
        mn_onset : Fraction
            The mn_onset of this key label, for error printing.

        Returns
        -------
        tonic : Union[str, int]
            The tonic of the given key, either absolute (int) if it was possible to
            calculate, or as a relative string (to be interpreted relative to the global
            key).
        mode : KeyMode
            The mode of the given key.
        tonic_type : str
            "abs" or "rel", depending on whether the returned tonic is an absolute int
            or a relative string.
        """
        mode = KeyMode.MINOR if tonic_str.split("/")[0].islower() else KeyMode.MAJOR

        if any([numeral in tonic_str for numeral in "VvIi"]):
            if global_tonic is None or global_mode is None or isinstance(global_tonic, str):
                id_type = "rel"
                logging.warning(
                    (
                        "Unknown tonic for relative label %s "
                        "at position (%s, %s). Storing force as relative string."
                    ),
                    label,
                    mc,
                    mn_onset,
                )
                tonic = tonic_str
            else:
                tonic, mode = decode_relative_keys(
                    tonic_str, global_tonic, global_mode, PitchType.TPC
                )
                id_type = "abs"

        else:
            id_type = "abs"
            tonic = get_pitch_from_string(tonic_str.split("/")[0], PitchType.TPC)

        return tonic, mode, id_type

    score = Score(score_path)
    labels: pd.DataFrame = score.annotations.get_labels(decode=True)

    chord_changes = set()
    chord_non_changes = set()
    key_changes = set()
    key_non_changes = set()
    chord_ids = dict()
    key_ids = dict()

    global_tonic = None
    global_mode = None
    local_tonic = None
    local_mode = None
    relative_tonic = None
    relative_mode = None
    for mc, mn_onset, label in zip(labels["mc"], labels["mn_onset"], labels["label"]):
        note_index = convert_score_position_to_note_index(mc, mn_onset, piece)

        added = False
        for regex, index_set in zip(
            [CHORD_CHANGE_REGEX, NO_CHORD_CHANGE_REGEX, KEY_CHANGE_REGEX, NO_KEY_CHANGE_REGEX],
            [chord_changes, chord_non_changes, key_changes, key_non_changes],
        ):
            match = regex.match(label)
            if match and match.group(0) == label:
                index_set.add(note_index)
                added = True
                if regex == KEY_CHANGE_REGEX:
                    local_tonic = None
                    local_mode = None
                    relative_tonic = None
                    relative_mode = None
                break
        if added:
            continue

        key_match = KEY_REGEX.match(label)
        if key_match and key_match.group(0) == label:
            tonic_str = key_match.group(1)
            local_tonic, local_mode, id_type = decode_key(
                tonic_str, global_tonic, global_mode, mc, mn_onset
            )
            if id_type == "abs":
                key_id = get_key_one_hot_index(local_mode, local_tonic, PitchType.TPC)
                if note_index == 0:
                    global_tonic = local_tonic
                    global_mode = local_mode
            else:
                key_id = local_tonic

            relative_tonic = local_tonic
            relative_mode = local_mode
            if note_index in key_ids and key_ids[note_index][0] != key_id:
                logging.warning(
                    "Overwriting key %s at position (%s, %s) with %s based on label %s",
                    key_ids[note_index][0],
                    mc,
                    mn_onset,
                    key_id,
                    label,
                )
            key_ids[note_index] = (key_id, id_type)

        # Can include key and chord, plus relative roots (also modeled as key changes)
        dcml_match = DCML_LABEL_REGEX.match(label)
        if dcml_match and dcml_match.group(0) == label:
            if "." in label:
                # Label has chord and key: handle the key here and save only the chord label
                idx = label.index(".")
                tonic_str = label[:idx]
                label = label[idx + 1 :]

                local_tonic, local_mode, id_type = decode_key(
                    tonic_str, global_tonic, global_mode, mc, mn_onset
                )
                if id_type == "abs":
                    key_id = get_key_one_hot_index(local_mode, local_tonic, PitchType.TPC)
                    if note_index == 0:
                        global_tonic = local_tonic
                        global_mode = local_mode
                else:
                    key_id = local_tonic

                relative_tonic = local_tonic
                relative_mode = local_mode
                if note_index in key_ids and key_ids[note_index][0] != key_id:
                    logging.warning(
                        "Overwriting key %s at position (%s, %s) with %s based on label %s",
                        key_ids[note_index][0],
                        mc,
                        mn_onset,
                        key_id,
                        label,
                    )
                key_ids[note_index] = (key_id, id_type)

            # Label is now only a chord label. We can match it to get groups.
            chord_match = CHORD_REGEX.match(label)
            root_string = chord_match.group(1)
            type_string = chord_match.group(5)
            figbass_string = chord_match.group(6)
            changes_string = chord_match.group(8)
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

            # Handle relroot_string (add to key force and set relative tonic and mode)
            if relroot_string is not None:
                if local_tonic is None or local_mode is None:
                    logging.warning(
                        (
                            "Unknown local key for symbol %s at position (%s, %s). "
                            "Ignoring relative root."
                        ),
                        relroot_string,
                        mc,
                        mn_onset,
                    )
                else:
                    relative_tonic, relative_mode, id_type = decode_key(
                        relroot_string, local_tonic, local_mode, mc, mn_onset
                    )
                    if id_type == "abs":
                        key_id = get_key_one_hot_index(relative_mode, relative_tonic, PitchType.TPC)
                    else:
                        relative_tonic = f"{relative_tonic}/{local_tonic}"
                        key_id = relative_tonic
                    if note_index in key_ids and key_ids[note_index][0] != key_id:
                        logging.warning(
                            "Overwriting key %s at position (%s, %s) with %s based on label %s",
                            key_ids[note_index][0],
                            mc,
                            mn_onset,
                            key_id,
                            label,
                        )
                    key_ids[note_index] = (key_id, id_type)

            # Back to handling chord
            if any([numeral in root_string for numeral in "VvIi"]):
                id_type = "rel"
                if isinstance(relative_tonic, int):
                    chord_root, _, id_type = decode_key(
                        root_string, relative_tonic, relative_mode, mc, mn_onset
                    )
                if id_type == "rel":
                    chord_id = (root_string, chord_type, inversion, changes_string)
                else:
                    chord_id = (
                        get_chord_one_hot_index(
                            chord_type,
                            chord_root,
                            PitchType.TPC,
                            inversion=inversion,
                        ),
                        changes_string,
                    )

                if relative_tonic is not None and relative_mode is not None:
                    # Also force key here, since the chord label is relative!
                    if isinstance(relative_tonic, str):
                        key_id = relative_tonic
                    else:
                        key_id = get_key_one_hot_index(relative_mode, relative_tonic, PitchType.TPC)
                    if note_index in key_ids and key_ids[note_index][0] != key_id:
                        logging.warning(
                            "Overwriting key %s at position (%s, %s) with %s based on label %s",
                            key_ids[note_index][0],
                            mc,
                            mn_onset,
                            key_id,
                            label,
                        )
                    key_ids[note_index] = (key_id, id_type)

            else:
                chord_id = (
                    get_chord_one_hot_index(
                        chord_type,
                        chord_root,
                        PitchType.TPC,
                        inversion=inversion,
                    ),
                    changes_string,
                )

            chord_ids[note_index] = (chord_id, id_type)

    return (
        chord_changes,
        chord_non_changes,
        key_changes,
        key_non_changes,
        chord_ids,
        key_ids,
    )
