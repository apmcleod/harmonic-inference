"""An object to represent Key information for music."""
import inspect
import logging
from typing import Dict

import numpy as np
import pandas as pd

from harmonic_inference.data.data_types import KeyMode, PitchType
from harmonic_inference.utils.harmonic_constants import (
    MAX_KEY_CHANGE_INTERVAL_TPC,
    MIN_KEY_CHANGE_INTERVAL_TPC,
    NUM_PITCHES,
)
from harmonic_inference.utils.harmonic_utils import (
    absolute_to_relative,
    decode_relative_keys,
    get_interval_from_scale_degree,
    get_key_one_hot_index,
    get_pitch_from_string,
    get_pitch_string,
    transpose_pitch,
)


class Key:
    """
    A musical key, with tonic and mode.
    """

    def __init__(
        self,
        relative_tonic: int,
        local_tonic: int,
        global_tonic: int,
        relative_mode: KeyMode,
        local_mode: KeyMode,
        global_mode: KeyMode,
        tonic_type: PitchType,
    ):
        """
        Create a new musical key object.

        Parameters
        ----------
        relative_tonic : int
            An integer representing the pitch class of the tonic of this key, including applied
            roots. If tonic_type is TPC, this is stored as a tonal pitch class (with C = 0,
            G = 1, etc. around the circle of fifths). If tonic_type is MIDI, this is stored as
            semitones above C (with C, B# = 0).
        local_tonic : int
            An integer representing the pitch class of the tonic of this key without taking
            applied roots into account, in the same format as relative_tonic.
        global_tonic : int
            An integer representing the pitch class of the tonic of the global key.
        relative_mode : KeyMode
            The mode of this key, including applied roots.
        local_mode : KeyMode
            The mode of this key, without taking applied roots into account.
        global_mode : KeyMode
            The mode of the global key of this piece.
        tonic_type : PitchType
            The PitchType in which this key's tonic is stored. If this is TPC, the
            tonic can be later converted into MIDI type, but not vice versa.
        """
        self.relative_tonic = relative_tonic
        self.local_tonic = local_tonic
        self.global_tonic = global_tonic

        self.relative_mode = relative_mode
        self.local_mode = local_mode
        self.global_mode = global_mode

        self.tonic_type = tonic_type

        self.params = inspect.getfullargspec(Key.__init__).args[1:]

    def to_pitch_type(self, pitch_type: PitchType) -> "Key":
        """
        Return a new Key with the given pitch type. Note that while the TPC -> MIDI conversion
        is well-defined, it is also lossy: the MIDI -> TPC conversion must arbitrarily choose
        a matching TPC pitch.

        Parameters
        ----------
        pitch_type : PitchType
            The desired pitch type.

        Returns
        -------
        key : Key
            The resulting key. A copy of this key, if the given pitch_type matches the key's
            tonic PitchType already.
        """
        new_params = {key: getattr(self, key) for key in self.params}
        new_params["tonic_type"] = pitch_type

        if pitch_type == self.tonic_type:
            return Key(**new_params)

        # Convert relative, local, and global tonic
        for key in ["relative_tonic", "local_tonic", "global_tonic"]:
            new_params[key] = get_pitch_from_string(
                get_pitch_string(new_params[key], self.tonic_type), pitch_type
            )

        return Key(**new_params)

    def get_key_change_vector_length(self, one_hot: bool = True) -> int:
        """
        Get the length of this Key's key change vector.

        Parameters
        ----------
        one_hot : bool
            True to return the length of a one-hot key change vector.

        Returns
        -------
        length : int
            The length of a single key-change vector of this Key.
        """
        return get_key_change_vector_length(self.tonic_type, one_hot=one_hot)

    def get_key_change_vector(self, next_key: "Key") -> np.array:
        """
        Get a non-one-hot key change vector.

        Parameters
        ----------
        next_key : Key
            The next key that this one is changing to.

        Returns
        -------
        change_vector : np.array
            The non-one hot key change vector representing this key change.
        """
        change_vector = np.zeros(
            self.get_key_change_vector_length(one_hot=False),
            dtype=np.float16,
        )

        # Relative tonic
        try:
            change_vector[
                absolute_to_relative(
                    next_key.relative_tonic,
                    self.relative_tonic,
                    self.tonic_type,
                    True,
                    pad=False,
                )
            ] = 1
        except ValueError:
            logging.warning(
                "Key change from {prev_key} to {key} falls outside of TPC key change range. "
                "This key change target vector will not have a 1 for the key tonic."
            )

        # Absolute mode of next key
        change_vector[-2 + next_key.relative_mode.value] = 1

        return change_vector

    def get_one_hot_index(self) -> int:
        """
        Get the one-hot index of this key.

        Returns
        -------
        index : int
            This Key's one-hot index.
        """
        return get_key_one_hot_index(
            self.relative_mode,
            self.relative_tonic,
            self.tonic_type,
        )

    def get_key_change_one_hot_index(self, next_key: "Key") -> int:
        """
        Get the key change as a one-hot index. The one-hot index is based on the mode of the next
        key and the interval from this key to the next one.

        Parameters
        ----------
        next_key : Key
            The next key in sequence.

        Returns
        -------
        index : int
            The one hot index of this key change.
        """
        interval = absolute_to_relative(
            next_key.relative_tonic,
            self.relative_tonic,
            self.tonic_type,
            True,
            pad=False,
        )

        if self.tonic_type == PitchType.MIDI:
            num_pitches = NUM_PITCHES[PitchType.MIDI]
        else:
            num_pitches = MAX_KEY_CHANGE_INTERVAL_TPC - MIN_KEY_CHANGE_INTERVAL_TPC

        return next_key.relative_mode.value * num_pitches + interval

    def is_repeated(self, other: "Key", use_relative: bool = True) -> bool:
        """
        Detect if a given key can be regarded as a repeat of this one in terms of tonic and
        mode.

        Parameters
        ----------
        other : Key
            The other key to check for repeat.
        use_relative : bool
            True to take use relative_tonic and relative_mode.
            False to use local_tonic and local_mode.

        Returns
        -------
        is_repeated : bool
            True if the given key is a repeat of this one. False otherwise.
        """
        return self.equals(other, use_relative=use_relative)

    def equals(self, other: "Key", use_relative: bool = True) -> bool:
        """
        Check if the tonic and mode of this Key are the same.

        Parameters
        ----------
        other : Key
            The other key to check for equality.
        use_relative : bool
            True to take use relative_tonic and relative_mode.
            False to use local_tonic and local_mode.

        Returns
        -------
        equals : bool
            True if the keys are equal. False otherwise.
        """
        if not isinstance(other, Key):
            return False

        attr_names = ["tonic_type"]
        if use_relative:
            attr_names.extend(["relative_tonic", "relative_mode"])
        else:
            attr_names.extend(["local_tonic", "local_mode"])

        for attr_name in attr_names:
            if getattr(self, attr_name) != getattr(other, attr_name):
                return False
        return True

    def to_dict(self) -> Dict:
        """
        Convert this Key to a dictionary, which can be called into the Key constructor as
        Key(**dict) to recreate a copy of this Key.

        Returns
        -------
        key_dict : Dict
            A dictionary representation of all of the fields of this Key.
        """
        return {field: getattr(self, field) for field in self.params}

    def __eq__(self, other: "Key") -> bool:
        if not isinstance(other, Key):
            return False
        for field in self.params:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def __repr__(self) -> str:
        params = ", ".join([f"{field}={getattr(self, field)}" for field in self.params])
        return f"Key({params})"

    def __str__(self) -> str:
        return f"{get_pitch_string(self.relative_tonic, self.tonic_type)} {self.relative_mode}"

    @staticmethod
    def from_series(chord_row: pd.Series, tonic_type: PitchType) -> "Key":
        """
        Create a Key object of the given pitch_type from the given pd.Series.

        Parameters
        ----------
        chord_row : pd.Series
            The chord row from which to make our Key object. It must contain at least the rows:
                'globalkey' (str): The global key A-G (major) or a-g (minor) with appended # and b.
                'globalkey_is_minor' (bool): True if the global key is minor. False if major.
                'localkey' (str): A Roman numeral representing the local key relative to the global
                                  key. E.g., 'biv' for a minor local key with a tonic on the flat-4
                                  of the global key.
                'localkey_is_minor' (bool): True if the local key is minor. False if major.
                'relativeroot' (str): The relative root for this chord, if any (otherwise null).
                                      Represented as 'r1', 'r1/r2', 'r1/r2/r3...'. The last
                                      relative root is relative to the local key, and each previous
                                      one is relative to that new applied key. Each root is in the
                                      same format as 'localkey'.
        tonic_type : PitchType
            The pitch type to use for the Key's tonic.

        Returns
        -------
        key : Key, or None
            The created Key object. If an error occurs, None is returned and the error is logged.
        """
        try:
            # Global key, absolute
            global_tonic = get_pitch_from_string(chord_row["globalkey"], pitch_type=tonic_type)
            global_mode = KeyMode.MINOR if chord_row["globalkey_is_minor"] else KeyMode.MAJOR

            # Local key is listed relative to global. We want it absolute.
            local_tonic, local_mode = decode_relative_keys(
                chord_row["localkey"], global_tonic, global_mode, tonic_type
            )

            # Treat applied dominants (and other slash chords) as new keys
            if not pd.isna(chord_row["relativeroot"]):
                relative_tonic, relative_mode = decode_relative_keys(
                    chord_row["relativeroot"], local_tonic, local_mode, tonic_type
                )
            else:
                relative_tonic, relative_mode = local_tonic, local_mode

            return Key(
                relative_tonic,
                local_tonic,
                global_tonic,
                relative_mode,
                local_mode,
                global_mode,
                tonic_type,
            )

        except Exception as exception:
            logging.error("Error parsing key from row %s", chord_row)
            logging.exception(exception)
            return None

    @staticmethod
    def from_labels_csv_row(
        chord_row: pd.Series,
        tonic_type: PitchType,
        global_key: "Key" = None,
    ) -> "Key":
        """
        Create a Key object of the given pitch_type from the given pd.Series taken
        from a labels csv file.

        Parameters
        ----------
        chord_row : pd.Series
            The chord row from which to make our Key object. It must contain at least the rows:
                'key' (str): The local key at the time of the given label. Flats and sharps
                             are represented with - and +, and major/minor is represented
                             by upper/lower-case.
                'degree' (str): The degree of the chord's root, relative to the local key,
                                and including applied chords with / notation. Flats and sharps
                                are represented with - and +.
        tonic_type : PitchType
            The pitch type to use for the Key's tonic.
        global_key : Key
            A Key which contains the global key for this piece (since the labels csv does
            not explicitly list the global key). If None, it is assumed that this key is
            the glboal key.

        Returns
        -------
        key : Key, or None
            The created Key object.
        """
        # Get local key
        key_str = chord_row["key"].replace("-", "b")
        key_str = key_str.replace("+", "#")

        local_tonic = get_pitch_from_string(key_str, tonic_type)
        local_mode = KeyMode.MAJOR if key_str[0].isupper() else KeyMode.MINOR

        # Get global key
        if global_key is None:
            global_tonic = local_tonic
            global_mode = local_mode
        else:
            global_tonic = global_key.global_tonic
            global_mode = global_key.global_mode

        # Get relative key
        relative_tonic = local_tonic
        relative_mode = local_mode

        degree_str = chord_row["degree"].replace("-", "b")
        degree_str = degree_str.replace("+", "#")

        if "/" in degree_str:
            degree_str, relative_root = degree_str.split("/")

            # Fix for labels_csv use #7 as standard in minor keys
            if "7" in relative_root and relative_mode == KeyMode.MINOR:
                if relative_root[0] == "b":
                    relative_root = relative_root[1:]
                else:
                    relative_root = "#" + relative_root

            # Fix for some labels adding to many sharps to 6 in minor
            if relative_mode == KeyMode.MINOR and "#6" in relative_root:
                relative_root = relative_root.replace("#6", "6")

            relative_transposition = get_interval_from_scale_degree(
                relative_root,
                True,
                relative_mode,
                pitch_type=tonic_type,
            )
            relative_tonic = transpose_pitch(
                relative_tonic,
                relative_transposition,
                pitch_type=tonic_type,
            )

            # TODO: Figure out relative mode
            if relative_mode == KeyMode.MAJOR:
                try:
                    relative_mode = {
                        "1": KeyMode.MAJOR,
                        "#1": KeyMode.MINOR,
                        "b2": KeyMode.MAJOR,
                        "2": KeyMode.MINOR,
                        "b3": KeyMode.MAJOR,
                        "3": KeyMode.MINOR,
                        "b4": KeyMode.MINOR,
                        "4": KeyMode.MAJOR,
                        "#4": KeyMode.MAJOR,
                        "5": KeyMode.MAJOR,
                        "#5": KeyMode.MAJOR,
                        "6": KeyMode.MINOR,
                        "b7": KeyMode.MAJOR,
                        "7": KeyMode.MINOR,
                    }[relative_root]
                except KeyError:
                    raise ValueError(f"Unknown mode for relative root in a major key: {chord_row}")
            else:
                try:
                    relative_mode = {
                        "b1": KeyMode.MAJOR,
                        "1": KeyMode.MINOR,
                        "b2": KeyMode.MAJOR,
                        "2": KeyMode.MINOR,
                        "3": KeyMode.MAJOR,
                        "b4": KeyMode.MINOR,
                        "4": KeyMode.MINOR,
                        "b5": KeyMode.MINOR,
                        "5": KeyMode.MINOR,
                        "6": KeyMode.MAJOR,
                        "b7": KeyMode.MINOR,
                        "7": KeyMode.MAJOR,
                        "#7": KeyMode.MAJOR,
                    }[relative_root]
                except KeyError:
                    raise ValueError(f"Unknown mode for relative root in a minor key: {chord_row}")

        return Key(
            relative_tonic,
            local_tonic,
            global_tonic,
            relative_mode,
            local_mode,
            global_mode,
            tonic_type,
        )


def get_key_change_vector_length(pitch_type: PitchType, one_hot: bool = True) -> int:
    """
    Get the length of a key change vector.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type for the given vector.
    one_hot : bool
        True to return the length of a one-hot key change vector.

    Returns
    -------
    length : int
        The length of a single key-change vector.
    """
    if pitch_type == PitchType.TPC:
        num_pitches = MAX_KEY_CHANGE_INTERVAL_TPC - MIN_KEY_CHANGE_INTERVAL_TPC
    elif pitch_type == PitchType.MIDI:
        num_pitches = 12
    else:
        raise ValueError(f"Invalid pitch_type: {pitch_type}")

    if one_hot:
        return num_pitches * len(KeyMode)
    return num_pitches + len(KeyMode)
