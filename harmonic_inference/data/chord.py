"""A Chord object and functions for chords."""
import inspect
import logging
from fractions import Fraction
from typing import DefaultDict, Dict, Tuple, Union

import numpy as np
import pandas as pd

from harmonic_inference.data.data_types import NO_REDUCTION, ChordType, KeyMode, PitchType
from harmonic_inference.data.key import Key
from harmonic_inference.utils.harmonic_constants import (
    MAX_RELATIVE_TPC,
    MIN_RELATIVE_TPC,
    NUM_PITCHES,
    RELATIVE_TPC_EXTRA,
)
from harmonic_inference.utils.harmonic_utils import (
    absolute_to_relative,
    get_bass_note,
    get_chord_inversion,
    get_chord_inversion_count,
    get_chord_one_hot_index,
    get_chord_string,
    get_chord_type_from_string,
    get_interval_from_scale_degree,
    get_pitch_from_string,
    get_pitch_string,
    tpc_interval_to_midi_interval,
    transpose_pitch,
)
from harmonic_inference.utils.rhythmic_utils import get_metrical_level


class Chord:
    """
    A musical chord, with a root and base note.
    """

    def __init__(
        self,
        root: int,
        bass: int,
        key_tonic: int,
        key_mode: KeyMode,
        chord_type: ChordType,
        inversion: int,
        onset: Union[float, Tuple[int, Fraction]],
        onset_level: int,
        offset: Union[float, Tuple[int, Fraction]],
        offset_level: int,
        duration: Union[float, Fraction],
        pitch_type: PitchType,
    ):
        """
        Create a new musical chord object.

        Parameters
        ----------
        root : int
            An integer representing pitch class of this chord's root either as semitones above C
            (if pitch_type is MIDI; with B#, C = 0), or as tonal pitch class (if pitch_type is TPC;
            with C = 0, G = 1, etc. around the circle of fifths).
        bass : int
            An integer representing the bass note of this chord, either as semitones above C
            (if pitch_type is MIDI; with B#, C = 0), or as tonal pitch class (if pitch_type is TPC;
            with C = 0, G = 1, etc. around the circle of fifths).
        key_tonic : int
            An integer representing the pitch of the tonic of the key during this chord. Used to
            easily get the chords root and bass relative to the key.
        key_mode : KeyMode
            The mode of the current key.
        chord_type : ChordType
            The type of chord this is (major, minor, diminished, etc.)
        inversion : int
            The inversion this chord is in.
        onset : Union[float, Tuple[int, Fraction]]
            The onset position of this Chord. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        onset_level : int
            The metrical level on which the onset lies. 0=none, 1=subbeat, 2=beat, 3=downbeat.
        offset : Union[float, Tuple[int, Fraction]]
            The offset position of this Chord. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        offset_level : int
            The metrical level on which the offset lies. 0=none, 1=subbeat, 2=beat, 3=downbeat.
        duration : Union[float, Fraction]
            The duration of this Chord. Either a float (representing time in seconds), ar a
            Fraction (representing whole notes).
        pitch_type : PitchType
            The PitchType in which this chord's root and bass note are stored. If this is TPC, the
            values can be later converted into MIDI, but not vice versa.
        """
        self.root = root
        self.bass = bass
        self.key_tonic = key_tonic
        self.key_mode = key_mode
        self.chord_type = chord_type
        self.inversion = inversion
        self.onset = onset
        self.onset_level = onset_level
        self.offset = offset
        self.offset_level = offset_level
        self.duration = duration
        self.pitch_type = pitch_type

        self.params = inspect.getfullargspec(Chord.__init__).args[1:]

    def to_pitch_type(self, pitch_type: PitchType) -> "Chord":
        """
        Return a new Chord with the given pitch type. Note that while the TPC -> MIDI conversion
        is well-defined, it is also lossy: the MIDI -> TPC conversion must arbitrarily choose
        a matching TPC pitch.

        Parameters
        ----------
        pitch_type : PitchType
            The desired pitch type.

        Returns
        -------
        chord : Chord
            The resulting chord. A copy of this chord, if the given pitch_type matches the chord's
            PitchType already.
        """
        new_params = {key: getattr(self, key) for key in self.params}
        new_params["pitch_type"] = pitch_type

        if pitch_type == self.pitch_type:
            return Chord(**self.params)

        # Convert key_tonic, root, and bass
        for key in ["key_tonic", "root", "bass"]:
            new_params[key] = get_pitch_from_string(
                get_pitch_string(new_params[key], self.pitch_type), pitch_type
            )

        return Chord(**new_params)

    def get_one_hot_index(
        self,
        relative: bool = False,
        use_inversion: bool = True,
        pad: bool = True,
        reduction: Dict[ChordType, ChordType] = None,
    ) -> int:
        """
        Get the one-hot index of this chord.

        Parameters
        ----------
        relative : bool
            True to get the relative one-hot index. False for the absolute one-hot index.
        use_inversion : bool
            True to use inversions. False otherwise.
        pad : bool
            Only taken into account if self.pitch_type is TPC and relative is True.
            In that case, if pad is True, an additional padding is used around the valid.
        reduction : Dict[ChordType, ChordType]
            A reduction for the chord type of the given chord.

        Returns
        -------
        index : int
            This Chord's one-hot index.
        """
        if reduction is None:
            reduction = NO_REDUCTION

        if relative:
            root = absolute_to_relative(
                self.root,
                self.key_tonic,
                self.pitch_type,
                False,
                pad=pad,
            )
        else:
            root = self.root

        return get_chord_one_hot_index(
            self.chord_type,
            root,
            self.pitch_type,
            inversion=self.inversion,
            use_inversion=use_inversion,
            relative=relative,
            pad=pad,
            reduction=reduction,
        )

    def get_chord_vector_length(
        self,
        one_hot: bool = True,
        relative: bool = True,
        use_inversions: bool = True,
        pad: bool = False,
        reduction: Dict[ChordType, ChordType] = None,
    ) -> int:
        """
        Get the length of this Chord's vector.

        Parameters
        ----------
        one_hot : bool
            True to return the one-hot chord change vector length.
        relative : bool
            True to return the length of a relative chord vector. False for absolute.
        use_inversions : bool
            True to return the length of the chord vector including inversions. False otherwise.
            Only relevant if one_hot == True.
        pad : bool
            If True, pitch_type is TPC, and relative is True, pad the possible pitches with
            extra spaces.
        reduction : Dict[ChordType, ChordType]
            A reduction mapping each chord type to a different chord type. This will affect
            only the one-hot chord vector lengths.

        Returns
        -------
        length : int
            The length of this Chord's vector.
        """
        return get_chord_vector_length(
            self.pitch_type,
            one_hot=one_hot,
            relative=relative,
            use_inversions=use_inversions,
            pad=pad,
            reduction=reduction,
        )

    def to_vec(
        self,
        relative_to: "Key" = None,
        pad: bool = False,
        reduction: Dict[ChordType, ChordType] = NO_REDUCTION,
    ) -> np.ndarray:
        """
        Get the vectorized representation of this chord.

        Parameters
        ----------
        relative_to : Key
            The key to make this chord vector relative to, if not its key.
        pad : bool
            If True, pitch_type is TPC, and relative is True, pad the possible pitches with
            extra spaces.
        reduction : Dict[ChordType, ChordType]
            A reduction mapping for this chord's chord_type.

        Returns
        -------
        chord : np.ndarray
            The vector of this Chord.
        """
        key_tonic = self.key_tonic if relative_to is None else relative_to.relative_tonic
        key_mode = self.key_mode if relative_to is None else relative_to.relative_mode

        if self.pitch_type == PitchType.MIDI:
            num_pitches = NUM_PITCHES[self.pitch_type]
        else:
            num_pitches = MAX_RELATIVE_TPC - MIN_RELATIVE_TPC
            if pad:
                num_pitches += 2 * RELATIVE_TPC_EXTRA

        vectors = []

        # Relative root as one-hot
        pitch = np.zeros(num_pitches, dtype=np.float16)
        index = absolute_to_relative(
            self.root,
            key_tonic,
            self.pitch_type,
            False,
            pad=pad,
        )
        pitch[index] = 1
        vectors.append(pitch)

        # Chord type
        chord_type = np.zeros(len(ChordType), dtype=np.float16)
        chord_type[reduction[self.chord_type].value] = 1
        vectors.append(chord_type)

        # Relative bass as one-hot
        bass_note = np.zeros(num_pitches, dtype=np.float16)
        index = absolute_to_relative(
            self.bass,
            key_tonic,
            self.pitch_type,
            False,
            pad=pad,
        )
        bass_note[index] = 1
        vectors.append(bass_note)

        # Inversion as one-hot
        inversion = np.zeros(4, dtype=np.float16)
        inversion[self.inversion] = 1
        vectors.append(inversion)

        # Onset metrical level as one-hot
        onset_level = np.zeros(4, dtype=np.float16)
        onset_level[self.onset_level] = 1
        vectors.append(onset_level)

        # Offset metrical level as one-hot
        offset_level = np.zeros(4, dtype=np.float16)
        offset_level[self.offset_level] = 1
        vectors.append(offset_level)

        # Duration as float
        vectors.append([float(self.duration)])

        # Binary -- is the current key major
        vectors.append([1 if key_mode == KeyMode.MAJOR else 0])

        return np.concatenate(vectors).astype(dtype=np.float16)

    def is_repeated(self, other: "Chord", use_inversion: bool = True) -> bool:
        """
        Detect if a given chord can be regarded as a repeat of this one in terms of root and
        chord_type, plus optionally inversion.

        Parameters
        ----------
        other : Chord
            The other chord to check for repeat.
        use_inversion : bool
            True to take inversions into account. False otherwise.

        Returns
        -------
        is_repeated : bool
            True if the given chord is a repeat of this one. False otherwise.
        """
        if not isinstance(other, Chord):
            return False

        attr_names = ["pitch_type", "root", "chord_type"]
        if use_inversion:
            attr_names.append("inversion")

        for attr_name in attr_names:
            if getattr(self, attr_name) != getattr(other, attr_name):
                return False
        return True

    def merge_with(self, next_chord: "Chord"):
        """
        Merge this chord with the next one, in terms of metrical information. Specifically,
        move this chord's offset and offset_level to the next_chord's, and set this chord's
        duration to their combined duration sum.

        Parameters
        ----------
        next_chord : Chord
            The chord to merge with this one.
        """
        self.offset = next_chord.offset
        self.offset_level = next_chord.offset_level
        self.duration += next_chord.duration

    def to_dict(self) -> Dict:
        """
        Convert this Chord to a dictionary, which can be called into the Chord constructor as
        Chord(**dict) to recreate a copy of this Key.

        Returns
        -------
        chord_dict : Dict
            A dictionary representation of all of the fields of this Chord.
        """
        return {field: getattr(self, field) for field in self.params}

    def __eq__(self, other: "Chord") -> bool:
        if not isinstance(other, Chord):
            return False
        for field in self.params:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def __repr__(self) -> str:
        params = ", ".join([f"{field}={getattr(self, field)}" for field in self.params])
        return f"Chord({params})"

    def __str__(self) -> str:
        if self.inversion == 0:
            inversion_str = "root position"
        elif self.inversion == 1:
            inversion_str = "1st inversion"
        elif self.inversion == 2:
            inversion_str = "2nd inversion"
        elif self.inversion == 3:
            inversion_str = "3rd inversion"
        else:
            inversion_str = f"{self.inversion}th inversion"

        return (
            f"{get_pitch_string(self.root, self.pitch_type)}:"
            f"{get_chord_string(self.chord_type)} {inversion_str} "
            f"BASS={get_pitch_string(self.bass, self.pitch_type)}: "
            f"{self.onset}--{self.offset}"
        )

    @staticmethod
    def from_series(
        chord_row: pd.Series,
        measures_df: pd.DataFrame,
        pitch_type: PitchType,
        reduction: Dict[ChordType, ChordType] = NO_REDUCTION,
        use_inversion: bool = True,
        use_relative: bool = True,
        key: "Key" = None,
        levels_cache: DefaultDict[str, Dict[Fraction, int]] = None,
    ) -> "Chord":
        """
        Create a Chord object of the given pitch_type from the given pd.Series.

        Parameters
        ----------
        chord_row : pd.Series
            The chord row from which to make our chord object. It must contain at least the rows:
                'numeral' (str): The numeral of the chord label. If this is null or '@none',
                                 None is returned.
                'root' (int): The interval of the root note above the local key tonic, in TPC.
                'bass_note' (int): The interval of the bass note above the local key tonic, in TPC.
                'chord_type' (str): The string representation of the chord type.
                'figbass' (str): The figured bass of the chord inversion.
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
                'mc' (int): The chord's onset measure.
                'onset' (Fraction): The chord's onset beat, in whole notes.
                'mc_next' (int): The chord's offset measure.
                'onset_next' (Fraction): The chord's offset beat, in whole notes.
                'duration' (Fraction): The chord's duration, in whole notes.
        measures_df : pd.DataFrame
            A pd.DataFrame of the measures in the piece of the chord. It is used to get metrical
            levels of the chord's onset and offset. Must have at least the columns:
                'mc' (int): The measure number, to match with the chord's onset and offset.
                'timesig' (str): The time signature of the measure.
        pitch_type : PitchType
            The pitch type to use for the Chord.
        reduction : Dict[ChordType, ChordType]
            A reduction mapping each possible ChordType to a reduced ChordType.
        use_inversion : bool
            True to store the chord's inversion. False to use inversion = 0 for all chords.
        use_relative : bool
            True to store the chord's key tonic and mode while treated relative roots as new keys.
            False to only use the annotated local keys.
        key : Key
            The key during this Chord. If not given, it will be calculated from chord_row.
        levels_cache : DefaultDict[str, Dict[Fraction, int]]
            If given, a dictionary-based cache mapping time signatures to a 2nd dictionary mapping
            beat positions to metrical levels. The outer-most dictionary should be a default-dict
            returning by default an empty dict.

        Returns
        -------
        chord : Chord, or None
            The created Note object. If an error occurs, None is returned and the error is logged.
        """
        try:
            if chord_row["numeral"] == "@none" or pd.isnull(chord_row["numeral"]):
                # Handle "No Chord" symbol
                return None

            # Root and bass note are relative to local key (not applied dominant)
            if key is None:
                key = Key.from_series(chord_row, pitch_type)

            # Root and bass note of chord, as intervals above the local key tonic
            root_interval = chord_row["root"]
            # bass_interval = chord_row["bass_note"]
            if pitch_type == PitchType.MIDI:
                root_interval = tpc_interval_to_midi_interval(root_interval)
                # bass_interval = hu.tpc_interval_to_midi_interval(bass_interval)

            # Absolute root and bass
            root = transpose_pitch(key.local_tonic, root_interval, pitch_type=pitch_type)
            # A bug in the corpus data makes this incorrect for half-diminished chords
            # bass = hu.transpose_pitch(key.local_tonic, bass_interval, pitch_type=pitch_type)

            # Additional chord info
            chord_type = reduction[get_chord_type_from_string(chord_row["chord_type"])]
            inversion = get_chord_inversion(chord_row["figbass"]) if use_inversion else 0
            bass = get_bass_note(chord_type, root, inversion, pitch_type)
            assert 0 <= bass < NUM_PITCHES[pitch_type]

            # Rhythmic info
            positions = [None, None]
            levels = [None, None]
            for i, (mc, beat) in enumerate(
                zip(
                    [chord_row.mc, chord_row.mc_next],
                    [chord_row.onset, chord_row.onset_next],
                )
            ):
                measure = measures_df.loc[measures_df["mc"] == mc].squeeze()

                if levels_cache is None:
                    level = get_metrical_level(beat, measure)
                else:
                    time_sig_cache = levels_cache[measure["timesig"]]
                    if beat in time_sig_cache:
                        level = time_sig_cache[beat]
                    else:
                        level = get_metrical_level(beat, measure)
                        time_sig_cache[beat] = level

                positions[i] = (mc, beat)
                levels[i] = level

            onset, offset = positions
            onset_level, offset_level = levels

            duration = chord_row.duration

            return Chord(
                root,
                bass,
                key.relative_tonic if use_relative else key.local_tonic,
                key.relative_mode if use_relative else key.local_mode,
                chord_type,
                inversion,
                onset,
                onset_level,
                offset,
                offset_level,
                duration,
                pitch_type,
            )

        except Exception as exception:
            logging.error("Error parsing chord from row %s", chord_row)
            logging.exception(exception)
            return None

    @staticmethod
    def from_labels_csv_row(
        chord_row: pd.Series,
        measures_df: pd.DataFrame,
        pitch_type: PitchType,
        reduction: Dict[ChordType, ChordType] = NO_REDUCTION,
        use_inversion: bool = True,
        use_relative: bool = True,
        key: "Key" = None,
        levels_cache: DefaultDict[str, Dict[Fraction, int]] = None,
    ) -> "Chord":
        """
        Create a Chord object of the given pitch_type from the given pd.Series.

        Parameters
        ----------
        chord_row : pd.Series
            The chord row from which to make our chord object, from a labels csv file.
            It must contain at least the rows:
                'on' (Fraction): The onset of the chord label, measured in whole notes since
                                 the beginning of the piece.
                'off' (Fraction): The offset of the chord label, measured in whole notes since
                                  the beginning of the piece.
                'key' (str): The local key at the time of the given label. Flats and sharps
                             are represented with - and +, and major/minor is represented
                             by upper/lower-case.
                'degree' (str): The degree of the chord's root, relative to the local key,
                                and including applied chords with / notation. Flats and sharps
                                are represented with - and +.
                'type' (str): A string representing the chord type.
                'inv' (int): The inversion of the chord.
        measures_df : pd.DataFrame
            A pd.DataFrame of the measures in the piece of the chord. It is used to get metrical
            levels of the chord's onset and offset. Must have at least the columns:
                'mc' (int): The measure number, to match with the chord's onset and offset.
                'timesig' (str): The time signature of the measure.
        pitch_type : PitchType
            The pitch type to use for the Chord.
        reduction : Dict[ChordType, ChordType]
            A reduction mapping each possible ChordType to a reduced ChordType.
        use_inversion : bool
            True to store the chord's inversion. False to use inversion = 0 for all chords.
        use_relative : bool
            True to store the chord's key tonic and mode while treated relative roots as new keys.
            False to only use the annotated local keys.
        key : Key
            The key during this Chord. If not given, it will be calculated from chord_row.
        levels_cache : DefaultDict[str, Dict[Fraction, int]]
            If given, a dictionary-based cache mapping time signatures to a 2nd dictionary mapping
            beat positions to metrical levels. The outer-most dictionary should be a default-dict
            returning by default an empty dict.

        Returns
        -------
        chord : Chord, or None
            The created Note object. If an error occurs, None is returned and the error is logged.
        """
        if key is None:
            key = Key.from_labels_csv_row(chord_row, pitch_type)

        # Categorical info
        chord_type = {
            "D7": ChordType.MAJ_MIN7,
            "M": ChordType.MAJOR,
            "d": ChordType.DIMINISHED,
            "d7": ChordType.DIM7,
            "m": ChordType.MINOR,
            "m7": ChordType.MIN_MIN7,
            "Gr+6": ChordType.DIM7,
            "h7": ChordType.HALF_DIM7,
        }[chord_row["type"]]

        # Harmonic info (root pitch)
        degree = chord_row["degree"].replace("-", "b")
        degree = degree.replace("+", "#")

        if "/" in degree:
            degree = degree.split("/")[-1]

        interval = get_interval_from_scale_degree(
            degree, False, key.relative_mode, pitch_type=pitch_type
        )
        root = transpose_pitch(key.relative_tonic, interval, pitch_type=pitch_type)

        # Metrical info
        onset_measure = measures_df.loc[measures_df["start"] <= chord_row["on"]].iloc[-1]
        offset_measure = measures_df.loc[measures_df["start"] >= chord_row["off"]].iloc[0]

        onset_beat = onset_measure["offset"] + chord_row["on"] - onset_measure["start"]
        offset_beat = offset_measure["offset"] + chord_row["off"] - offset_measure["start"]

        levels = [None, None]

        for i, (beat, measure) in enumerate(
            zip(
                [onset_beat, offset_beat],
                [onset_measure, offset_measure],
            )
        ):
            if levels_cache is None:
                level = get_metrical_level(beat, measure)

            else:
                time_sig_cache = levels_cache[measure["timesig"]]
                if beat in time_sig_cache:
                    level = time_sig_cache[beat]
                else:
                    level = get_metrical_level(beat, measure)
                    time_sig_cache[beat] = level

            levels[i] = level

        onset_level, offset_level = levels

        return Chord(
            root,
            get_bass_note(chord_type, root, chord_row["inv"], pitch_type),
            key.relative_tonic if use_relative else key.local_tonic,
            key.relative_mode if use_relative else key.local_mode,
            reduction[chord_type],
            chord_row["inv"],
            (onset_measure["mc"], onset_beat),
            onset_level,
            (offset_measure["mc"], offset_beat),
            offset_level,
            chord_row["off"] - chord_row["on"],
            pitch_type,
        )


def get_chord_vector_length(
    pitch_type: PitchType,
    one_hot: bool = True,
    relative: bool = True,
    use_inversions: bool = True,
    pad: bool = False,
    reduction: Dict[ChordType, ChordType] = None,
) -> int:
    """
    Get the length of a chord vector.

    Parameters
    ----------
    pitch_type : PitchType
        The pitch type for the given vector.
    one_hot : bool
        True to return the one-hot chord change vector length.
    relative : bool
        True to return the length of a relative chord vector. False for absolute.
    use_inversions : bool
        True to return the length of the chord vector including inversions. False otherwise.
        Only relevant if one_hot == True.
    pad : bool
        If True, pitch_type is TPC, and relative is True, pad the possible pitches with
        extra spaces.
    reduction : Dict[ChordType, ChordType]
        A reduction mapping each chord type to a different chord type. This will affect
        only the one-hot chord vector lengths.

    Returns
    -------
    length : int
        The length of a single chord vector.
    """
    if reduction is None:
        reduction = NO_REDUCTION

    if relative and pitch_type == PitchType.TPC:
        num_pitches = MAX_RELATIVE_TPC - MIN_RELATIVE_TPC
        if pad:
            num_pitches += RELATIVE_TPC_EXTRA * 2
    else:
        num_pitches = NUM_PITCHES[pitch_type]

    if one_hot:
        if use_inversions:
            return np.sum(
                np.array(
                    [
                        get_chord_inversion_count(chord_type)
                        for chord_type in set(reduction.values())
                    ]
                )
                * num_pitches
            )
        return num_pitches * len(set(reduction.values()))

    return (
        num_pitches  # Root
        + num_pitches  # Bass
        + len(ChordType)  # chord type
        + 14  # 4 each for inversion, onset level, offset level; 1 for duration, 1 for is_major
    )
