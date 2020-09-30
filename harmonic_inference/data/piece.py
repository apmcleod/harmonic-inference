"""A class storing a musical piece from score, midi, or audio format."""
from typing import List, Union, Tuple, Dict
from fractions import Fraction
import logging
import inspect

from tqdm import tqdm
import pandas as pd
import numpy as np

from harmonic_inference.data.data_types import KeyMode, PitchType, ChordType, PieceType
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES
import harmonic_inference.utils.harmonic_utils as hu
import harmonic_inference.utils.rhythmic_utils as ru
import harmonic_inference.utils.harmonic_constants as hc


class Note():
    """
    A representation of a musical Note, with pitch, onset, duration, and offset.
    """
    def __init__(
        self,
        pitch_class: int,
        octave: int,
        onset: Union[float, Tuple[int, Fraction]],
        onset_level: int,
        duration: Union[float, Fraction],
        offset: Union[float, Tuple[int, Fraction]],
        offset_level: int,
        pitch_type: PitchType,
    ):
        """
        Create a new musical Note.

        Parameters
        ----------
        pitch_class : int
            An integer representing pitch class either as semitones above C (if pitch_type is MIDI;
            with B#, C = 0), or as tonal pitch class (if pitch_type is TPC; with C = 0, G = 1, etc.
            around the circle of fifths).
        octave : int
            An integer representing the octave in which the note lies.
        onset : Union[float, Tuple[int, Fraction]]
            The onset position of this Note. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        onset_level : int
            The metrical level on which the onset lies. 0=none, 1=subbeat, 2=beat, 3=downbeat.
        duration : Union[float, Fraction]
            The duration of this Note. Either a float (representing time in seconds), ar a Fraction
            (representing whole notes).
        offset : Union[float, Tuple[int, Fraction]]
            The onset position of this Note. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        offset_level : int
            The metrical level on which the offset lies. 0=none, 1=subbeat, 2=beat, 3=downbeat.
        pitch_type : PitchType
            The PitchType in which this note's pitch_class is stored. If this is TPC, the
            pitch_class can be later converted into MIDI, but not vice versa.
        """
        self.pitch_class = pitch_class
        self.octave = octave
        self.onset = onset
        self.onset_level = onset_level
        self.duration = duration
        self.offset = offset
        self.offset_level = offset_level
        self.pitch_type = pitch_type

        self.params = inspect.getfullargspec(Note.__init__).args[1:]

    @staticmethod
    def get_note_vector_length(pitch_type: PitchType) -> int:
        """
        Get the length of a note vector.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type of the note.

        Returns
        -------
        length : int
            The length of a single note vector of the given pitch type.
        """
        return (
            hc.NUM_PITCHES[PitchType.TPC] +  # Pitch class
            127 // hc.NUM_PITCHES[PitchType.MIDI] +  # octave
            12  # 4 onset level, 4 offset level, onset, offset, duration, is_lowest
        )

    def to_vec(
        self,
        chord_onset: Union[float, Tuple[int, Fraction]] = None,
        chord_offset: Union[float, Tuple[int, Fraction]] = None,
        chord_duration: Union[float, Fraction] = None,
        measures_df: pd.DataFrame = None,
        min_pitch: Tuple[int, int] = None,
        note_onset: Fraction = None,
    ) -> np.array:
        """
        Get the vectorized representation of this note given a chord.

        Parameters
        ----------
        chord_onset : Union[float, Tuple[int, Fraction]]
            The onset position of the chord the vector should be relative to (since we might want
            relative positions or durations). None to not include chord-relative information in the
            vector.

        chord_offset : Union[float, Tuple[int, Fraction]]
            The offset position of the chord the vector should be relative to (since we might want
            relative positions or durations). None to not include chord-relative information in the
            vector.

        chord_duration : Union[float, Fraction]
            The duration of the chord the vector should be relative to (since we might want
            relative positions or durations). None to not include chord-relative information in the
            vector.

        measures_df : pd.DataFrame
            The measures DataFrame for this piece, to be used for getting metrical range
            information. None to not include chord-relative metrical information in the
            vector.

        min_pitch : Tuple[int, int]
            The minimum pitch of any note in this set of notes, expressed as a (octave, pitch)
            tuple. None to not include the binary is_lowest vector entry.

        note_onset : Fraction
            The duration from the chord onset to the note's onset. If given, this speeds up
            computation by eliminating a call to rhythmic_utils.get_range(...).

        Returns
        -------
        vector : np.array
            The vector of this Note.
        """
        vectors = []

        # Pitch as one-hot
        pitch = np.zeros(hc.NUM_PITCHES[self.pitch_type])
        pitch[self.pitch_class] = 1
        vectors.append(pitch)

        # Octave as one-hot
        octave = np.zeros(127 // hc.NUM_PITCHES[PitchType.MIDI])
        octave[self.octave] = 1
        vectors.append(octave)

        # Onset metrical level as one-hot
        onset_level = np.zeros(4)
        onset_level[self.onset_level] = 1
        vectors.append(onset_level)

        # Offset metrical level as one-hot
        offset_level = np.zeros(4)
        offset_level[self.offset_level] = 1
        vectors.append(offset_level)

        # onset, offset, duration as floats, as proportion of chord's range
        if (
            chord_onset is not None and
            chord_offset is not None and
            chord_duration is not None and
            measures_df is not None
        ):
            if note_onset is None:
                onset, offset, duration = ru.get_rhythmic_info_as_proportion_of_range(
                    pd.Series({
                        'mc': self.onset[0],
                        'onset': self.onset[1],
                        'duration': self.duration,
                    }),
                    chord_onset,
                    chord_offset,
                    measures_df,
                    range_len=chord_duration,
                )
            else:
                onset = note_onset / chord_duration
                duration = self.duration / chord_duration
                offset = onset + duration
            metrical = np.array([onset, offset, duration], dtype=float)
            vectors.append(metrical)
        else:
            vectors.append(np.zeros(3, dtype=float))

        # Binary -- is this the lowest note in this set of notes
        min_pitch = np.array(
            [1 if min_pitch is not None and (self.octave, self.pitch_class) == min_pitch else 0]
        )
        vectors.append(min_pitch)

        return np.concatenate(vectors)

    def __eq__(self, other: 'Note') -> bool:
        if not isinstance(other, Note):
            return False
        for field in self.params:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def to_dict(self) -> Dict:
        return {field: getattr(self, field) for field in self.params}

    def __repr__(self) -> str:
        params = ", ".join([f"{field}={getattr(self, field)}" for field in self.params])
        return f"Note({params})"

    def __str__(self) -> str:
        return (
            f'{hu.get_pitch_string(self.pitch_class, self.pitch_type)}{self.octave}: '
            f'{self.onset}--{self.offset}'
        )

    @staticmethod
    def from_series(
        note_row: pd.Series,
        measures_df: pd.DataFrame,
        pitch_type: PitchType
    ) -> 'Note':
        """
        Create a new Note object from a pd.Series, and return it.

        Parameters
        ----------
        note_row : pd.Series
            A pd.Series of a note. Must have at least the fields:
                'midi' (int): MIDI pitch, from 0 to 127.
                'tpc' (int): The note's TPC pitch, where C = 0. Required if pitch_type is TPC.
                'mc' (int): The note's onset measure.
                'onset' (Fraction): The note's onset beat, in whole notes.
                'offset_mc' (int): The note's offset measure.
                'offset_beat' (Fraction): The note's offset beat, in whole notes.
                'duration' (Fraction): The note's duration, in whole notes.
        measures_df : pd.DataFrame
            A pd.DataFrame of the measures in the piece of the note. It is used to get metrical
            levels of the note's onset and offset. Must have at least the columns:
                'mc' (int): The measure number, to match with the note's onset and offset.
                'timesig' (str): The time signature of the measure.
        pitch_type : PitchType
            The pitch type to use for the Note.

        Returns
        -------
        note : Note, or None
            The created Note object. If an error occurs, None is returned and the error is logged.
        """
        try:
            if pitch_type == PitchType.TPC:
                pitch = note_row.tpc + hc.TPC_C
                if pitch < 0 or pitch >= hc.NUM_PITCHES[PitchType.TPC]:
                    raise ValueError(f"TPC pitch {pitch} is outside of valid range.")
            elif pitch_type == PitchType.MIDI:
                pitch = note_row.midi % hc.NUM_PITCHES[PitchType.MIDI]
            else:
                raise ValueError(f"Invalid pitch type: {pitch_type}")
            octave = note_row.midi // hc.NUM_PITCHES[PitchType.MIDI]

            onset = (note_row["mc"], note_row["onset"])
            onset_level = ru.get_metrical_level(
                note_row["onset"],
                measures_df.loc[measures_df["mc"] == note_row["mc"]].squeeze(),
            )

            offset = (note_row["offset_mc"], note_row["offset_beat"])
            offset_level = ru.get_metrical_level(
                note_row["offset_beat"],
                measures_df.loc[measures_df["mc"] == note_row["offset_mc"]].squeeze(),
            )

            return Note(pitch, octave, onset, onset_level, note_row.duration, offset,
                        offset_level, pitch_type)

        except Exception as e:
            logging.error(f"Error parsing note from row {note_row}")
            logging.exception(e)
            return None


class Chord():
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
        pitch_type: PitchType
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

    def get_one_hot_index(self, relative: bool = False, use_inversion: bool = True) -> int:
        """
        Get the one-hot index of this chord.

        Parameters
        ----------
        relative : bool
            True to get the relative one-hot index. False for the absolute one-hot index.
        use_inversion : bool
            True to use inversions. False otherwise.

        Returns
        -------
        index : int
            This Chord's one-hot index.
        """
        if relative:
            root = hu.absolute_to_relative(self.root, self.key_tonic, self.pitch_type, False)
        else:
            root = self.root

        return hu.get_chord_one_hot_index(
            self.chord_type,
            root,
            self.pitch_type,
            inversion=self.inversion,
            use_inversion=use_inversion,
            relative=relative,
        )

    @staticmethod
    def get_chord_vector_length(
        pitch_type: PitchType,
        one_hot: bool = True,
        relative: bool = True,
        use_inversions: bool = True,
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

        Returns
        -------
        length : int
            The length of a single chord vector.
        """
        if relative and pitch_type == PitchType.TPC:
            num_pitches = hc.MAX_RELATIVE_TPC - hc.MIN_RELATIVE_TPC + hc.RELATIVE_TPC_EXTRA
        else:
            num_pitches = hc.NUM_PITCHES[pitch_type]

        if one_hot:
            if use_inversions:
                return np.sum(
                    num_pitches * np.array(
                        [
                            hu.get_chord_inversion_count(chord_type)
                            for chord_type in ChordType
                        ]
                    )
                )
            return num_pitches * len(ChordType)

        return (
            num_pitches +  # Root
            num_pitches +  # Bass
            len(ChordType) +  # chord type
            13  # 4 each for inversion, onset level, offset level; 1 for is_major
        )

    def to_vec(self, relative_to: 'Key' = None) -> np.ndarray:
        """
        Get the vectorized representation of this chord.

        Parameters
        ----------
        relative_to : Key
            The key to make this chord vector relative to, if not its key.

        Returns
        -------
        chord : np.ndarray
            The vector of this Chord.
        """
        key_tonic = self.key_tonic if relative_to is None else relative_to.relative_tonic
        key_mode = self.key_mode if relative_to is None else relative_to.relative_mode

        num_pitches = (
            hc.NUM_PITCHES[self.pitch_type]
            if self.pitch_type == PitchType.MIDI else
            hc.MAX_RELATIVE_TPC - hc.MIN_RELATIVE_TPC + 2 * hc.RELATIVE_TPC_EXTRA
        )

        vectors = []

        # Relative root as one-hot
        pitch = np.zeros(num_pitches)
        pitch[
            hu.absolute_to_relative(
                self.root,
                key_tonic,
                self.pitch_type,
                False,
                check=False,
            ) + hc.RELATIVE_TPC_EXTRA
        ] = 1
        vectors.append(pitch)

        # Chord type
        chord_type = np.zeros(len(ChordType))
        chord_type[self.chord_type.value] = 1
        vectors.append(chord_type)

        # Relative bass as one-hot
        bass_note = np.zeros(num_pitches)
        bass_note[
            hu.absolute_to_relative(
                self.bass,
                key_tonic,
                self.pitch_type,
                False,
                check=False,
            ) + hc.RELATIVE_TPC_EXTRA
        ] = 1
        vectors.append(bass_note)

        # Inversion as one-hot
        inversion = np.zeros(4)
        inversion[self.inversion] = 1
        vectors.append(inversion)

        # Onset metrical level as one-hot
        onset_level = np.zeros(4)
        onset_level[self.onset_level] = 1
        vectors.append(onset_level)

        # Offset metrical level as one-hot
        offset_level = np.zeros(4)
        offset_level[self.offset_level] = 1
        vectors.append(offset_level)

        # Binary -- is the current key major
        is_major = np.array(
            [1 if key_mode == KeyMode.MAJOR else 0]
        )
        vectors.append(is_major)

        return np.concatenate(vectors)

    def is_repeated(self, other: 'Chord', use_inversion: bool = True) -> bool:
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

        attr_names = ['pitch_type', 'root', 'chord_type']
        if use_inversion:
            attr_names.append('inversion')

        for attr_name in attr_names:
            if getattr(self, attr_name) != getattr(other, attr_name):
                return False
        return True

    def merge_with(self, next_chord: 'Chord'):
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

    def __eq__(self, other: 'Chord') -> bool:
        if not isinstance(other, Chord):
            return False
        for field in self.params:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def to_dict(self) -> Dict:
        return {field: getattr(self, field) for field in self.params}

    def __repr__(self) -> str:
        params = ", ".join([f"{field}={getattr(self, field)}" for field in self.params])
        return f"Chord({params})"

    def __str__(self) -> str:
        if self.inversion == 0:
            inversion_str = 'root position'
        elif self.inversion == 1:
            inversion_str = '1st inversion'
        elif self.inversion == 2:
            inversion_str = '2nd inversion'
        elif self.inversion == 3:
            inversion_str = '3rd inversion'
        else:
            inversion_str = f'{self.inversion}th inversion'

        return (f'{hu.get_pitch_string(self.root, self.pitch_type)}:'
                f'{hu.get_chord_string(self.chord_type)} {inversion_str} '
                f'BASS={hu.get_pitch_string(self.bass, self.pitch_type)}: '
                f'{self.onset}--{self.offset}')

    @staticmethod
    def from_series(
        chord_row: pd.Series,
        measures_df: pd.DataFrame,
        pitch_type: PitchType,
        key=None,
    ) -> 'Chord':
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
        key : Key
            The key during this Chord. If not given, it will be calculated from chord_row.

        Returns
        -------
        chord : Chord, or None
            The created Note object. If an error occurs, None is returned and the error is logged.
        """
        try:
            if chord_row['numeral'] == '@none' or pd.isnull(chord_row['numeral']):
                # Handle "No Chord" symbol
                return None

            # Root and bass note are relative to local key (not applied dominant)
            if key is None:
                key = Key.from_series(chord_row, pitch_type)

            # Root and bass note of chord, as intervals above the local key tonic
            root_interval = chord_row["root"]
            # bass_interval = chord_row["bass_note"]
            if pitch_type == PitchType.MIDI:
                root_interval = hu.tpc_interval_to_midi_interval(root_interval)
                # bass_interval = hu.tpc_interval_to_midi_interval(bass_interval)

            # Absolute root and bass
            root = hu.transpose_pitch(key.local_tonic, root_interval, pitch_type=pitch_type)
            # bass = hu.transpose_pitch(key.local_tonic, bass_interval, pitch_type=pitch_type)

            # Additional chord info
            chord_type = hu.get_chord_type_from_string(chord_row['chord_type'])
            inversion = hu.get_chord_inversion(chord_row['figbass'])
            bass = hu.get_bass_note(chord_type, root, inversion, pitch_type)
            assert 0 <= bass < hc.NUM_PITCHES[pitch_type]

            # Rhythmic info
            onset = (chord_row.mc, chord_row.onset)
            onset_level = ru.get_metrical_level(
                chord_row["onset"],
                measures_df.loc[measures_df["mc"] == chord_row["mc"]].squeeze(),
            )

            offset = (chord_row.mc_next, chord_row.onset_next)
            offset_level = ru.get_metrical_level(
                chord_row["onset_next"],
                measures_df.loc[measures_df["mc"] == chord_row["mc_next"]].squeeze(),
            )

            duration = chord_row.duration

            return Chord(root, bass, key.relative_tonic, key.relative_mode, chord_type, inversion,
                         onset, onset_level, offset, offset_level, duration, pitch_type)

        except Exception as e:
            logging.error(f"Error parsing chord from row {chord_row}")
            logging.exception(e)
            return None


class Key():
    """
    A musical key, with tonic and mode.
    """
    def __init__(
        self,
        relative_tonic: int,
        local_tonic: int,
        relative_mode: KeyMode,
        local_mode: KeyMode,
        tonic_type: PitchType
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
        relative_mode : KeyMode
            The mode of this key, including applied roots.
        local_mode : KeyMode
            The mode of this key, without taking applied roots into account.
        tonic_type : PitchType
            The PitchType in which this key's tonic is stored. If this is TPC, the
            tonic can be later converted into MIDI type, but not vice versa.
        """
        self.relative_tonic = relative_tonic
        self.local_tonic = local_tonic
        self.relative_mode = relative_mode
        self.local_mode = local_mode
        self.tonic_type = tonic_type

        self.params = inspect.getfullargspec(Key.__init__).args[1:]

    @staticmethod
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
            num_pitches = hc.MAX_KEY_CHANGE_INTERVAL_TPC - hc.MIN_KEY_CHANGE_INTERVAL_TPC
        elif pitch_type == PitchType.MIDI:
            num_pitches = 12
        else:
            raise ValueError(f"Invalid pitch_type: {pitch_type}")

        if one_hot:
            return num_pitches * len(KeyMode)
        return num_pitches + len(KeyMode)

    def get_key_change_vector(self, next_key: 'Key') -> np.array:
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
        change_vector = np.zeros(Key.get_key_change_vector_length(self.tonic_type, one_hot=False))

        # Relative tonic
        change_vector[
            hu.absolute_to_relative(
                next_key.relative_tonic,
                self.relative_tonic,
                self.tonic_type,
                True,
            )
        ] = 1

        # Absolute mode of next key
        change_vector[-2 + next_key.relative_mode.value] = 1

        return change_vector

    def get_key_change_one_hot_index(self, next_key: 'Key') -> int:
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
        interval = hu.absolute_to_relative(
            next_key.relative_tonic,
            self.relative_tonic,
            self.tonic_type,
            True,
        )

        if self.tonic_type == PitchType.MIDI:
            num_pitches = NUM_PITCHES[PitchType.MIDI]
        else:
            num_pitches = hc.MAX_KEY_CHANGE_INTERVAL_TPC - hc.MIN_KEY_CHANGE_INTERVAL_TPC
        return next_key.relative_mode.value * num_pitches + interval

    def is_repeated(self, other: 'Key', use_relative: bool = True) -> bool:
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
        if not isinstance(other, Key):
            return False

        attr_names = ['tonic_type']
        if use_relative:
            attr_names.extend(['relative_tonic', 'relative_mode'])
        else:
            attr_names.extend(['local_tonic', 'local_mode'])

        for attr_name in attr_names:
            if getattr(self, attr_name) != getattr(other, attr_name):
                return False
        return True

    def __eq__(self, other: 'Key') -> bool:
        if not isinstance(other, Key):
            return False
        for field in self.params:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def to_dict(self) -> Dict:
        return {field: getattr(self, field) for field in self.params}

    def __repr__(self) -> str:
        params = ", ".join([f"{field}={getattr(self, field)}" for field in self.params])
        return f"Key({params})"

    def __str__(self) -> str:
        return f'{hu.get_pitch_string(self.relative_tonic, self.tonic_type)} {self.relative_mode}'

    @staticmethod
    def from_series(chord_row: pd.Series, tonic_type: PitchType) -> 'Key':
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
        pitch_type : PitchType
            The pitch type to use for the Key's tonic.

        Returns
        -------
        key : Key, or None
            The created Key object. If an error occurs, None is returned and the error is logged.
        """
        try:
            # Global key, absolute
            global_tonic = hu.get_pitch_from_string(chord_row['globalkey'], pitch_type=tonic_type)
            global_mode = KeyMode.MINOR if chord_row['globalkey_is_minor'] else KeyMode.MAJOR

            # Local key is listed relative to global. We want it absolute.
            local_mode = KeyMode.MINOR if chord_row['localkey_is_minor'] else KeyMode.MAJOR
            local_transposition = hu.get_interval_from_numeral(
                chord_row['localkey'], global_mode, pitch_type=tonic_type
            )
            local_tonic = hu.transpose_pitch(
                global_tonic, local_transposition, pitch_type=tonic_type
            )

            # Treat applied dominants (and other slash chords) as new keys
            relative_full = chord_row['relativeroot']
            relative_tonic = local_tonic
            relative_mode = local_mode
            if not pd.isna(relative_full):
                # Handle doubly-relative chords iteratively
                for relative in reversed(relative_full.split('/')):
                    # Relativeroot is listed relative to local key. We want it absolute.
                    relative_transposition = hu.get_interval_from_numeral(
                        relative, relative_mode, pitch_type=tonic_type
                    )
                    relative_mode = KeyMode.MINOR if relative[-1].islower() else KeyMode.MAJOR
                    relative_tonic = hu.transpose_pitch(relative_tonic, relative_transposition,
                                                        pitch_type=tonic_type)

            return Key(relative_tonic, local_tonic, relative_mode, local_mode, tonic_type)

        except Exception as e:
            logging.error(f"Error parsing key from row {chord_row}")
            logging.exception(e)
            return None


def get_reduction_mask(inputs: List[Union[Chord, Key]], kwargs: Dict = {}) -> List[bool]:
    """
    Return a boolean mask that will remove repeated inputs when applied to the given inputs list
    as inputs[mask].

    Parameters
    ----------
    inputs : List[Union[Chord, Key]]
        A List of either Chord or Key objects.
    kwargs : Dict
        A Dictionary of kwargs to pass along to each given input's is_repeated() function.

    Returns
    -------
    mask : List[bool]
        A boolean mask that will remove repeated inputs when applied to the given inputs list
        as inputs = inputs[mask].
    """
    mask = np.full(len(inputs), True, dtype=bool)

    for prev_index, (prev, next) in enumerate(zip(inputs[:-1], inputs[1:])):
        if next.is_repeated(prev, **kwargs):
            mask[prev_index + 1] = False

    return mask


def get_chord_note_input(
    notes: List[Note],
    measures_df: pd.DataFrame,
    chord_onset: Union[float, Tuple[int, Fraction]],
    chord_offset: Union[float, Tuple[int, Fraction]],
    chord_duration: Union[float, Fraction],
    onset_index: int,
    offset_index: int,
    window: int,
    duration_cache: np.array = None,
) -> np.array:
    """
    Get an np.array or input vectors relative to a given chord.

    Parameters
    ----------
    notes : List[Note]
        A List of all of the Notes in the Piece.
    measures_df : pd.DataFrame
        The measures_df for this particular Piece.
    chord_onset : Union[float, Tuple[int, Fraction]]
        The onset location of the chord.
    chord_offset : Union[float, Tuple[int, Fraction]]
        The offset location of the chord.
    chord_duration : Union[float, Fraction]
        The duration of the chord.
    onset_index : int
        The index of the first note of the chord.
    offset_index : int
        The index of the last note of the chord.
    window : int
        The number of notes to pad on each end of the chord's notes. If this goes past the
        bounds of the given notes list, the remaining vectors will contain only 0.
    duration_cache : np.array
        The duration from each note's onset time to the next note's onset time,
        generated by get_duration_cache(...).

    Returns
    -------
    chord_input : np.array
        The input note vectors for this chord.
    """
    # Add window
    window_onset_index = onset_index - window
    window_offset_index = offset_index + window

    # Get the notes within the window
    first_note_index = max(window_onset_index, 0)
    last_note_index = min(window_offset_index, len(notes))
    chord_notes = notes[first_note_index:last_note_index]

    # Get all note vectors within the window
    min_pitch = min([(note.octave, note.pitch_class) for note in chord_notes])
    if duration_cache is None:
        note_onsets = np.full(len(chord_notes), None)
    else:
        note_onsets = []
        for note_index in range(first_note_index, last_note_index):
            if note_index < onset_index:
                note_onset = -np.sum(duration_cache[note_index:onset_index])
            elif note_index > onset_index:
                note_onset = np.sum(duration_cache[onset_index:note_index])
            else:
                note_onset = Fraction(0)
            note_onsets.append(note_onset)

    note_vectors = np.vstack(
        [
            note.to_vec(
                chord_onset=chord_onset,
                chord_offset=chord_offset,
                chord_duration=chord_duration,
                measures_df=measures_df,
                min_pitch=min_pitch,
                note_onset=note_onset,
            ) for note, note_onset in zip(chord_notes, note_onsets)
        ]
    )

    # Place the note vectors within the final tensor and return
    chord_input = np.zeros((window_offset_index - window_onset_index, note_vectors.shape[1]))
    start = 0 + (first_note_index - window_onset_index)
    end = len(chord_input) - (window_offset_index - last_note_index)
    chord_input[start:end] = note_vectors
    return chord_input


class Piece():
    """
    A single musical piece, which can be from score, midi, or audio.
    """

    def __init__(self, data_type: PieceType):
        """
        Create a new musical Piece object of the given data type.

        Parameters
        ----------
        data_type : PieceType
            The data type of the piece.
        """
        self.DATA_TYPE = data_type

    def get_inputs(self) -> List[Note]:
        """
        Get a list of the inputs for this Piece.

        Returns
        -------
        inputs : np.array
            A List of the inputs for this musical piece.
        """
        raise NotImplementedError

    def get_chord_change_indices(self) -> List[int]:
        """
        Get a List of the indexes (into the input list) at which there are chord changes.

        Returns
        -------
        chord_change_indices : np.array[int]
            The indices (into the inputs list) at which there is a chord change.
        """
        raise NotImplementedError

    def get_chords(self) -> List[Chord]:
        """
        Get a List of the chords in this piece.

        Returns
        -------
        chords : np.array[Chord]
            The chords present in this piece. The ith chord occurs for the inputs between
            chord_change_index i (inclusive) and i+1 (exclusive).
        """
        raise NotImplementedError

    def get_chord_note_inputs(
        self,
        window: int = 2,
        ranges: List[Tuple[int, int]] = None,
    ) -> np.array:
        """
        Get a list of the note input vectors for each chord in this piece, using an optional
        window on both sides. The ith element in the returned array will be an nd-array of
        size (2 * window + num_notes, note_vector_length).

        Parameters
        ----------
        window : int
            Add this many neighboring notes to each side of each input tensor. Fill with 0s if
            this goes beyond the bounds of all notes.
        ranges : List[Tuple[int, int]]
            A List of chord ranges to use to get the inputs, if not using the ground truth
            chord symbols themselves.

        Returns
        -------
        chord_inputs : np.array
            The input note tensor for each chord in this piece.
        """
        raise NotImplementedError

    def get_duration_cache(self) -> List[Fraction]:
        """
        Get a List of the distance from the onset of each input of this Piece to the
        following input. The last value will be the distance from the onset of the last
        input to the offset of the last chord.

        Returns
        -------
        duration_cache : np.array[Fraction]
            A list of the distance from the onset of each input to the onset of the
            following input.
        """
        raise NotImplementedError

    def get_key_change_indices(self) -> List[int]:
        """
        Get a List of the indexes (into the chord list) at which there are key changes.

        Returns
        -------
        key_change_indices : np.array[int]
            The indices (into the chords list) at which there is a key change.
        """
        raise NotImplementedError

    def get_keys(self) -> List[Key]:
        """
        Get a List of the keys in this piece.

        Returns
        -------
        keys : np.array[Key]
            The keys present in this piece. The ith key occurs for the chords between
            key_change_index i (inclusive) and i+1 (exclusive).
        """
        raise NotImplementedError


class ScorePiece(Piece):
    """
    A single musical piece, in score format.
    """

    def __init__(
        self,
        notes_df: pd.DataFrame,
        chords_df: pd.DataFrame,
        measures_df: pd.DataFrame,
        piece_dict: Dict = None,
    ):
        """
        Create a ScorePiece object from the given 3 pandas DataFrames.

        Parameters
        ----------
        notes_df : pd.DataFrame
            A DataFrame containing information about the notes contained in the piece.
        chords_df : pd.DataFrame
            A DataFrame containing information about the chords contained in the piece.
        measures_df : pd.DataFrame
            A DataFrame containing information about the measures in the piece.
        piece_dict : Dict
            An optional dict, to load data from instead of calculating everything from the dfs.
        """
        super().__init__(PieceType.SCORE)
        self.measures_df = measures_df

        if piece_dict is None:
            notes = np.array([
                [note, note_id] for note_id, note in enumerate(notes_df.apply(
                    Note.from_series,
                    axis='columns',
                    measures_df=measures_df,
                    pitch_type=PitchType.TPC,
                )) if note is not None
            ])
            self.notes, self.note_ilocs = np.hsplit(notes, 2)
            self.notes = np.squeeze(self.notes)
            self.note_ilocs = np.squeeze(self.note_ilocs).astype(int)

            chords = np.array([
                [chord, chord_id] for chord_id, chord in enumerate(chords_df.apply(
                    Chord.from_series,
                    axis='columns',
                    measures_df=measures_df,
                    pitch_type=PitchType.TPC,
                )) if chord is not None
            ])
            chords, chord_ilocs = np.hsplit(chords, 2)
            chords = np.squeeze(chords)
            chord_ilocs = np.squeeze(chord_ilocs).astype(int)

            # Remove accidentally repeated chords
            non_repeated_mask = get_reduction_mask(chords, kwargs={'use_inversion': True})
            self.chords = []
            for chord, mask in zip(chords, non_repeated_mask):
                if mask:
                    self.chords.append(chord)
                else:
                    self.chords[-1].merge_with(chord)
            self.chords = np.array(self.chords)
            self.chord_ilocs = chord_ilocs[non_repeated_mask]

            # The index of the notes where there is a chord change
            self.chord_changes = np.zeros(len(self.chords), dtype=int)
            note_index = 0
            for chord_index, chord in enumerate(self.chords):
                while (
                    note_index + 1 < len(self.notes) and
                    self.notes[note_index].onset < chord.onset
                ):
                    note_index += 1
                self.chord_changes[chord_index] = note_index

            key_cols = chords_df.loc[
                chords_df.index[self.chord_ilocs],
                [
                    'globalkey',
                    'globalkey_is_minor',
                    'localkey_is_minor',
                    'localkey',
                    'relativeroot',
                ]
            ]
            key_cols = key_cols.fillna('-1')
            changes = key_cols.ne(key_cols.shift()).fillna(True)

            key_changes = np.arange(len(changes))[changes.any(axis=1)]
            keys = np.array(
                [
                    key for key in chords_df.loc[
                        chords_df.index[self.chord_ilocs[key_changes]]
                    ].apply(Key.from_series, axis='columns', tonic_type=PitchType.TPC)
                    if key is not None
                ]
            )

            # Remove accidentally repeated keys
            non_repeated_mask = get_reduction_mask(keys, kwargs={'use_relative': True})
            self.keys = keys[non_repeated_mask]
            self.key_changes = key_changes[non_repeated_mask]

        else:
            self.notes = np.array([Note(**note) for note in piece_dict['notes']])
            self.chords = np.array([Chord(**chord) for chord in piece_dict['chords']])
            self.keys = np.array([Key(**key) for key in piece_dict['keys']])
            self.chord_changes = np.array(piece_dict['chord_changes'])
            self.key_changes = np.array(piece_dict['key_changes'])

    def get_duration_cache(self):
        if not hasattr(self, 'duration_cache'):
            fake_last_note = Note(
                0, 0, self.chords[-1].offset, 0, Fraction(0), (0, Fraction(0)), 0, PitchType.TPC
            )

            self.duration_cache = np.array(
                [
                    ru.get_range_length(prev_note.onset, next_note.onset, self.measures_df)
                    for prev_note, next_note in
                    zip(self.notes, list(self.notes[1:]) + [fake_last_note])
                ]
            )

        return self.duration_cache

    def get_inputs(self) -> List[Note]:
        return self.notes

    def get_chord_change_indices(self) -> List[int]:
        return self.chord_changes

    def get_chords(self) -> List[Chord]:
        return self.chords

    def get_chord_note_inputs(self, window: int = 2, ranges: List[Tuple[int, int]] = None):
        if ranges is None:
            chord_note_inputs = [
                get_chord_note_input(
                    self.notes,
                    self.measures_df,
                    chord.onset,
                    chord.offset,
                    chord.duration,
                    onset_index,
                    offset_index,
                    window,
                ) for chord, onset_index, offset_index in zip(
                    self.chords,
                    self.chord_changes,
                    list(self.chord_changes[1:]) + [len(self.notes)],
                )
            ]
        else:
            last_offset = self.chords[-1].offset
            duration_cache = self.get_duration_cache()
            durations = [np.sum(duration_cache[start:end]) for start, end in ranges]

            chord_note_inputs = []
            for duration, (onset_index, offset_index) in tqdm(
                zip(durations, ranges),
                desc="Generating chord classification inputs",
                total=len(ranges),
            ):
                onset = self.notes[onset_index].onset
                try:
                    offset = self.notes[offset_index].onset
                except IndexError:
                    offset = last_offset

                chord_note_inputs.append(
                    get_chord_note_input(
                        self.notes,
                        self.measures_df,
                        onset,
                        offset,
                        duration,
                        onset_index,
                        offset_index,
                        window,
                        duration_cache=duration_cache,
                    )
                )

        return chord_note_inputs

    def get_key_change_indices(self) -> List[int]:
        return self.key_changes

    def get_keys(self) -> List[Key]:
        return self.keys

    def to_dict(self) -> Dict[str, List]:
        return {
            'notes': [note.to_dict() for note in self.get_inputs()],
            'chords': [chord.to_dict() for chord in self.get_chords()],
            'keys': [key.to_dict() for key in self.get_keys()],
            'chord_changes': self.get_chord_change_indices(),
            'key_changes': self.get_key_change_indices(),
        }
