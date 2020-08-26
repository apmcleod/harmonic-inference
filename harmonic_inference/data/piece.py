"""A class storing a musical piece from score, midi, or audio format."""
from typing import Union, Tuple, List
from fractions import Fraction
import logging

import pandas as pd
import numpy as np

from harmonic_inference.data.data_types import KeyMode, PitchType, ChordType, PieceType
from harmonic_inference.utils import harmonic_utils as hu
from harmonic_inference.utils import harmonic_constants as hc


class Note():
    """
    A representation of a musical Note, with pitch, onset, duration, and offset.
    """
    def __init__(self, pitch_class: int, octave: int, onset: Union[float, Tuple[int, Fraction]],
                 duration: Union[float, Fraction], offset: Union[float, Tuple[int, Fraction]],
                 pitch_type: PitchType):
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
        duration : Union[float, Fraction]
            The duration of this Note. Either a float (representing time in seconds), ar a Fraction
            (representing whole notes).
        offset : Union[float, Tuple[int, Fraction]]
            The onset position of this Note. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        pitch_type : PitchType
            The PitchType in which this note's pitch_class is stored. If this is TPC, the
            pitch_class can be later converted into MIDI, but not vice versa.
        """
        self.pitch_class = pitch_class
        self.octave = octave
        self.onset = onset
        self.duration = duration
        self.offset = offset
        self.pitch_type = pitch_type

    def __eq__(self, other):
        if not isinstance(other, Note):
            return False
        return (
            self.pitch_class == other.pitch_class and
            self.octave == other.octave and
            self.onset == other.onset and
            self.duration == other.duration and
            self.offset == other.offset and
            self.pitch_type == other.pitch_type
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f'{hu.get_pitch_string(self.pitch_class, self.pitch_type)}{self.octave}: '
            f'{self.onset}--{self.offset}'
        )

    @staticmethod
    def from_series(note_row: pd.Series, pitch_type: PitchType):
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
            octave = note_row.midi // hc.NUM_PITCHES[PitchType.MIDI]

            onset = (note_row.mc, note_row.onset)
            offset = (note_row.offset_mc, note_row.offset_beat)

            return Note(pitch, octave, onset, note_row.duration, offset, pitch_type)

        except BaseException as e:
            logging.error(f"Error parsing note from row {note_row}")
            logging.exception(e)
            return None


class Chord():
    """
    A musical chord, with a root and base note.
    """
    def __init__(self, root: int, bass: int, chord_type: ChordType, inversion: int,
                 onset: Union[float, Tuple[int, Fraction]],
                 offset: Union[float, Tuple[int, Fraction]],
                 duration: Union[float, Fraction], pitch_type: PitchType):
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
        chord_type : ChordType
            The type of chord this is (major, minor, diminished, etc.)
        inversion : int
            The inversion this chord is in.
        onset : Union[float, Tuple[int, Fraction]]
            The onset position of this Chord. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        offset : Union[float, Tuple[int, Fraction]]
            The offset position of this Chord. Either a float (representing time in seconds), or an
            (int, Fraction) tuple (representing measure count and beat in whole notes).
        duration : Union[float, Fraction]
            The duration of this Chord. Either a float (representing time in seconds), ar a
            Fraction (representing whole notes).
        pitch_type : PitchType
            The PitchType in which this chord's root and bass note are stored. If this is TPC, the
            values can be later converted into MIDI, but not vice versa.
        """
        self.root = root
        self.bass = bass
        self.chord_type = chord_type
        self.inversion = inversion
        self.onset = onset
        self.offset = offset
        self.duration = duration
        self.pitch_type = pitch_type

    def __eq__(self, other):
        if not isinstance(other, Chord):
            return False
        return (
            self.root == other.root and
            self.bass == other.bass and
            self.chord_type == other.chord_type and
            self.inversion == other.inversion and
            self.onset == other.onset and
            self.duration == other.duration and
            self.offset == other.offset and
            self.pitch_type == other.pitch_type
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
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
    def from_series(chord_row: pd.Series, pitch_type: PitchType):
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

        pitch_type : PitchType
            The pitch type to use for the Chord.

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
            local_key = Key.from_series(chord_row, pitch_type, do_relative=False)

            # Root note of chord, absolute
            root_interval = (
                chord_row['root'] if pitch_type == PitchType.TPC else
                hu.tpc_interval_to_midi_interval(chord_row['root'])
            )
            root = hu.transpose_pitch(local_key.tonic, root_interval, pitch_type=pitch_type)

            # Bass note of chord, absolute
            bass_interval = (
                chord_row['bass_note'] if pitch_type == PitchType.TPC else
                hu.tpc_interval_to_midi_interval(chord_row['bass_note'])
            )
            bass = hu.transpose_pitch(local_key.tonic, bass_interval, pitch_type=pitch_type)

            # Additional chord info
            chord_type = hu.get_chord_type_from_string(chord_row['chord_type'])
            inversion = hu.get_chord_inversion(chord_row['figbass'])

            # Rhythmic info - Even "No Chord" symbols have these.
            onset = (chord_row.mc, chord_row.onset)
            offset = (chord_row.mc_next, chord_row.onset_next)
            duration = chord_row.duration

            return Chord(root, bass, chord_type, inversion, onset, offset, duration, pitch_type)

        except BaseException as e:
            logging.error(f"Error parsing chord from row {chord_row}")
            logging.exception(e)
            return None


class Key():
    """
    A musical key, with tonic and mode.
    """
    def __init__(self, tonic: int, mode: KeyMode, tonic_type: PitchType):
        """
        Create a new musical key object.

        Parameters
        ----------
        tonic : int
            An integer representing the pitch class of the tonic of this key. If tonic_type is
            TPC, this is stored as a tonal pitch class (with C = 0, G = 1, etc. around the circle
            of fifths). If tonic_type is MIDI, this is stored as semitones above C
            (with C, B# = 0).
        mode : KeyMode
            The mode of this key.
        tonic_type : PitchType
            The PitchType in which this key's tonic is stored. If this is TPC, the
            tonic can be later converted into MIDI type, but not vice versa.
        """
        self.tonic = tonic
        self.mode = mode
        self.tonic_type = tonic_type

    def __eq__(self, other):
        if not isinstance(other, Key):
            return False
        return (
            self.tonic == other.tonic and
            self.mode == other.mode and
            self.tonic_type == other.tonic_type
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'{hu.get_pitch_string(self.tonic, self.tonic_type)} {self.mode}'

    @staticmethod
    def from_series(chord_row: pd.Series, tonic_type: PitchType, do_relative: bool = True):
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
        do_relative : bool
            True to treat slash chords (e.g., applied dominants) as Keys. False to stop after
            considering the local key only.

        Returns
        -------
        chord : Chord, or None
            The created Note object. If an error occurs, None is returned and the error is logged.
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
            local_tonic = hu.transpose_pitch(global_tonic, local_transposition, pitch_type=tonic_type)

            # Treat applied dominants (and other slash chords) as new keys
            relative_full = chord_row['relativeroot']
            if do_relative and not pd.isna(relative_full):
                # Handle doubly-relative chords iteratively
                for relative in reversed(relative_full.split('/')):
                    # Relativeroot is listed relative to local key. We want it absolute.
                    relative_mode = KeyMode.MINOR if relative[-1].islower() else KeyMode.MAJOR
                    relative_transposition = hu.get_interval_from_numeral(
                        relative, local_mode, pitch_type=tonic_type
                    )
                    relative_tonic = hu.transpose_pitch(local_tonic, relative_transposition,
                                                        pitch_type=tonic_type)
                    local_mode, local_tonic = relative_mode, relative_tonic

            return Key(local_tonic, local_mode, tonic_type)

        except BaseException as e:
            logging.error(f"Error parsing key from row {chord_row}")
            logging.exception(e)
            return None


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
        # pylint: disable=invalid-name
        self.DATA_TYPE = data_type

    def get_inputs(self) -> np.array:
        """
        Get a list of the inputs for this Piece.

        Returns
        -------
        inputs : np.array
            A List of the inputs for this musical piece.
        """
        raise NotImplementedError

    def get_chord_change_indices(self) -> np.array:
        """
        Get a List of the indexes (into the input list) at which there are chord changes.

        Returns
        -------
        chord_change_indices : np.array[int]
            The indices (into the inputs list) at which there is a chord change.
        """
        raise NotImplementedError

    def get_chords(self) -> np.array:
        """
        Get a List of the chords in this piece.

        Returns
        -------
        chords : np.array[Chord]
            The chords present in this piece. The ith chord occurs for the inputs between
            chord_change_index i (inclusive) and i+1 (exclusive).
        """
        raise NotImplementedError

    def get_key_change_indices(self) -> np.array:
        """
        Get a List of the indexes (into the chord list) at which there are key changes.

        Returns
        -------
        key_change_indices : np.array[int]
            The indices (into the chords list) at which there is a key change.
        """
        raise NotImplementedError

    def get_keys(self) -> np.array:
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

    def __init__(self, notes_df: pd.DataFrame, chords_df: pd.DataFrame):
        """
        Create a ScorePiece object from the given 3 pandas DataFrames.

        Parameters
        ----------
        notes_df : pd.DataFrame
            A DataFrame containing information about the notes contained in the piece.
        chords_df : pd.DataFrame
            A DataFrame containing information about the chords contained in the piece.
        """
        super().__init__(PieceType.SCORE)
        notes = np.array([
            [note, note_id] for note_id, note in enumerate(notes_df.apply(
                Note.from_series, axis='columns', pitch_type=PitchType.TPC
            )) if note is not None
        ])
        self.notes, self.note_ilocs = np.hsplit(notes, 2)
        self.notes = np.squeeze(self.notes)
        self.note_ilocs = np.squeeze(self.note_ilocs).astype(int)

        chords = np.array([
            [chord, chord_id] for chord_id, chord in enumerate(chords_df.apply(
                Chord.from_series, axis='columns', pitch_type=PitchType.TPC
            )) if chord is not None
        ])
        self.chords, self.chord_ilocs = np.hsplit(chords, 2)
        self.chords = np.squeeze(self.chords)
        self.chord_ilocs = np.squeeze(self.chord_ilocs).astype(int)

        # The index of the notes where there is a chord change
        self.chord_changes = np.zeros(len(self.chords))
        note_index = 0
        for chord_index, chord in enumerate(self.chords):
            while self.notes[note_index].onset < chord.onset:
                note_index += 1
            self.chord_changes[chord_index] = note_index

        key_cols = chords_df.loc[chords_df.index[self.chord_ilocs], [
            'globalkey', 'globalkey_is_minor', 'localkey_is_minor', 'localkey', 'relativeroot']
        ]
        key_cols = key_cols.fillna('-1')
        changes = key_cols.ne(key_cols.shift()).fillna(True)

        self.key_changes = changes.loc[changes.any(axis=1)].index.to_numpy()
        self.keys = np.array([
            key for key in chords_df.loc[chords_df.index[self.chord_ilocs[self.key_changes]]].apply(
                Key.from_series, axis='columns', tonic_type=PitchType.TPC, do_relative=True
            ) if key is not None
        ])

    def get_inputs(self) -> np.array:
        return self.notes

    def get_chord_change_indices(self) -> np.array:
        return self.chord_changes

    def get_chords(self) -> np.array:
        return self.chords

    def get_key_change_indices(self) -> np.array:
        return self.key_changes

    def get_keys(self) -> np.array:
        return self.keys
