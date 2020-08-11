"""A class storing a musical piece from score, midi, or audio format."""
from typing import Union, Tuple, List
from fractions import Fraction
from .data_types import *
from harmonic_inference.utils import harmonic_utils as hu


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
        self.pitch = pitch
        self.octave = octave
        self.onset = onset
        self.duration = duration
        self.offset = offset
        self.pitch_type = pitch_type

    @staticmethod
    def from_series(note_row: pd.Series, measures_df: pd.DataFrame, pitch_type: PitchType) -> Note:
        pitch = note_row.tpc + hu.TPC_C if pitch_type == PitchType.TPC else note_row.midi % 12

        onset = (note_row.mc, note_row.onset)
        offset = (note_row.offset_mc, note_row.offset_beat)

        return Note(pitch, note_row.octaves, onset, note_row.duration, offset, pitch_type)


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
        self.quality = quality
        self.inversion = inversion
        self.onset = onset
        self.offset = offset
        self.duration = duration
        self.pitch_type = pitch_type

    @staticmethod
    def from_series(chord_row: pd.Series, measures_df: pd.DataFrame,
                    pitch_type: PitchType) -> Chord:
        key = Key.from_series(chord_row, pitch_type)
        root_interval = hu.get_interval_from_scale_degree(
            chord_row['numeral'], True, True, key.mode, pitch_type=pitch_type
        )
        root = hu.transpose_pitch(key.tonic, root_interval, pitch_type=pitch_type)

        # Bass step is listed relative to local key (not applied dominant)
        local_key = from_series(chord_row, pitch_type, do_relative=False)
        bass_interval = hu.get_interval_from_scale_degree(
            chord_row['bass_step'], False, False, local_key.mode, pitch_type=pitch_type
        )
        bass = hu.transpose_pitch(local_key.tonic, bass_interval, pitch_type=pitch_type)

        chord_type = hu.get_chord_type(chord_row['numeral'].isupper(), chord_row['form'],
                                       chord_row['figbass'])
        inversion = hu.get_chord_inversion(chord_row['figbass'])

        onset = (chord_row.mc, chord_row.onset)
        offset = (chord_row.mc_next, chord_row.onset_next)

        return Chord(root, bass, chord_type, inversion, onset, offset, chord_row.chord_length,
                     pitch_type)


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

    @staticmethod
    def from_series(chord_row: pd.Series, tonic_type: PitchType, do_relative: bool = True) -> Key:
        global_tonic = hu.get_pitch_from_string(chord_row['globalkey'], pitch_type=tonic_type)
        global_mode = KeyMode.MINOR if chord_row['globalminor'] else KeyMode.MAJOR

        local_mode = KeyMode.MINOR if chord_row['localminor'] else KeyMode.MAJOR
        local_transposition = hu.get_interval_from_scale_degree(
            chord_row['key'], True, True, global_mode, pitch_type=tonic_type
        )
        local_tonic = hu.transpose_pitch(global_tonic, local_transposition, pitch_type=tonic_type)

        # Treat applied dominants (and other slash chords) as new keys
        relative = chord_row['relative_root']
        if do_relative and not pd.isna(relative):
            relative_mode = KeyMode.MINOR if relative[-1].islower() else KeyMode.MAJOR
            relative_transposition = hu.get_interval_from_scale_degree(
                relative, True, True, local_mode, pitch_type=tonic_type
            )
            relative_tonic = hu.transpose_pitch(local_tonic, relative_transposition,
                                                pitch_type=tonic_type)
            local_mode, local_tonic = relative_mode, relative_tonic

        return Key(local_tonic, local_mode, tonic_type)


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

    def get_inputs(self) -> List:
        """
        Get a list of the inputs for this Piece.

        Returns
        -------
        inputs : List
            A List of the inputs for this musical piece.
        """
        raise NotImplementedError

    def get_chord_change_indices(self) -> List[int]:
        """
        Get a List of the indexes (into the input list) at which there are chord changes.

        Returns
        -------
        chord_change_indices : List[int]
            The indices (into the inputs list) at which there is a chord change.
        """
        raise NotImplementedError

    def get_chords(self) -> List[Chord]:
        """
        Get a List of the chords in this piece.

        Returns
        -------
        chords : List[Chord]
            The chords present in this piece. The ith chord occurs for the inputs between
            chord_change_index i (inclusive) and i+1 (exclusive).
        """
        raise NotImplementedError

    def get_key_change_indices(self) -> List[int]:
        """
        Get a List of the indexes (into the chord list) at which there are key changes.

        Returns
        -------
        key_change_indices : List[int]
            The indices (into the chords list) at which there is a key change.
        """
        raise NotImplementedError

    def get_keys(self) -> List[Key]:
        """
        Get a List of the keys in this piece.

        Returns
        -------
        keys : List[Key]
            The keys present in this piece. The ith key occurs for the chords between
            key_change_index i (inclusive) and i+1 (exclusive).
        """
        raise NotImplementedError


class ScorePiece(Piece):
    """
    A single musical piece, in score format.
    """

    def __init__(self, notes_df: pd.DataFrame, chords_df: pd.DataFrame, measures_df: pd.DataFrame):
        """
        Create a ScorePiece object from the given 3 pandas DataFrames.

        Parameters
        ----------
        notes_df : pd.DataFrame
            A DataFrame containing information about the notes contained in the piece.
        chords_df : pd.DataFrame
            A DataFrame containing information about the chords contained in the piece.
        measures_df : pd.DataFrame
            A DataFrame containing information about the measures contained in the piece.
        """
        super().__init__(PieceType.SCORE)
        self.notes = [
            Note.from_series(note, measures_df, PitchType.TPC)
            for _, note in notes_df.iterrows()
        ]

        self.chords = [
            Chord.from_series(chord, measures_df, PitchType.TPC)
            for _, chord in chords_df.iterrows()
        ]

        self.chord_changes = [0] * len(self.chords)
        note_index = 0
        for chord_index, chord in enumerate(self.chords):
            if self.notes[note_index].onset >= chord.onset:
                self.chord_changes[chord_index] = note_index
            else:
                note_index += 1

        key_cols = chords_df.loc[:, ['globalkey', 'globalminor', 'localminor', 'key',
                                     'relativeroot']]
        key_cols = key_cols.fillna('-1')
        changes = key_cols.ne(key_cols.shift()).fillna(True)

        self.key_changes = changes.loc[changes.any(axis=1)].index.to_list()
        self.keys = [
            Key.from_series(chord, PitchType.TPC, do_relative=True)
            for _, chord in chords_df.loc[self.key_changes].iterrows()
        ]

    def get_inputs(self) -> List[Note]:
        return self.notes

    def get_chord_change_indices(self) -> List[int]:
        return self.chord_changes

    def get_chords(self) -> List[Chord]:
        return self.chords

    def get_key_change_indices(self) -> List[int]:
        return self.key_changes

    def get_keys(self) -> List[Key]:
        return self.keys
