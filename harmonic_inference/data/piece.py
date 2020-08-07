"""A class storing a musical piece from score, midi, or audio format."""
from typing import Union, Tuple
from fractions import Fraction
from .data_types import *
from harmonic_inference.utils import harmonic_utils as hu


position = Union[float, Tuple[int, Fraction]]


class Note():
    """
    A representation of a musical Note, with pitch, onset, duration, and offset.
    """
    def __init__(self, pitch: int, octave: int, onset: position, duration: Fraction,
                 offset: position, pitch_type: PitchType):
        self.pitch = pitch
        self.octave = octave
        self.onset = onset
        self.duration = duration
        self.offset = offset
        self.pitch_type = pitch_type

    @staticmethod
    def from_series(note_row: pd.Series, measures_df: pd.DataFrame, pitch_type: PitchType) -> Note:
        pitch = note_row.tpc if pitch_type == PitchType.TPC else note_row.midi % 12

        onset = (note_row.mc, note_row.onset)
        offset = (note_row.offset_mc, note_row.offset_beat)

        return Note(pitch, note_row.octaves, onset, note_row.duration, offset, pitch_type)


class Chord():
    """
    A musical chord, with a root and base note.
    """
    def __init__(self, root: int, bass: int, chord_type: ChordType, inversion: int,
                 onset: position, offset: position, duration: Fraction, pitch_type: PitchType):
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
        root_interval = hu.get_interval_from_numeral(chord_row['numeral'], key.mode,
                                                     pitch_type=pitch_type)
        root = hu.transpose(key.tonic, chord_row.root, pitch_type=pitch_type)

        # Bass step is listed relative to local key (not applied dominant)
        local_key = from_series(chord_row, pitch_type, do_relative=False)
        bass_interval = hu.get_interval_from_number(chord_row['bass_step'], local_key.mode,
                                                    pitch_type=pitch_type)
        bass = hu.transpose(local_key.tonic, bass_interval, pitch_type=pitch_type)

        chord_type = hu.get_chord_type(chord_row['numeral'].isupper(), chord_row['form'],
                                       chord_row['figbass'])
        inversion = hu.get_chord_inversion(chord_type, chord_row['figbass'])

        onset = (chord_row.mc, chord_row.onset)
        offset = (chord_row.mc_next, chord_row.onset_next)

        return Chord(root, bass, chord_type, inversion, onset, offset, chord_row.chord_length,
                     pitch_type)


class Key():
    """
    A musical key, with tonic and mode.
    """
    def __init__(self, tonic: int, mode: KeyMode, tonic_type: PitchType):
        self.tonic = tonic
        self.mode = mode
        self.tonic_type = tonic_type

    @staticmethod
    def from_series(chord_row: pd.Series, tonic_type: PitchType, do_relative: bool = True) -> Key:
        global_tonic = hu.get_pitch_from_string(chord_row['globalkey'], pitch_type=tonic_type)
        global_mode = KeyMode.MINOR if chord_row['globalminor'] else KeyMode.MAJOR

        local_mode = KeyMode.MINOR if chord_row['localminor'] else KeyMode.MAJOR
        local_transposition = hu.get_interval_from_numeral(chord_row['key'], global_mode,
                                                           pitch_type=tonic_type)
        local_tonic = hu.transpose_pitch(global_tonic, local_transposition, pitch_type=tonic_type)

        # Treat applied dominants (and other slash chords) as new keys
        relative = chord_row['relative_root']
        if do_relative and not pd.isna(relative):
            relative_mode = KeyMode.MINOR if relative[-1].islower() else KeyMode.MAJOR
            relative_transposition = hu.get_interval_from_numeral(relative, local_mode,
                                                                  pitch_type=tonic_type)
            relative_tonic = hu.transpose_pitch(local_tonic, relative_transposition,
                                                pitch_type=tonic_type)
            local_mode, local_tonic = relative_mode, relative_tonic

        return Key(local_tonic, local_mode, tonic_type)


class Piece():
    """
    A single musical piece, which can be from score, midi, or audio.
    """
    def __init__(self, data_type: PieceType):
        # pylint: disable=invalid-name
        self.DATA_TYPE = data_type

    def get_inputs(self):
        raise NotImplementedError

    def get_chord_change_indices(self):
        raise NotImplementedError

    def get_chords(self):
        raise NotImplementedError

    def get_key_change_indices(self):
        raise NotImplementedError

    def get_keys(self):
        raise NotImplementedError


class ScorePiece(Piece):
    """
    A single musical piece, in score format.
    """
    def __init__(self, notes_df, chords_df, measures_df):
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

    def get_inputs(self):
        return self.notes

    def get_chord_change_indices(self):
        return self.chord_changes

    def get_chords(self):
        return self.chords

    def get_key_change_indices(self):
        return self.key_changes

    def get_keys(self):
        return self.keys
