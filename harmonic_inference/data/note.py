"""A musical Note object, with pitch, duration, and metrical information."""
import inspect
import logging
from fractions import Fraction
from typing import Dict, Tuple, Union

import music21
import numpy as np
import pandas as pd

from harmonic_inference.data.corpus_constants import MEASURE_OFFSET, NOTE_ONSET_BEAT
from harmonic_inference.data.data_types import PitchType
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES, TPC_C
from harmonic_inference.utils.harmonic_utils import get_pitch_from_string, get_pitch_string
from harmonic_inference.utils.rhythmic_utils import (
    get_metrical_level,
    get_rhythmic_info_as_proportion_of_range,
)


class Note:
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
        mc_onset: Fraction = None,
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
        mc_onset : Fraction
            The mc_onset of this note, if wanted later. If None, onset[1] is stored as mc_onset.
        """
        self.pitch_class = pitch_class
        self.octave = octave
        self.onset = onset
        self.onset_level = onset_level
        self.duration = duration
        self.offset = offset
        self.offset_level = offset_level
        self.pitch_type = pitch_type
        self.mc_onset = self.onset[1] if mc_onset is None else mc_onset

        self.params = inspect.getfullargspec(Note.__init__).args[1:]

    def get_vector_length(self) -> int:
        """
        Get the length of this Note's vector.

        Returns
        -------
        length : int
            The length of this Note's vector.
        """
        return get_note_vector_length(self.pitch_type)

    def to_pitch_type(self, pitch_type: PitchType) -> "Note":
        """
        Convert this Note to a different PitchType, and return a new copy.

        Parameters
        ----------
        pitch_type : PitchType
            Return a copy of this note, with this pitch_type.

        Returns
        -------
        note : Note
            A copy of this note, with the given pitch type.
        """
        new_params = {key: getattr(self, key) for key in self.params}
        new_params["pitch_type"] = pitch_type

        if pitch_type == self.pitch_type:
            return Note(**new_params)

        # Convert pitch
        new_params["pitch_class"] = get_pitch_from_string(
            get_pitch_string(new_params["pitch_class"], self.pitch_type), pitch_type
        )

        return Note(**new_params)

    def get_midi_note_number(self) -> int:
        """
        Get the MIDI note number of this note, where 0 == C0.

        Returns
        -------
        midi : int
            The MIDI note number of this Note's pitch.
        """
        if self.pitch_type == PitchType.MIDI:
            return NUM_PITCHES[PitchType.TPC] * self.octave + self.pitch_class

        midi_pitch_class = get_pitch_from_string(
            get_pitch_string(self.pitch_class, self.pitch_type), PitchType.MIDI
        )
        return NUM_PITCHES[PitchType.MIDI] * self.octave + midi_pitch_class

    def to_vec(
        self,
        chord_onset: Union[float, Tuple[int, Fraction]] = None,
        chord_offset: Union[float, Tuple[int, Fraction]] = None,
        chord_duration: Union[float, Fraction] = None,
        measures_df: pd.DataFrame = None,
        min_pitch: Tuple[int, int] = None,
        max_pitch: Tuple[int, int] = None,
        note_onset: Fraction = None,
        dur_from_prev: Union[float, Fraction] = None,
        dur_to_next: Union[float, Fraction] = None,
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
            Otherwise, this needs:
                'mc' (int): The measure number of each measure.
                'timesig' (str): The time signature of the measure.
                'act_dur' (Fraction): The duration of the measure in whole notes.
                'offset' (Fraction): The starting position of the measure in whole notes.
                                    Should be 0 except for incomplete pick-up measures.
                'next' (int): The mc of the measure that follows each measure
                              (or None for the last measure).

        min_pitch : Tuple[int, int]
            The minimum pitch of any note in this set of notes, expressed as an (octave,
            MIDI note number) tuple. None to not include the binary is_lowest vector entry,
            or any other relative pitch height measures.

        max_pitch : Tuple[int, int]
            The maximum pitch of any note in this set of notes, expressed as an (octave,
            MIDI note number) tuple. If this or min_pitch is None, the chord-relative
            normalized pitch height will not be available.

        note_onset : Fraction
            The duration from the chord onset to the note's onset. If given, this speeds up
            computation by eliminating a call to rhythmic_utils.get_range(...).

        dur_from_prev : Union[float, Fraction]
            The duration from to this note's onset from the previous note's onset.

        dur_to_next : Union[float, Fraction]
            The duration from this note's onset to the next note's onset.

        Returns
        -------
        vector : np.array
            The vector of this Note.
        """
        vectors = []

        # Pitch as one-hot
        pitch = np.zeros(NUM_PITCHES[self.pitch_type], dtype=np.float16)
        pitch[self.pitch_class] = 1
        vectors.append(pitch)

        # Octave as one-hot
        num_octaves = 127 // NUM_PITCHES[PitchType.MIDI]
        octave = np.zeros(num_octaves, dtype=np.float16)
        octave[self.octave] = 1
        vectors.append(octave)

        # Onset metrical level as one-hot
        onset_level = np.zeros(4, dtype=np.float16)
        onset_level[self.onset_level] = 1
        vectors.append(onset_level)

        # Offset metrical level as one-hot
        offset_level = np.zeros(4, dtype=np.float16)
        offset_level[self.offset_level] = 1
        vectors.append(offset_level)

        # onset, offset, duration as floats, as proportion of chord's range
        if (
            chord_onset is not None
            and chord_offset is not None
            and chord_duration is not None
            and measures_df is not None
        ):
            if note_onset is None:
                onset, offset, duration = get_rhythmic_info_as_proportion_of_range(
                    pd.Series(
                        {
                            "mc": self.onset[0],
                            NOTE_ONSET_BEAT: self.onset[1],
                            "duration": self.duration,
                        }
                    ),
                    chord_onset,
                    chord_offset,
                    measures_df,
                    range_len=chord_duration,
                )
            else:
                try:
                    onset = note_onset / chord_duration
                    duration = self.duration / chord_duration
                    offset = onset + duration
                except Exception:
                    # Bugfix for chord duration 0, due to an error in the TSVs
                    onset = Fraction(1)
                    duration = Fraction(1)
                    offset = Fraction(1)
            metrical = np.array([onset, offset, duration], dtype=np.float16)
            vectors.append(metrical)
        else:
            vectors.append(np.zeros(3, dtype=np.float16))

        # Duration to surrounding notes
        durations = [
            0 if dur_from_prev is None else dur_from_prev,
            0 if dur_to_next is None else dur_to_next,
        ]
        vectors.append(durations)

        # Binary -- is this the lowest note in this set of notes
        midi_note_number = self.get_midi_note_number()
        is_min = [1 if min_pitch is not None and midi_note_number == min_pitch[1] else 0]
        vectors.append(is_min)

        # Octave related to surrounding notes as one-hot
        relative_octave = np.zeros(num_octaves, dtype=np.float16)
        lowest_octave = 0 if min_pitch is None else min_pitch[0]
        relative_octave[self.octave - lowest_octave] = 1
        vectors.append(relative_octave)

        # Normalized pitch height
        norm_pitch_height = [midi_note_number / 127]
        vectors.append(norm_pitch_height)

        # Relative to surrounding notes
        if min_pitch is not None and max_pitch is not None:
            range_size = max_pitch[1] - min_pitch[1]

            # If min pitch equals max pitch, we set the range to 1 and every note will have
            # norm_relative = 0 (as if they were all the bass note).
            if range_size == 0:
                range_size = 1
                max_pitch = (max_pitch[0], max_pitch[1] + 1)

            relative_norm_pitch_height = [(midi_note_number - min_pitch[1]) / range_size]
            vectors.append(relative_norm_pitch_height)

        else:
            vectors.append([0])

        return np.concatenate(vectors).astype(np.float16)

    def to_dict(self) -> Dict:
        """
        Convert this Note to a dictionary, which can be called into the Note constructor as
        Note(**dict) to recreate a copy of this Note.

        Returns
        -------
        note_dict : Dict
            A dictionary representation of all of the fields of this Note.
        """
        return {field: getattr(self, field) for field in self.params}

    def __eq__(self, other: "Note") -> bool:
        if not isinstance(other, Note):
            return False
        for field in self.params:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def __repr__(self) -> str:
        params = ", ".join([f"{field}={getattr(self, field)}" for field in self.params])
        return f"Note({params})"

    def __str__(self) -> str:
        return (
            f"{get_pitch_string(self.pitch_class, self.pitch_type)}{self.octave}: "
            f"{self.onset}--{self.offset}"
        )

    @staticmethod
    def from_series(
        note_row: pd.Series,
        measures_df: pd.DataFrame,
        pitch_type: PitchType,
        levels_cache: Dict[str, Dict[Fraction, int]] = None,
    ) -> "Note":
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
        levels_cache : Dict[str, Dict[Fraction, int]]
            If given, a dictionary-based cache mapping time signatures to a 2nd dictionary mapping
            beat positions to metrical levels. The outer-most dictionary should be a default-dict
            returning by default an empty dict.

        Returns
        -------
        note : Note, or None
            The created Note object. If an error occurs, None is returned and the error is logged.
        """
        try:
            if pitch_type == PitchType.TPC:
                pitch = note_row["tpc"] + TPC_C
                if pitch < 0 or pitch >= NUM_PITCHES[PitchType.TPC]:
                    raise ValueError(f"TPC pitch {pitch} is outside of valid range.")
            elif pitch_type == PitchType.MIDI:
                pitch = note_row["midi"] % NUM_PITCHES[PitchType.MIDI]
            else:
                raise ValueError(f"Invalid pitch type: {pitch_type}")
            octave = note_row["midi"] // NUM_PITCHES[PitchType.MIDI]

            # Rhythmic info
            positions = [None, None]
            levels = [None, None]
            for i, (mc, beat) in enumerate(
                zip(
                    [note_row["mc"], note_row["offset_mc"]],
                    [note_row[NOTE_ONSET_BEAT], note_row["offset_beat"]],
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

            return Note(
                pitch,
                octave,
                onset,
                onset_level,
                note_row["duration"],
                offset,
                offset_level,
                pitch_type,
                mc_onset=note_row["mc_onset"],
            )

        except Exception as exception:
            logging.error("Error parsing note from row %s", note_row)
            logging.exception(exception)
            return None

    @staticmethod
    def from_music21(
        m21_note: music21.note.Note,
        measures_df: pd.DataFrame,
        mc: int,
        pitch_type: PitchType,
        m21_chord: music21.chord.Chord = None,
        levels_cache: Dict[str, Dict[Fraction, int]] = None,
    ) -> "Note":
        """
        Create a new Note object from a pd.Series, and return it.

        Parameters
        ----------
        m21_note : music21.note.Note
            A music21 Note object to turn into our Note.
        measures_df : pd.DataFrame
            A pd.DataFrame of the measures in the piece of the note. It is used to get metrical
            levels of the note's onset and offset. Must have at least the columns:
                'mc' (int): The measure number, to match with the note's onset and offset.
                'timesig' (str): The time signature of each measure.
                'start' (Fraction): The position at the start of each measure, in whole notes
                                    since the beginning of the piece.
        mc : int
            The mc of the measure containing the note's onset.
        pitch_type : PitchType
            The pitch type to use for the Note.
        m21_chord : music21.chord.Chord
            If a note is in a chord, it doesn't have rhythmic attributes. Rather, these belong to
            a chord object in music21. This is that chord object.
        levels_cache : Dict[str, Dict[Fraction, int]]
            If given, a dictionary-based cache mapping time signatures to a 2nd dictionary mapping
            beat positions to metrical levels. The outer-most dictionary should be a default-dict
            returning by default an empty dict.

        Returns
        -------
        note : Note
            The created Note object.
        """
        m21_rhythmic = m21_note if m21_chord is None else m21_chord

        note_start = Fraction(m21_rhythmic.offset) / 4
        note_duration = Fraction(m21_rhythmic.quarterLength) / 4

        onset_measure = measures_df.loc[measures_df["mc"] == mc].iloc[0]

        # Find the offset measure
        offset_measure = onset_measure
        tmp_duration = note_duration + note_start - onset_measure[MEASURE_OFFSET]
        while tmp_duration >= offset_measure["act_dur"] and not pd.isna(offset_measure["next"]):
            tmp_duration -= offset_measure["act_dur"]
            offset_measure = measures_df.loc[measures_df["mc"] == offset_measure["next"]].iloc[0]

        onset_beat = note_start
        offset_beat = tmp_duration + offset_measure[MEASURE_OFFSET]

        onset_beat += onset_measure[MEASURE_OFFSET]
        offset_beat += offset_measure[MEASURE_OFFSET]

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

        return Note(
            get_pitch_from_string(m21_note.pitch.name.replace("-", "b"), pitch_type),
            m21_note.octave,
            (onset_measure["mc"], onset_beat),
            onset_level,
            note_duration,
            (offset_measure["mc"], offset_beat),
            offset_level,
            pitch_type,
            mc_onset=onset_beat - onset_measure["mc_offset"],
        )


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
    num_octaves = 127 // NUM_PITCHES[PitchType.MIDI]
    # 4 onset level
    # 4 offset level
    # 3 onset, offset, duration relative to chord
    # 2 durations to next and from prev
    # 1 is_lowest
    # 1 normalized pitch height
    # 1 normalized pitch height relative to window
    extra = 16

    return (
        NUM_PITCHES[pitch_type]  # Pitch class
        + num_octaves  # Absolute octave
        + num_octaves  # Relative octave (above lowest note in chord window)
        + extra
    )
