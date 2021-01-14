"""A musical Note object, with pitch, duration, and metrical information."""
import inspect
import logging
from fractions import Fraction
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

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
        new_params["tonic_type"] = pitch_type

        if pitch_type == self.pitch_type:
            return Note(**self.params)

        # Convert relative, local, and global tonic
        new_params["pitch_class"] = get_pitch_from_string(
            get_pitch_string(new_params["pitch_class"], self.pitch_type), pitch_type
        )

        return Note(**new_params)

    def to_vec(
        self,
        chord_onset: Union[float, Tuple[int, Fraction]] = None,
        chord_offset: Union[float, Tuple[int, Fraction]] = None,
        chord_duration: Union[float, Fraction] = None,
        measures_df: pd.DataFrame = None,
        min_pitch: Tuple[int, int] = None,
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

        min_pitch : Tuple[int, int]
            The minimum pitch of any note in this set of notes, expressed as a (octave, pitch)
            tuple. None to not include the binary is_lowest vector entry.

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
        octave = np.zeros(127 // NUM_PITCHES[PitchType.MIDI], dtype=np.float16)
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
                            "onset": self.onset[1],
                            "duration": self.duration,
                        }
                    ),
                    chord_onset,
                    chord_offset,
                    measures_df,
                    range_len=chord_duration,
                )
            else:
                onset = note_onset / chord_duration
                duration = self.duration / chord_duration
                offset = onset + duration
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
        min_pitch = [
            1 if min_pitch is not None and (self.octave, self.pitch_class) == min_pitch else 0
        ]
        vectors.append(min_pitch)

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
                pitch = note_row.tpc + TPC_C
                if pitch < 0 or pitch >= NUM_PITCHES[PitchType.TPC]:
                    raise ValueError(f"TPC pitch {pitch} is outside of valid range.")
            elif pitch_type == PitchType.MIDI:
                pitch = note_row.midi % NUM_PITCHES[PitchType.MIDI]
            else:
                raise ValueError(f"Invalid pitch type: {pitch_type}")
            octave = note_row.midi // NUM_PITCHES[PitchType.MIDI]

            # Rhythmic info
            positions = [None, None]
            levels = [None, None]
            for i, (mc, beat) in enumerate(
                zip(
                    [note_row.mc, note_row.offset_mc],
                    [note_row.onset, note_row.offset_beat],
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
                note_row.duration,
                offset,
                offset_level,
                pitch_type,
            )

        except Exception as exception:
            logging.error("Error parsing note from row %s", note_row)
            logging.exception(exception)
            return None


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
        NUM_PITCHES[pitch_type]  # Pitch class
        + 127 // NUM_PITCHES[PitchType.MIDI]  # Octave
        + 14  # 4 onset level, 4 offset level, onset, offset, durations, is_lowest
    )
