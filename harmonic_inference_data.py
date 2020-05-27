"""A package for working with harmonic inference data. This contains Dataset objects for creating
music data -> chord label datasets from various data formats."""

import pandas as pd
from fractions import Fraction as frac
import torch
from torch.utils.data import Dataset
import numpy as np

import corpus_utils
from corpus_reading import read_dump


MAX_PITCH_DEFAULT = 127
PITCHES_PER_OCTAVE = 12


class MusicScoreDataset(Dataset):
    """Harmonic inference dataset, parsed from tsvs created from MuseScore files."""
    
    def __init__(self, chords_df=None, notes_df=None, measures_df=None, files_df=None,
                 chords_tsv=None, notes_tsv=None, measures_tsv=None, files_tsv=None,
                 use_offsets=True, merge_ties=True):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        chords_tsv : string
            The path of the chords tsv file.
            
        notes_tsv : string
            The path of the notes tsv file.
            
        measures_tsv : string
            The path of the measures tsv file.
            
        files_tsv : string
            The path of the files tsv file.
            
        use_offsets : bool
            True if one of the get_*_vector functions needs offsets. This will
            precalculate them.
            
        merge_ties : bool
            True to use merged_notes for the note vectors. This will precalculate
            them.
            
        get_chord_vector : function
            A function to get a chord label vector from a chord pandas entry.
            
        select_notes : function
            A function to select and return a DataFrame of notes given a chord.
            
        get_note_vector : function
            A function to get a note vector from a note pd.Series.
        """
        assert chords_df is not None or chords_tsv is not None, (
            "Either chords_df or chords_tsv is required."
        )
        self.chords = read_dump(chords_tsv) if chords_df is None else chords_df.copy()
        
        assert notes_df is not None or notes_tsv is not None, (
            "Either notes_df or notes_tsv is required."
        )
        self.notes = read_dump(notes_tsv, index_col=[0,1,2]) if notes_df is None else notes_df.copy()
        
        assert measures_df is not None or measures_tsv is not None, (
            "Either measures_df or measures_tsv is required."
        )
        self.measures = read_dump(measures_tsv) if measures_df is None else measures_df.copy()
        
        assert files_df is not None or files_tsv is not None, (
            "Either files_df or files_tsv is required."
        )
        self.files = read_dump(files_tsv, index_col=0) if files_df is None else files_df.copy()
        
        # Remove measure repeats
        if type(self.measures.iloc[0].next) is list:
            self.measures = corpus_utils.remove_repeats(self.measures)
        
        # Add offsets
        if use_offsets and not all([column in self.notes.columns
                                    for column in ['offset_beat', 'offset_mc']]):
            offset_mc, offset_beat = corpus_utils.get_offsets(self.notes, self.measures)
            self.notes = self.notes.assign(offset_mc=offset_mc, offset_beat=offset_beat)
        
        # Merge ties
        if merge_ties:
            self.notes = corpus_utils.merge_ties(self.notes, measures=self.measures)
        
        self.MAX_PITCH = max(MAX_PITCH_DEFAULT, self.notes.midi.max())
        
        self.notes['midi_pitch_norm'] = self.notes.midi / self.MAX_PITCH
        self.notes['midi_pitch_tpc'] = self.notes.midi % PITCHES_PER_OCTAVE
        self.notes['midi_pitch_octave'] = self.notes.midi // PITCHES_PER_OCTAVE
        
        self.data_points = np.full(len(self.chords), None)
        
        
    def __len__(self):
        return len(self.chords)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if type(idx) is int:
            idx = [idx]
        elif type(idx) is slice:
            start, stop, step = idx.indices(len(self))
            idx = list(range(start, stop, step))
            
        data = []
            
        for index in idx:
            if self.data_points[index] is not None:
                data.append(self.data_points[index])
                continue
                
            chord = self.chords.iloc[index]
            chord_vector = self.get_chord_vector(chord)

            note_vectors = self.get_note_vectors(self.select_notes_with_onset(chord), chord)

            sample = {'notes': note_vectors, 'chord': chord_vector}
            
            self.data_points[index] = sample
            data.append(sample)
            
        if len(data) == 0:
            return None
        if len(data) == 1:
            return data[0]
        return data
    
    
    
    def select_notes_with_onset(self, chord):
        """
        Select the notes which onset during the given a chord.

        Parameters
        ----------
        chord : pd.Series
            The chord whose notes we want.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the selected notes.
        """
        all_notes = corpus_utils.get_notes_during_chord(chord, self.notes)

        has_onset = all_notes.overlap.isna() | (all_notes.overlap == 1)
        return all_notes.loc[has_onset]



    def get_note_vectors(self, notes, chord):
        """
        Get the matrix representation of a given notes.

        Parameters
        ----------
        note : pd.DataFrame
            A pandas dataframe containing each musical note whose vector we need.

        chord : pd.Series
            The chord to which this note belongs.

        Returns
        -------
        matrix : np.array
            The matrix representation of the given notes.
        """
        vector_length = (
            1 +
            PITCHES_PER_OCTAVE +
            self.MAX_PITCH // PITCHES_PER_OCTAVE + 1 +
            2 +
            3 +
            1
        )
        
        matrix = np.zeros((len(notes), vector_length))
        
        # Pitch info
        matrix[:, 0] = notes.midi_pitch_norm
        
        # TPC one-hots
        matrix[np.arange(len(notes)), notes.midi_pitch_tpc.to_numpy(dtype=int) + 1] = 1
        
        # Octave one-hots
        matrix[np.arange(len(notes)), notes.midi_pitch_octave.to_numpy(dtype=int) + 1 + PITCHES_PER_OCTAVE] = 1

        # Metrical level at onset and offset
        for i, (note_id, note) in enumerate(notes.iterrows()):
            file_measures = self.measures.loc[chord.name[0]]
            onset_measure = file_measures.loc[note.mc]
            offset_measure = onset_measure if note.offset_mc == note.mc else file_measures.loc[note.offset_mc]
            
            onset_level = corpus_utils.get_metrical_level(note.mc, note.onset, onset_measure)
            offset_level  = corupus_utils.get_metrical_level(note.offset_mc, note.offset_beat, offset_measure)

            # Duration/rhythmic info as percentage of chord duration
            onset, offset, duration = corpus_utils.get_rhythmic_info_as_proportion_of_range(
                note, (chord.mc, chord.onset), (chord.mc_next, chord.onset_next), file_measures
            )
            
            matrix[i, -6:-1] = [onset_level, offset_level, float(onset), float(offset), float(duration)]
        
        # is min pitch
        matrix[:, -1] = (notes.midi == notes.midi.min()).to_numpy(dtype=int)

        return matrix



    def get_chord_vector(self, chord):
        """
        Get the vector representation of a given chord.

        Parameters
        ----------
        chord : pd.Series
            The pandas row of a chord.

        Returns
        -------
        vector : np.array
            The vector representation of the given chord.
        """
        vector = np.zeros(10)
        
        # Key info
        key = chord.key # Roman numeral
        global_key = chord.globalkey # Capital or lowercase letter

        # Rhythmic info
        file_measures = self.measures.loc[chord.name[0]]
        onset_measure = file_measures.loc[chord.mc]
        offset_measure = onset_measure if chord.mc_next == chord.mc else file_measures.loc[chord.mc_next]
        
        onset_level = corpus_utils.get_metrical_level(chord.mc, chord.onset, onset_measure)
        offset_level  = corupus_utils.get_metrical_level(chord.mc_next, chord.onset_next, offset_measure)
        
        # Bass note
        

        # Chord notes
        
        
        return vector
