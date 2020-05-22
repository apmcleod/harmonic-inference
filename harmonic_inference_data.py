"""A package for working with harmonic inference data. This contains Dataset objects for creating
music data -> chord label datasets from various data formats, as well as functions for
transforming the input data into various vector representations."""

import pandas as pd
from fractions import Fraction as frac
import torch
from torch.utils.data import Dataset
import numpy as np

import corpus_utils
from corpus_reading import read_dump


class MusicScoreDataset(Dataset):
    """Harmonic inference dataset, parsed from tsvs created from MuseScore files."""
    
    def __init__(self, chords_df=None, notes_df=None, measures_df=None, files_df=None,
                 chords_tsv=None, notes_tsv=None, measures_tsv=None, files_tsv=None,
                 use_offsets=True, merge_ties=True, transform=None):
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
            
        transform : function
            A transform to apply to each returned data point.
        """
        assert chords_df is not None or chords_tsv is not None, (
            "Either chords_df or chords_tsv is required."
        )
        self.chords = read_dump(chords_tsv) if chords_df is None else chords_df
        
        assert notes_df is not None or notes_tsv is not None, (
            "Either notes_df or notes_tsv is required."
        )
        self.notes = read_dump(notes_tsv, index_col=[0,1,2]) if notes_df is None else notes_df
        
        assert measures_df is not None or measures_tsv is not None, (
            "Either measures_df or measures_tsv is required."
        )
        self.measures = read_dump(measures_tsv) if measures_df is None else measures_df
        
        assert files_df is not None or files_tsv is not None, (
            "Either files_df or files_tsv is required."
        )
        self.files = read_dump(files_tsv, index_col=0) if files_df is None else files_df
        
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
        
        self.transform = transform
        
        
    def __len__(self):
        return len(self.chords)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        chord = self.chords.iloc[idx]
        chord_vector = self.get_chord_vector(chord)
        
        all_notes = self.select_notes_with_onset(chord)
        
        note_vectors = np.array([self.get_note_vector(note, chord, all_notes.midi.min())
                                 for idx, note in all_notes.iterrows()])
        
        sample = {'notes': note_vectors, 'chord': chord_vector}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
    
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



    def get_note_vector(self, note, chord, lowest_pitch):
        """
        Get the vector representation of a given note.

        Parameters
        ----------
        note : pd.Series
            The pandas row of a musical note.

        chord : pd.Series
            The chord to which this note belongs.
            
        lowest_pitch : int
            The lowest pitch of all notes in this chord.

        Returns
        -------
        vector : np.array
            The vector representation of the given note.
        """
        def create_one_hot(length, value):
            """
            Create and return a one-hot numpy vector of the given length with a 1 in the
            given location.
            
            Parameters
            ----------
            length : int
                The length of the resulting one-hot vector.
                
            value : int
                The index at which to place a 1 in the resulting vector.
                
            Returns
            -------
            vector : np.ndarray
                A vector of length "length" with all 0's except a 1 at index "value".
            """
            vector = np.zeros(length)
            vector[value] = 1
            return vector
        
        # Pitch info
        midi_pitch = note.midi
        midi_pitch_norm = midi_pitch / 88
        
        tpc = midi_pitch % 12
        tpc_one_hot = create_one_hot(12, tpc)
        
        octave = midi_pitch // 12
        octave_one_hot = create_one_hot(88 // 12 + 1, octave)

        # Metrical level at onset and offset
        onset_level, offset_level = corpus_utils.get_metrical_levels(note, measures=self.measures.loc[chord.name[0]])
        
        # Duration/rhythmic info as percentage of chord duration
        onset, offset, duration = corpus_utils.get_rhythmic_info_as_proportion_of_range(
            note, (chord.mc, chord.onset), (chord.mc_next, chord.onset_next), self.measures.loc[chord.name[0]]
        )
        
        # Categorical
        is_lowest = int(midi_pitch == lowest_pitch)

        return np.concatenate((np.array([midi_pitch_norm]), tpc_one_hot, octave_one_hot,
                               np.array([onset_level]), np.array([offset_level]),
                               np.array([float(onset)]), np.array([float(offset)]), np.array([float(duration)]),
                               np.array([is_lowest])))



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
        # Key info
        key = chord.key # Roman numeral
        global_key = chord.globalkey # Capital or lowercase letter

        # Rhythmic info
        onset_beat = chord.onset
        duration = chord.chord_length

        # Bass note


        # Chord notes

        pass
