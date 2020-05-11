"""A package for working with harmonic inference data. This contains Dataset objects for creating
music data -> chord label datasets from various data formats, as well as functions for
transforming the input data into various vector representations."""

import pandas as pd
from fractions import Fraction as frac
import torch
from torch.utils.data import Dataset

import corpus_utils
from corpus_reading import read_dump


class MusicScoreDataset(Dataset):
    """Harmonic inference dataset, parsed from tsvs created from MuseScore files."""
    
    def __init__(self, chords_tsv, notes_tsv, measures_tsv, files_tsv, use_offsets=True,
                 merge_ties=True, get_chord_vector=get_chord_vector,
                 select_notes=select_notes_with_onset, get_note_vector=get_note_vector,
                 transform=None):
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
        self.chords = read_dump(chords_tsv)
        self.notes = read_dump(notes_tsv, index_col=[0,1,2])
        self.measures = read_dump(measures_tsv)
        self.files = read_dump(files_tsv, index_col=0)
        
        # Add offsets
        if use_offsets:
            offset_mc, offset_beat = corpus_utils.get_offsets(self.notes, self.measures)
            self.notes = self.notes.assign(offset_mc=offset_mc, offset_beat=offset_beat)
        
        # Merge ties
        if merge_ties:
            self.notes = corpus_utils.merge_ties(self.notes, measures=self.measures)
        
        self.get_chord_vector = get_chord_vector
        self.select_notes = select_notes
        self.get_note_vector = get_note_vector
        self.transform = transform
        
        
    def __len__(self):
        return len(self.chords)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        chord = self.chords.iloc[idx]
        chord_vector = self.get_chord_vector(chord)
        
        note_vectors = np.array([self.get_note_vector(note, chord) for idx, note in
                                 self.select_notes(chord, self.notes).iterrows()])
        
        sample = {'notes': note_vectors, 'chord': chord_vector}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
def select_notes_with_onset(chord, notes):
    """
    Select the notes which onset during the given a chord.
    
    Parameters
    ----------
    chord : pd.Series
        The chord whose notes we want.
        
    notes : pd.DataFrame
        A DataFrame containing the notes to select from.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected notes.
    """
    all_notes = corpus_utils.get_notes_during_chord(chord, notes)
    
    has_onset = all_notes.overlap.isna() | (all_notes.overlap == 1)
    return all_notes.loc[has_onset]
    
    
    
def get_note_vector(note, chord):
    """
    Get the vector representation of a given note.
    
    Parameters
    ----------
    note : pd.Series
        The pandas row of a musical note.
        
    chord : pd.Series
        The chord to which this note belongs.
        
    Returns
    -------
    vector : np.array
        The vector representation of the given note.
    """
    midi_pitch = note.midi
    onset_beat = note.onset
    onset_mc = note.mc
    duration = note.duration
    offset_beat = note.offset_beat
    offset_mc = note.offset_mc
    pass



def get_chord_vector(chord):
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
    global_key = chord.global_key # Capital or lowercase letter
    
    # Rhythmic info
    onset_beat = chord.onset
    duration = chord.chord_length
    
    # bass note
    
    
    # Chord notes
    
    pass
