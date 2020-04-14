"""A package for working with harmonic inference data. This contains Dataset objects for creating
music data -> chord label datasets from various data formats, as well as functions for
transforming the input data into various vector representations."""

import pandas as pd
from fractions import Fraction as frac
import torch
from torch.utils.data import Dataset

import corpus_utils


class MusicScoreDataset(Dataset):
    """Harmonic inference dataset, parsed from tsvs created from MuseScore files."""
    
    def __init__(self, chords_tsv, notes_tsv, get_chord_vector=get_chord_vector,
                 get_note_vector=get_note_vector, transform=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        chords_tsv : string
            The path of the chords tsv file.
            
        notes_tsv : string
            The path of the notes tsv file.
            
        get_chord_vector : function
            A function to get a chord label vector from a chord pandas entry.
            
        get_note_vector : function
            A function to get a note vector from a note pandas entry.
            
        transform : function
            A transform to apply to each returned data point.
        """
        self.chords = pd.read_csv(chords_tsv, sep='\t', na_filter=False)
        self.notes = pd.read_csv(notes_tsv, sep='\t')
        
        self.get_chord_vector = get_chord_vector
        self.get_note_vector = get_note_vector
        self.transform = transform
        
        
    def __len__(self):
        return len(self.chords)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        chord = self.chords.iloc[idx]
        chord_vector = self.get_chord_vector(chord)
        
        notes = np.array([self.get_note_vector(note)
                          for note in self.notes.loc[self.notes.chord_id == chord.chord_id]])
        
        sample = {'notes': notes, 'chords': chord_vector}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
def get_note_vector(note):
    """
    Get the vector representation of a given note.
    
    Parameters
    ----------
    note : dict
        The pandas row of a musical note.
        
    Returns
    -------
    vector : np.array
        The vector representation of the given note.
    """
    midi_pitch = note.midi
    note_onset = note.onset
    note_duration = note.duration
    note_offset = get_note_offset()


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
    pass