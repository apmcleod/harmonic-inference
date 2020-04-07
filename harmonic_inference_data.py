"""A package for working with harmonic inference data."""

import torch
from torch.utils.data import Dataset


class HarmonicInferenceDataset(Dataset):
    """Harmonic inference dataset. Musical notes -> chord symbols."""
    
    def __init__(self, chords_tsv, notes_tsv, get_chord_vector, get_note_vector, transform=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        chords_tsv : string
            The path of the chords tsv file.
            
        notes_tsv : string
            The path of the notes tsv file.
            
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
    