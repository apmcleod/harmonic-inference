"""A package for working with harmonic inference data. This contains Dataset objects for creating
music data -> chord label datasets from various data formats."""

import pandas as pd
from fractions import Fraction as frac
import numpy as np
import traceback
import os
import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from corpus_reading import read_dump
import corpus_utils
import rhythmic_utils
import harmonic_utils




def get_train_valid_test_splits(chords_df=None, notes_df=None, measures_df=None, files_df=None,
                                chords_tsv=None, notes_tsv=None, measures_tsv=None, files_tsv=None,
                                seed=None, train_prop=0.8, test_prop=0.1, valid_prop=0.1,
                                create_h5=True, h5_directory='.', h5_prefix='data', make_dfs=False,
                                transpose_global=False, transpose_local=False):
    """
    chords_df : pd.DataFrame
        The full chords data.
        
    notes_df : pd.DataFrame
        The full notes data.
        
    measures_df : pd.DataFrame
        The full measures data.
        
    files_df : pd.DataFrame
        The full files data.
        
    chords_tsv : string
        The path of a chords tsv file. Used only if chords_df is None.

    notes_tsv : string
        The path of a notes tsv file. Used only if notes_df is None.

    measures_tsv : string
        The path of a measures tsv file. Used only if measures_tsv is None.

    files_tsv : string
        The path of a files tsv file. Used only if files_df is None.
        
    seed : int
        The seed to use for splitting the files into train, valid, and test sets.
        
    train_prop : float
        The proportion of files which should be used as training. train_prop, valid_prop, and test_prop
        are normalized to sum to 1.
        
    valid_prop : float
        The proportion of files which should be used as validation. train_prop, valid_prop, and test_prop
        are normalized to sum to 1.
        
    test_prop : float
        The proportion of files which should be used as testing. train_prop, valid_prop, and test_prop
        are normalized to sum to 1.
        
    create_h5 : boolean
        True to create h5 data files to save the splits (or load from one if it alread exists).
        
    h5_directory : string
        The directory in which to store the h5 data files.
        
    h5_prefix : string
        A prefix to use for the h5 data filenames. They will be named {prefix}_{seed}_{split}.h5.
        
    make_dfs : bool
        Force the Dataset to make the DataFrames even if it loads the data from an h5 file.
        
    transpose_global : bool
        Transpose all chords and notes to global key C maj/A min.
        
    transpose_local : bool
        Transpose all chords and notes to local key C maj/A min.
    """
    assert chords_df is not None or chords_tsv is not None, (
        "Either chords_df or chords_tsv is required."
    )
    if chords_df is None:
        chords_df = read_dump(chords_tsv)

    assert notes_df is not None or notes_tsv is not None, (
        "Either notes_df or notes_tsv is required."
    )
    if notes_df is None:
        notes_df = read_dump(notes_tsv, index_col=[0,1,2])

    assert measures_df is not None or measures_tsv is not None, (
        "Either measures_df or measures_tsv is required."
    )
    if measures_df is None:
        measures_df = read_dump(measures_tsv)

    assert files_df is not None or files_tsv is not None, (
        "Either files_df or files_tsv is required."
    )
    if files_df is None:
        files_df = read_dump(files_tsv, index_col=0)
        
    norm_sum = train_prop + valid_prop + test_prop
    train_prop /= norm_sum
    valid_prop /= norm_sum
    test_prop /= norm_sum
    
    if seed is None:
        seed = np.random.randint(0, 2**32)
    np.random.seed(seed)
    
    # Shuffle and split data
    num_files = len(files_df)
    file_ids = files_df.index.to_numpy()
    np.random.shuffle(file_ids)
    train_ids, valid_ids, test_ids = np.split(file_ids, [int(train_prop * num_files), int((1 - test_prop) * num_files)])
    
    # Create datasets
    train_dataset = MusicScoreDataset(h5_file=os.path.join(h5_directory, f'{h5_prefix}_{seed}_train.h5') if create_h5 else None,
                                      chords_df=chords_df.loc[train_ids], notes_df=notes_df.loc[train_ids],
                                      measures_df=measures_df.loc[train_ids], files_df=files_df.loc[train_ids],
                                      make_dfs=make_dfs, transpose_global=transpose_global, transpose_local=transpose_local)
    valid_dataset = MusicScoreDataset(h5_file=os.path.join(h5_directory, f'{h5_prefix}_{seed}_valid.h5') if create_h5 else None,
                                      chords_df=chords_df.loc[valid_ids], notes_df=notes_df.loc[valid_ids],
                                      measures_df=measures_df.loc[valid_ids], files_df=files_df.loc[valid_ids],
                                      make_dfs=make_dfs, transpose_global=transpose_global, transpose_local=transpose_local)
    test_dataset = MusicScoreDataset(h5_file=os.path.join(h5_directory, f'{h5_prefix}_{seed}_test.h5') if create_h5 else None,
                                     chords_df=chords_df.loc[test_ids], notes_df=notes_df.loc[test_ids],
                                     measures_df=measures_df.loc[test_ids], files_df=files_df.loc[test_ids],
                                     make_dfs=make_dfs, transpose_global=transpose_global, transpose_local=transpose_local)
    
    return train_dataset, valid_dataset, test_dataset




def create_music_score_h5(music_score_dataset, directory='.', filename='music_score_data.h5'):
    """
    Write a MusicScoreDataset object out to chord and note h5 files.
    
    Parameters
    ----------
    music_score_dataset : MusicScoreDataset
        The data we will write out to the h5 file.
        
    directory : string
        The directory in which to save the data files. This will be created if it does not exist.
        
    filename : string
        The filename for the h5py files.
    """
    os.makedirs(directory, exist_ok=True)
    
    note_vectors = []
    note_indexes = []
    chord_vectors = []
    chord_indexes = []
    chord_note_pointer_starts = []
    chord_note_pointer_lengths = []
    chord_rhythm_vectors = []
    chord_one_hots = []
    
    note_vector_index = 0
    for data in tqdm(music_score_dataset):
        if data is None:
            continue
            
        # Raw data
        if len(data['notes']) != 0:
            note_vectors.append(data['notes'])
            note_indexes.append(data['note_indexes'])
        chord_vectors.append(data['chord']['vector'])
        chord_indexes.append(data['chord_index'])
        chord_rhythm_vectors.append(data['chord']['rhythm'])
        chord_one_hots.append(data['chord']['one_hot'])
        
        # Note pointers
        chord_note_pointer_starts.append(note_vector_index)
        chord_note_pointer_lengths.append(len(data['notes']))
        note_vector_index += len(data['notes'])
        
    # Write out data
    h5_file = h5py.File(os.path.join(directory, filename), 'w')
    h5_file.create_dataset('note_vectors', data=np.vstack(note_vectors), compression="gzip")
    h5_file.create_dataset('note_indexes', data=np.vstack(note_indexes), compression="gzip")
    h5_file.create_dataset('chord_vectors', data=np.vstack(chord_vectors), compression="gzip")
    h5_file.create_dataset('chord_indexes', data=np.vstack(chord_indexes), compression="gzip")
    h5_file.create_dataset('chord_rhythm_vectors', data=np.vstack(chord_rhythm_vectors), compression="gzip")
    h5_file.create_dataset('chord_one_hots', data=np.array(chord_one_hots), compression="gzip")
    h5_file.create_dataset('chord_note_pointers', data=np.vstack((chord_note_pointer_starts,
                                                                  chord_note_pointer_lengths)).T, compression="gzip")
    h5_file.close()




def pad_and_collate_samples(batch):
    """
    Collate the samples of a given batch into torch tensors. [chord] (and all fields within)
    are collated as default. [notes] are padded with 0s to the maximum of the length of any
    notes list within the samples before collating (and the original length of each list
    is returned as [num_notes]).
    
    Parameters
    ----------
    batch : list
        A list of samples drawn from a harmonic inference dataset.
        
    Returns
    -------
    collated_batch : dict
        A dict of torch tensors from collating the given samples.
    """
    collated_batch = {}
    collated_batch['chord'] = default_collate([sample['chord'] for sample in batch])
    notes = [torch.tensor(sample['notes']) for sample in batch]
    collated_batch['notes'] = pad_sequence(notes, batch_first=True)
    collated_batch['num_notes'] = torch.tensor([len(n) for n in notes])
    
    return collated_batch
    




class MusicScoreDataset(Dataset):
    """Harmonic inference dataset, parsed from tsvs created from MuseScore files."""
    
    def __init__(self, h5_file=None, h5_overwrite=False, chords_df=None, notes_df=None, measures_df=None, files_df=None,
                 chords_tsv=None, notes_tsv=None, measures_tsv=None, files_tsv=None,
                 use_offsets=True, merge_ties=True, cache=True, make_dfs=False, transpose_global=False,
                 transpose_local=False):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        h5_file : string
            Path of an h5py file to read data from. If this is given and it exists, all other options
            are ignored and data is read exclusively from the h5 file. If this is given but the file
            doesn't exist, the data is pre-computed and written out to the h5 file for faster loading
            during runtime.
            
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
            
        cache : boolean
            Save data points in a cache when they are first created for faster subsequent loading.
            
        make_dfs : bool
            Force the Dataset to make the DataFrames even if it loads the data from an h5 file.
            
        transpose_global : bool
            Transpose all chords and notes to global key C maj/A min.

        transpose_local : bool
            Transpose all chords and notes to local key C maj/A min. Overrides transpose_global if
            both are True.
        """
        self.transpose_global = transpose_global
        self.transpose_local = transpose_local
        if self.transpose_local:
            self.transpose_global = False
            
        # First, check h5_file
        self.h5_file = h5_file
        self.h5_data_present = False
        if self.h5_file is not None:
            if not h5_overwrite and os.path.isfile(h5_file):
                self.h5_data_present = True
            
        if not self.h5_data_present or make_dfs:
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

            self.MAX_PITCH = max(harmonic_utils.MAX_PITCH_DEFAULT, self.notes.midi.max())

            # Pitch info
            self.notes['midi_pitch_norm'] = self.notes.midi / self.MAX_PITCH
            self.notes['midi_pitch_flat'] = self.notes.midi % harmonic_utils.PITCHES_PER_OCTAVE
            self.notes['midi_pitch_octave'] = self.notes.midi // harmonic_utils.PITCHES_PER_OCTAVE

            self.cache = cache
            if self.cache:
                self.data_points = np.full(len(self.chords), None)
                
        if self.h5_file is not None:
            if not self.h5_data_present:
                create_music_score_h5(self, directory=os.path.split(h5_file)[0], filename=os.path.split(h5_file)[1])
                self.h5_data_present = True
            
            with h5py.File(self.h5_file, 'r') as h5_file_obj:
                self.note_vectors = np.array(h5_file_obj['note_vectors'])
                self.note_indexes = np.array(h5_file_obj['note_indexes'])
                self.chord_vectors = np.array(h5_file_obj['chord_vectors'])
                self.chord_indexes = np.array(h5_file_obj['chord_indexes'])
                self.chord_rhythm_vectors = np.array(h5_file_obj['chord_rhythm_vectors'])
                self.chord_one_hots = np.array(h5_file_obj['chord_one_hots'])
                self.chord_note_pointers = np.array(h5_file_obj['chord_note_pointers'])
        
        
    def __len__(self):
        if self.h5_data_present:
            return len(self.chord_vectors)
        else:
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
            if self.h5_data_present:
                data.append(self.get_sample_from_h5_index(index))
                continue
                
            if self.cache and self.data_points[index] is not None:
                data.append(self.data_points[index])
                continue
                
            try:
                chord = self.chords.iloc[index]
                notes = corpus_utils.get_notes_during_chord(chord, self.notes, onsets_only=True)
                
                chord_data, transposition = self.get_chord_data(chord, notes.midi.min())

                note_vectors = self.get_note_vectors(notes, chord, transposition)

                sample = {
                    'notes': note_vectors,
                    'chord': chord_data,
                    'note_indexes': [list(index) for index in notes.index],
                    'chord_index': list(chord.name)
                }

                if self.cache:
                    self.data_points[index] = sample
                data.append(sample)
            except Exception as e:
                print(f'Error at index {index}:')
                traceback.print_exc()
            
        if len(data) == 0:
            return None
        if len(data) == 1:
            return data[0]
        return data
    
    
    def get_sample_from_h5_index(self, index):
        """
        Get a the sample at the given index, parsed from the h5 data fields.
        
        Parameters
        ----------
        index : int
            The index of the sample to return.
            
        Returns
        -------
        sample : dict
            The data point.
        """
        sample = {}
        
        sample['chord'] = {
            'vector': self.chord_vectors[index],
            'one_hot': self.chord_one_hots[index],
            'rhythm': self.chord_rhythm_vectors[index]
        }
        sample['chord_index'] = self.chord_indexes[index]
        
        note_indexes = range(self.chord_note_pointers[index, 0],
                             self.chord_note_pointers[index, 0] + self.chord_note_pointers[index, 1])
        
        sample['notes'] = self.note_vectors[note_indexes]
        sample['note_indexes'] = self.note_indexes[note_indexes]
        
        return sample



    def get_note_vectors(self, notes, chord, transposition):
        """
        Get the matrix representation of a given notes.

        Parameters
        ----------
        note : pd.DataFrame
            A pandas dataframe containing each musical note whose vector we need.

        chord : pd.Series
            The chord to which this note belongs.
            
        transposition : int
            The amount (in semitones) by which to transpose each chord.

        Returns
        -------
        matrix : np.array
            The matrix representation of the given notes.
        """
        vector_length = (
            1 +
            harmonic_utils.PITCHES_PER_OCTAVE +
            self.MAX_PITCH // harmonic_utils.PITCHES_PER_OCTAVE + 1 +
            2 +
            3 +
            1
        )
        
        # Transpose all relevant info
        if transposition != 0:
            notes = notes.copy()
            notes.midi_pitch_norm += transposition / self.MAX_PITCH
            notes.midi_pitch_flat += transposition
            notes.loc[notes.midi_pitch_flat >= 12, 'midi_pitch_octave'] += 1
            notes.loc[notes.midi_pitch_flat < 0, 'midi_pitch_octave'] -= 1
            notes.midi_pitch_flat %= 12
        
        matrix = np.zeros((len(notes), vector_length))
        
        # Pitch info
        matrix[:, 0] = notes.midi_pitch_norm
        
        # TPC one-hots
        matrix[np.arange(len(notes)), notes.midi_pitch_flat.to_numpy(dtype=int) + 1] = 1
        
        # Octave one-hots
        matrix[np.arange(len(notes)), notes.midi_pitch_octave.to_numpy(dtype=int) + 1 + harmonic_utils.PITCHES_PER_OCTAVE] = 1

        # Metrical level at onset and offset
        for i, (note_id, note) in enumerate(notes.iterrows()):
            file_measures = self.measures.loc[chord.name[0]]
            onset_measure = file_measures.loc[note.mc]
            offset_measure = onset_measure if note.offset_mc == note.mc else file_measures.loc[note.offset_mc]
            
            onset_level = rhythmic_utils.get_metrical_level(note.onset, onset_measure)
            offset_level  = rhythmic_utils.get_metrical_level(note.offset_beat, offset_measure)

            # Duration/rhythmic info as percentage of chord duration
            onset, offset, duration = rhythmic_utils.get_rhythmic_info_as_proportion_of_range(
                note, (chord.mc, chord.onset), (chord.mc_next, chord.onset_next), file_measures
            )
            
            matrix[i, -6:-1] = [onset_level, offset_level, float(onset), float(offset), float(duration)]
        
        # is min pitch
        matrix[:, -1] = (notes.midi == notes.midi.min()).to_numpy(dtype=int)

        return matrix



    def get_chord_data(self, chord, lowest_note):
        """
        Get the data of a given chord.

        Parameters
        ----------
        chord : pd.Series
            The pandas row of a chord.
            
        lowest_note : int
            The pitch of the lowest note in this chord. This is used only in the case
            of an error in the bass_step column.

        Returns
        -------
        data : dict
            A dict containing the chord data:
                'one_hot': The overall one-hot chord label, including root and type.
                'vector': The chord vector the model will try to match internally.
                'rhythm': The rhythmic vector the model may use earlier.
        """
        data = {}
        
        # Harmonic info
        
        # Global key absolute [0-12)
        transposition = 0
        global_key, global_key_is_major = harmonic_utils.get_key(chord.globalkey)
        if self.transpose_global:
            if global_key_is_major:
                transposition = 12 - global_key
            else:
                transposition = harmonic_utils.NOTE_TO_INDEX['A'] - global_key
            global_key = 0
        
        # Local key (relative to global key)
        local_key_relative, local_key_is_major = harmonic_utils.get_numeral_semitones(chord.key, global_key_is_major)
        local_key_absolute = (global_key + local_key_relative) % harmonic_utils.PITCHES_PER_OCTAVE
        if self.transpose_local:
            if local_key_is_major:
                transposition = 12 - local_key_absolute
            else:
                transposition = harmonic_utils.NOTE_TO_INDEX['A'] - local_key_absolute
            local_key_absolute = 0
            
        # Fix transposition
        transposition %= 12
        if transposition > 6:
            transposition -= 12
        
        # Applied root (relative to local key)
        if pd.isnull(chord.relativeroot):
            applied_root_relative = 0
            applied_root_is_major = local_key_is_major
            applied_root_absolute = local_key_absolute
        else:
            applied_root_relative, applied_root_is_major = harmonic_utils.get_numeral_semitones(chord.relativeroot, local_key_is_major)
            applied_root_absolute = (local_key_absolute + applied_root_relative) % harmonic_utils.PITCHES_PER_OCTAVE
        
        # Chord tonic (relative to applied root)
        chord_root_relative, chord_is_major = harmonic_utils.get_numeral_semitones(chord.numeral, applied_root_is_major)
        chord_root_absolute = (applied_root_absolute + chord_root_relative) % harmonic_utils.PITCHES_PER_OCTAVE
        
        # Bass note (relative to local key)
        bass_note_relative = harmonic_utils.get_bass_step_semitones(chord.bass_step, local_key_is_major)
        if bass_note_relative is None:
            # bass_step was invalid. Guess based on lowest note in chord
            bass_note_absolute = (lowest_note + transposition) % harmonic_utils.PITCHES_PER_OCTAVE
        else:
            bass_note_absolute = (local_key_absolute + bass_note_relative) % harmonic_utils.PITCHES_PER_OCTAVE
        
        # Chord notes
        chord_type_string = harmonic_utils.get_chord_type_string(chord_is_major, form=chord.form, figbass=chord.figbass)
        chord_vector_relative = harmonic_utils.get_vector_from_chord_type(chord_type_string)
        chord_vector_absolute = harmonic_utils.transpose_chord_vector(chord_vector_relative, chord_root_absolute)
        
        # TODO: chord changes/additions
        
        # vector contains bass_note, root, and the pitch presence vector
        vector = np.zeros(harmonic_utils.PITCHES_PER_OCTAVE * 3)
        vector[bass_note_absolute] = 1
        vector[harmonic_utils.PITCHES_PER_OCTAVE + chord_root_absolute] = 1
        vector[-harmonic_utils.PITCHES_PER_OCTAVE:] = chord_vector_absolute
        data['vector'] = vector
        
        # Target one-hot label
        data['one_hot'] = harmonic_utils.CHORD_TYPES.index(chord_type_string) * harmonic_utils.PITCHES_PER_OCTAVE + chord_root_absolute

        # Rhythmic info
        file_measures = self.measures.loc[chord.name[0]]
        onset_measure = file_measures.loc[chord.mc]
        
        if chord.mc_next == chord.mc:
            offset_measure = onset_measure
            offset_mc = chord.mc_next
            offset_beat = chord.onset_next
        else:
            try:
                offset_measure = file_measures.loc[chord.mc_next]
                offset_mc = chord.mc_next
                offset_beat = chord.onset_next
            except KeyError:
                # mc_next is the downbeat after the last measure
                offset_mc = file_measures.index.max()
                offset_measure = file_measures.loc[offset_mc]
                offset_beat = offset_measure.act_dur
        
        onset_level = rhythmic_utils.get_metrical_level(chord.onset, onset_measure)
        offset_level  = rhythmic_utils.get_metrical_level(offset_beat, offset_measure)
        duration = rhythmic_utils.get_range_length((chord.mc, chord.onset), (offset_mc, offset_beat), file_measures)
        
        # Create rhythmic vector
        data['rhythm'] = np.array([onset_level, offset_level, duration], dtype=float)
        
        return data, transposition
