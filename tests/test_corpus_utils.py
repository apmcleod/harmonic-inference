"""Tests for corpu_utils.py"""
import pytest

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
from fractions import Fraction

import rhythmic_utils as ru
import corpus_utils as cu


from corpus_reading import read_dump

chords_tsv = 'data/chord_list.tsv'
notes_tsv = 'data/note_list.tsv'
measures_tsv = 'data/measure_list.tsv'
#files_tsv = 'data/file_list.tsv'

chords_df = read_dump(chords_tsv)
notes_df = read_dump(notes_tsv, index_col=[0,1,2])
measures_df = read_dump(measures_tsv)
#files_df = read_dump(files_tsv, index_col=0)

# Bugfixes
measures_df.loc[(685, 487), 'next'][0] = 488

removed_repeats = cu.remove_repeats(measures_df)
offset_mc, offset_beat = cu.get_offsets(notes_df, removed_repeats)
notes_df = notes_df.assign(offset_mc=offset_mc, offset_beat=offset_beat)

def test_remove_repeats():
    # Test well-formedness
    for piece_id, piece_df in removed_repeats.groupby('id'):
        assert len(piece_df.loc[piece_df['next'].isnull()]) == 1, "Not exactly 1 mc ends."
        
        # Check that we can iterate through
        mc = 0
        measures = []
        while mc is not None:
            assert type(mc) is int, "Next is not int."
            assert mc not in measures, "Visited measure twice."
            measures.append(mc)
            mc = piece_df.loc[(piece_id, mc), 'next']
            
            
            
            
def test_get_offsets():
    NUM_TESTS = 10000
    indexes = np.random.randint(low=0, high=len(notes_df), size=NUM_TESTS)
    
    for i in indexes:
        note = notes_df.iloc[i]
        range_len = ru.get_range_length((note.mc, note.onset), (note.offset_mc, note.offset_beat), removed_repeats.loc[note.name[0]])
        assert range_len == note.duration, "Note duration not equal to onset offset range"
        
        
        
        
def test_get_notes_during_chord():
    def comes_before(t1, t2):
        t1_mc, t1_beat = t1
        t2_mc, t2_beat = t2
        if t1_mc < t2_mc:
            return True
        if t1_mc > t2_mc:
            return False
        return t1_beat < t2_beat
        
    NUM_TESTS = 1000
    indexes = np.random.randint(low=0, high=len(chords_df), size=NUM_TESTS)
    return_sizes = []
    return_non_onsets = []
    
    for i in indexes:
        chord = chords_df.iloc[i]
        notes = cu.get_notes_during_chord(chord, notes_df)
        return_sizes.append(len(notes))
        return_non_onsets.append(0)
        for note_id, note in notes.iterrows():
            if pd.isna(note.overlap):
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is not after chord
                assert not comes_before((chord.mc_next, chord.onset_next), (note.offset_mc, note.offset_beat))
                
            elif note.overlap == -1:
                return_non_onsets[-1] += 1
                # Note onset is before chord
                assert comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is not after chord
                assert not comes_before((chord.mc_next, chord.onset_next), (note.offset_mc, note.offset_beat))
                
            elif note.overlap == 0:
                return_non_onsets[-1] += 1
                # Note onset is before chord
                assert comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is after chord
                assert comes_before((chord.mc_next, chord.onset_next), (note.offset_mc, note.offset_beat))
                
            elif note.overlap == 1:
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is after chord
                assert comes_before((chord.mc_next, chord.onset_next), (note.offset_mc, note.offset_beat))
                
            else:
                assert False, "Invalid overlap value returned: " + str(note.overlap)
                
    for list_index, i in enumerate(indexes):
        chord = chords_df.iloc[i]
        notes = cu.get_notes_during_chord(chord, notes_df, onsets_only=True)
        assert len(notes) == return_sizes[list_index] - return_non_onsets[list_index], (
            "Length of returned df incorrect with onsets_only"
        )
        for note_id, note in notes.iterrows():
            if pd.isna(note.overlap):
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is not after chord
                assert not comes_before((chord.mc_next, chord.onset_next), (note.offset_mc, note.offset_beat))
                
            elif note.overlap == -1:
                assert False, "onsets_only returned an overlap -1"
                
            elif note.overlap == 0:
                assert False, "onsets_only returned an overlap 0"
                
            elif note.overlap == 1:
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is after chord
                assert comes_before((chord.mc_next, chord.onset_next), (note.offset_mc, note.offset_beat))
                
            else:
                assert False, "Invalid overlap value returned: " + str(note.overlap)
        
