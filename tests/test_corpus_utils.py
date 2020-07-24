"""Tests for corpu_utils.py"""
from fractions import Fraction
from typing import Tuple

import pandas as pd
import numpy as np

from harmonic_inference.utils import rhythmic_utils as ru
from harmonic_inference.utils import corpus_utils as cu
from harmonic_inference.data.corpus_reading import read_dump

CHORDS_TSV = 'data/chord_list.tsv'
NOTES_TSV = 'data/note_list.tsv'
MEASURES_TSV = 'data/measure_list.tsv'

chords_df = read_dump(CHORDS_TSV)
notes_df = read_dump(NOTES_TSV, index_col=[0, 1, 2])
measures_df = read_dump(MEASURES_TSV)

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
        mc_current = 0
        measures = []
        while mc_current is not None:
            assert isinstance(mc_current, int), "Next is not int."
            assert mc_current not in measures, "Visited measure twice."
            measures.append(mc_current)
            mc_current = piece_df.loc[(piece_id, mc_current), 'next']




def test_get_offsets():
    num_tests = 10000
    indexes = np.random.randint(low=0, high=len(notes_df), size=num_tests)

    for i in indexes:
        note = notes_df.iloc[i]
        range_len = ru.get_range_length((note.mc, note.onset), (note.offset_mc, note.offset_beat),
                                        removed_repeats.loc[note.name[0]])
        assert range_len == note.duration, "Note duration not equal to onset offset range"




def test_get_notes_during_chord():
    def comes_before(t1_loc: Tuple[int, Fraction], t2_loc: Tuple[int, Fraction]) -> bool:
        t1_mc, t1_beat = t1_loc
        t2_mc, t2_beat = t2_loc
        if t1_mc < t2_mc:
            return True
        if t1_mc > t2_mc:
            return False
        return t1_beat < t2_beat

    num_tests = 1000
    indexes = np.random.randint(low=0, high=len(chords_df), size=num_tests)
    return_sizes = []
    return_non_onsets = []

    for i in indexes:
        chord = chords_df.iloc[i]
        notes = cu.get_notes_during_chord(chord, notes_df)
        return_sizes.append(len(notes))
        return_non_onsets.append(0)
        for _, note in notes.iterrows():
            if pd.isna(note.overlap):
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is not after chord
                assert not comes_before((chord.mc_next, chord.onset_next),
                                        (note.offset_mc, note.offset_beat))

            elif note.overlap == -1:
                return_non_onsets[-1] += 1
                # Note onset is before chord
                assert comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is not after chord
                assert not comes_before((chord.mc_next, chord.onset_next),
                                        (note.offset_mc, note.offset_beat))

            elif note.overlap == 0:
                return_non_onsets[-1] += 1
                # Note onset is before chord
                assert comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is after chord
                assert comes_before((chord.mc_next, chord.onset_next),
                                    (note.offset_mc, note.offset_beat))

            elif note.overlap == 1:
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is after chord
                assert comes_before((chord.mc_next, chord.onset_next),
                                    (note.offset_mc, note.offset_beat))

            else:
                assert False, "Invalid overlap value returned: " + str(note.overlap)

    for list_index, i in enumerate(indexes):
        chord = chords_df.iloc[i]
        notes = cu.get_notes_during_chord(chord, notes_df, onsets_only=True)
        assert len(notes) == return_sizes[list_index] - return_non_onsets[list_index], (
            "Length of returned df incorrect with onsets_only"
        )
        for _, note in notes.iterrows():
            if pd.isna(note.overlap):
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is not after chord
                assert not comes_before((chord.mc_next, chord.onset_next),
                                        (note.offset_mc, note.offset_beat))

            elif note.overlap == -1:
                assert False, "onsets_only returned an overlap -1"

            elif note.overlap == 0:
                assert False, "onsets_only returned an overlap 0"

            elif note.overlap == 1:
                # Note onset is not before chord
                assert not comes_before((note.mc, note.onset), (chord.mc, chord.onset))
                # Note offset is after chord
                assert comes_before((chord.mc_next, chord.onset_next),
                                    (note.offset_mc, note.offset_beat))

            else:
                assert False, "Invalid overlap value returned: " + str(note.overlap)
