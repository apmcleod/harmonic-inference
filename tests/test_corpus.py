"""Tests for corpus_data structure"""
from fractions import Fraction
from typing import Tuple, List
from pathlib import Path

import pandas as pd
import numpy as np

from harmonic_inference.utils import rhythmic_utils as ru
from harmonic_inference.utils import corpus_utils as cu
from harmonic_inference.data.corpus_reading import read_dump

TSV_BASE = Path('corpus_data')
FILES_TSV = TSV_BASE / 'files.tsv'
CHORDS_TSV = TSV_BASE / 'chords.tsv'
NOTES_TSV = TSV_BASE / 'notes.tsv'
MEASURES_TSV = TSV_BASE / 'measures.tsv'

files_df = read_dump(FILES_TSV, index_col=0)
measures_df = read_dump(MEASURES_TSV)
chords_df = read_dump(CHORDS_TSV, low_memory=False)
notes_df = read_dump(NOTES_TSV)

removed_repeats = cu.remove_repeats(measures_df, remove_unreachable=True)

chords_df_removed = cu.remove_unmatched(chords_df, removed_repeats)
chords_df = chords_df.drop(chords_df.loc[(chords_df.numeral == '@none') | chords_df.numeral.isnull()].index)
offsets_chords_df = cu.add_chord_metrical_data(chords_df_removed, removed_repeats)

notes_df_removed = cu.remove_unmatched(notes_df, removed_repeats)
offsets_notes_df = cu.add_note_offsets(notes_df_removed, removed_repeats)
merged_notes_df = cu.merge_ties(offsets_notes_df)


def test_measures():
    for file_id, piece_df in removed_repeats.groupby('file_id'):
        piece_df = piece_df.copy()

        assert len(piece_df.loc[piece_df['next'].isnull()]) == 1, "Not exactly 1 mc ends."

        # Check that every measure can be reached at most once
        assert piece_df.next.value_counts().max() == 1

        # Check that every measure can be reached except the start_mc
        assert set(piece_df.mc) - set(piece_df.next) == set([piece_df.iloc[0].mc])

        # Check that every measure points forwards (ensures no disjoint loops)
        assert len(piece_df.loc[piece_df.next <= piece_df.mc]) == 0


def test_add_chord_metrical_data():
    # Check well-formedness, structure, size, etc
    assert offsets_chords_df.mc_next.dtype == 'Int64'
    assert isinstance(offsets_chords_df.onset_next.values[0], Fraction)
    assert isinstance(offsets_chords_df.duration.values[0], Fraction)
    assert all(offsets_chords_df.index == chords_df_removed.index)
    assert set(offsets_chords_df.columns) - set(chords_df_removed.columns) == set(['mc_next',
                                                                                  'onset_next',
                                                                                  'duration'])
    assert len(set(chords_df_removed.columns) - set(offsets_chords_df.columns)) == 0
    assert len(offsets_chords_df.loc[offsets_chords_df.mc_next.isnull()]) == 0
    assert len(offsets_chords_df.loc[offsets_chords_df.onset_next.isnull()]) == 0

    # Check accuracy
    for file_id, piece_df in offsets_chords_df.loc[:20].groupby('file_id'):
        last_measure = removed_repeats.loc[file_id].loc[removed_repeats.loc[file_id].next.isnull()]
        last_chord = piece_df.iloc[-1]
        assert last_chord.mc_next == last_measure.mc.values[0]
        assert last_chord.onset_next == (last_measure.act_dur + last_measure.offset).values[0]
        assert ru.get_range_length((last_chord.mc, last_chord.onset),
                                   (last_chord.mc_next, last_chord.onset_next),
                                   removed_repeats.loc[file_id]) == last_chord.duration

        for (_, prev_chord), (_, next_chord) in zip(piece_df.iloc[:-1].iterrows(),
                                                    piece_df.iloc[1:].iterrows()):
            assert prev_chord.mc_next == next_chord.mc
            assert prev_chord.onset_next == next_chord.onset
            assert ru.get_range_length((prev_chord.mc, prev_chord.onset),
                                       (next_chord.mc, next_chord.onset),
                                       removed_repeats.loc[file_id]) == prev_chord.duration, (
                f'{file_id}: {prev_chord}'
            )


def test_get_notes_during_chord():
    num_tests = 1000
    indexes = np.random.randint(low=0, high=len(offsets_chords_df), size=num_tests)
    return_sizes = []
    return_non_onsets = []

    for i in indexes:
        chord = offsets_chords_df.iloc[i]
        notes = cu.get_notes_during_chord(chord, offsets_notes_df)
        return_sizes.append(len(notes))
        return_non_onsets.append(0)
        for _, note in notes.iterrows():
            if pd.isna(note.overlap):
                # Note onset is not before chord
                assert (note.mc, note.onset) >= (chord.mc, chord.onset)
                # Note offset is not after chord
                assert (chord.mc_next, chord.onset_next) >= (note.offset_mc, note.offset_beat)

            elif note.overlap == -1:
                return_non_onsets[-1] += 1
                # Note onset is before chord
                assert (note.mc, note.onset) < (chord.mc, chord.onset)
                # Note offset is not after chord
                assert (chord.mc_next, chord.onset_next) >= (note.offset_mc, note.offset_beat)

            elif note.overlap == 0:
                return_non_onsets[-1] += 1
                # Note onset is before chord
                assert (note.mc, note.onset) < (chord.mc, chord.onset)
                # Note offset is after chord
                assert (chord.mc_next, chord.onset_next) < (note.offset_mc, note.offset_beat)

            elif note.overlap == 1:
                # Note onset is not before chord
                assert (note.mc, note.onset) >= (chord.mc, chord.onset)
                # Note offset is after chord
                assert (chord.mc_next, chord.onset_next) < (note.offset_mc, note.offset_beat)

            else:
                assert False, "Invalid overlap value returned: " + str(note.overlap)

    for list_index, i in enumerate(indexes):
        chord = offsets_chords_df.iloc[i]
        notes = cu.get_notes_during_chord(chord, offsets_notes_df, onsets_only=True)
        assert len(notes) == return_sizes[list_index] - return_non_onsets[list_index], (
            "Length of returned df incorrect with onsets_only"
        )
        for _, note in notes.iterrows():
            if pd.isna(note.overlap):
                # Note onset is not before chord
                assert (note.mc, note.onset) >= (chord.mc, chord.onset)
                # Note offset is not after chord
                assert (chord.mc_next, chord.onset_next) >= (note.offset_mc, note.offset_beat)

            elif note.overlap == -1:
                assert False, "onsets_only returned an overlap -1"

            elif note.overlap == 0:
                assert False, "onsets_only returned an overlap 0"

            elif note.overlap == 1:
                # Note onset is not before chord
                assert (note.mc, note.onset) >= (chord.mc, chord.onset)
                # Note offset is after chord
                assert (chord.mc_next, chord.onset_next) < (note.offset_mc, note.offset_beat)

            else:
                assert False, "Invalid overlap value returned: " + str(note.overlap)
