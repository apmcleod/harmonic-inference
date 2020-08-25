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

files_dfs = {
    'default': None
}

measures_dfs = {
    'default': None,
    'removed': None,
}

chords_dfs = {
    'default': None,
    'removed': None,
    'dropped': None,
    'offsets': None
}

notes_dfs = {
    'default': None,
    'removed': None,
    'offsets': None,
    'merged': None
}


def test_files():
    global files_dfs, measures_dfs, chords_dfs, notes_dfs
    files_dfs['default'] = read_dump(FILES_TSV, index_col=0)


def test_measures():
    global files_dfs, measures_dfs, chords_dfs, notes_dfs
    measures_dfs['default'] = read_dump(MEASURES_TSV)
    measures_dfs['removed'] = cu.remove_repeats(measures_dfs['default'], remove_unreachable=True)

    for df in measures_dfs.values():
        assert all(df.index.get_level_values(0).isin(files_dfs['default'].index))

    # Well-formedness
    for file_id, piece_df in measures_dfs['removed'].groupby('file_id'):
        piece_df = piece_df.copy()

        assert len(piece_df.loc[piece_df['next'].isnull()]) == 1, "Not exactly 1 mc ends."

        # Check that every measure can be reached at most once
        assert piece_df.next.value_counts().max() == 1

        # Check that every measure can be reached except the start_mc
        assert set(piece_df.mc) - set(piece_df.next) == set([piece_df.iloc[0].mc])

        # Check that every measure points forwards (ensures no disjoint loops)
        assert len(piece_df.loc[piece_df.next <= piece_df.mc]) == 0


def test_chords():
    global files_dfs, measures_dfs, chords_dfs, notes_dfs
    chords_dfs['default'] = read_dump(CHORDS_TSV, low_memory=False)
    chords_dfs['removed'] = cu.remove_unmatched(chords_dfs['default'], measures_dfs['removed'])
    chords_dfs['dropped'] = chords_dfs['removed'].drop(
        chords_dfs['removed'].loc[(chords_dfs['removed'].numeral == '@none') |
                                  chords_dfs['removed'].numeral.isnull()].index
    )
    chords_dfs['offsets'] = cu.add_chord_metrical_data(chords_dfs['dropped'],
                                                       measures_dfs['removed'])

    for df in chords_dfs.values():
        assert all(df.index.get_level_values(0).isin(files_dfs['default'].index))

    assert all(chords_dfs['default'].mc.isin(measures_dfs['default'].mc))
    assert all(chords_dfs['removed'].mc.isin(measures_dfs['removed'].mc))
    assert all(chords_dfs['dropped'].mc.isin(measures_dfs['removed'].mc))
    assert all(chords_dfs['offsets'].mc.isin(measures_dfs['removed'].mc))
    assert all(chords_dfs['offsets'].mc_next.isin(measures_dfs['removed'].mc))


def test_notes():
    global files_dfs, measures_dfs, chords_dfs, notes_dfs
    notes_dfs['default'] = read_dump(NOTES_TSV)
    notes_dfs['removed'] = cu.remove_unmatched(notes_dfs['default'], measures_dfs['removed'])
    notes_dfs['offsets'] = cu.add_note_offsets(notes_dfs['removed'], measures_dfs['removed'])
    notes_dfs['merged'] = cu.merge_ties(notes_dfs['offsets'])

    for df in notes_dfs.values():
        assert all(df.index.get_level_values(0).isin(files_dfs['default'].index))

    assert all(notes_dfs['default'].mc.isin(measures_dfs['default'].mc))
    assert all(notes_dfs['removed'].mc.isin(measures_dfs['removed'].mc))
    assert all(notes_dfs['offsets'].mc.isin(measures_dfs['removed'].mc))
    assert all(notes_dfs['offsets'].offset_mc.isin(measures_dfs['removed'].mc))
    assert all(notes_dfs['merged'].mc.isin(measures_dfs['removed'].mc))
    assert all(notes_dfs['merged'].offset_mc.isin(measures_dfs['removed'].mc))
