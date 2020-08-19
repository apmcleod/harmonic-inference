"""Tests for corpus_utils.py"""
from fractions import Fraction
from typing import Tuple, List

import pandas as pd
import numpy as np

import harmonic_inference.utils.corpus_utils as cu
import harmonic_inference.utils.rhythmic_utils as ru


measures_dicts = [
    # No offsets
    pd.DataFrame({
        'mc': [1, 2, 3, 4, 5, 6, 7],
        'act_dur': [Fraction(1),
                    Fraction(1),
                    Fraction(1, 2),
                    Fraction(1),
                    Fraction(1, 2),
                    Fraction(1),
                    Fraction(1, 2)],
        'offset': [Fraction(0)] * 7,
        'extra': 0,
        'next': [(1, 2), (3,), (4,), (5, 6), (6, ), (7, ), (1, 2, 3, -1)]
    }),
    # Offsets
    pd.DataFrame({
        'mc': [1, 2, 3, 4, 5, 6, 7],
        'act_dur': [Fraction(1, 2),
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(1),
                    Fraction(1, 4)],
        'offset': [Fraction(1, 2),
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(0),
                    Fraction(1, 4)],
        'extra': 0,
        'next': [(1, 2), (3,), (4,), (6, 5), (6, ), (7, ), (1, 2, -1, 3)]
    }),
    pd.DataFrame({ # Alternate offsets
        'mc': [1, 2, 3, 4, 5, 6, 7],
        'act_dur': [Fraction(1),
                    Fraction(1),
                    Fraction(1, 2),
                    Fraction(1),
                    Fraction(1, 2),
                    Fraction(1),
                    Fraction(1, 2)],
        'offset': [Fraction(0),
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(0),
                    Fraction(1, 2)],
        'extra': 0,
        'next': [(1, 2, 3), (3,), (4,), (2, 5, 6), (2, 6), (7, ), (1, 2, 3, -1)]
    })
]
measures = pd.concat(
    measures_dicts, keys=[0, 1, 2], axis=0, names=['file_id', 'measure_id']
)
measures.mc = measures.mc.astype('Int64')

removed_repeats = cu.remove_repeats(measures, remove_unreachable=True)
removed_repeats_with_unreachable = cu.remove_repeats(measures, remove_unreachable=False)


def test_remove_repeats():
    # Test well-formedness
    assert removed_repeats.next.dtype == 'Int64'
    assert all(measures.columns == removed_repeats.columns)
    assert measures.index.name == removed_repeats.index.name
    columns = list(set(measures.columns) - set(['next']))
    assert measures.loc[removed_repeats.index, columns].equals(removed_repeats.loc[:, columns])

    # Check accuracy
    assert list(removed_repeats.next.to_numpy()) == [
        2, 3, 4, 6, 7, pd.NA, 2, 3, 4, 6, 7, pd.NA, 3, 4, 6, 7, pd.NA
    ]

    # Tests with removed_repeats_with_unreachable
    assert removed_repeats_with_unreachable.next.dtype == 'Int64'
    assert all(measures.columns == removed_repeats_with_unreachable.columns)
    assert measures.index.name == removed_repeats_with_unreachable.index.name
    columns = list(set(measures.columns) - set(['next']))
    assert measures.loc[removed_repeats_with_unreachable.index, columns].equals(
        removed_repeats_with_unreachable.loc[:, columns]
    )

    # Check accuracy
    assert list(removed_repeats_with_unreachable.next.to_numpy()) == [
        2, 3, 4, 6, 6, 7, pd.NA, 2, 3, 4, 6, 6, 7, pd.NA, 3, 3, 4, 6, 6, 7, pd.NA
    ]


def test_remove_unmatched():
    for id_name in ['chord_id', 'note_id']:
        df = pd.DataFrame({
            'file_id': [0, 0, 2, 2, 2],
            id_name: [0, 1, 0, 1, 2],
            'mc': [1, 5, 2, 3, 12],
            'extra': Fraction(2, 3)
        }).set_index(['file_id', id_name])

        df_matched = cu.remove_unmatched(df, removed_repeats)
        assert df_matched.equals(pd.DataFrame({
            'file_id': [0, 2],
            id_name: [0, 1],
            'mc': [1, 3],
            'extra': Fraction(2, 3)
        }).set_index(['file_id', id_name]))

        df_matched = cu.remove_unmatched(df, removed_repeats_with_unreachable)
        assert df_matched.equals(pd.DataFrame({
            'file_id': [0, 0, 2, 2],
            id_name: [0, 1, 0, 1],
            'mc': [1, 5, 2, 3],
            'extra': Fraction(2, 3)
        }).set_index(['file_id', id_name]))


def test_add_chord_metrical_data():
    chords = pd.DataFrame({
        'file_id': [0, 0, 0, 2, 2, 2],
        'onset': [Fraction(1, 2),
                  Fraction(3, 4),
                  Fraction(0),
                  Fraction(3, 4),
                  Fraction(4, 5),
                  Fraction(1, 2)],
        'chord_id': [0, 1, 2, 0, 1, 2],
        'mc': [1, 1, 2, 2, 2, 5],
        'extra': Fraction(2, 3)
    }).set_index(['file_id', 'chord_id'])

    offsets_chords_df = cu.add_chord_metrical_data(chords, removed_repeats_with_unreachable)

    # Check well-formedness, structure, size, etc
    assert offsets_chords_df.mc_next.dtype == 'Int64'
    assert isinstance(offsets_chords_df.onset_next.values[0], Fraction)
    assert isinstance(offsets_chords_df.duration.values[0], Fraction)
    assert all(offsets_chords_df.index == chords.index)
    assert set(offsets_chords_df.columns) - set(chords.columns) == set(['mc_next', 'onset_next',
                                                                        'duration'])
    assert len(set(chords.columns) - set(offsets_chords_df.columns)) == 0
    assert len(offsets_chords_df.loc[offsets_chords_df.mc_next.isnull()]) == 0
    assert len(offsets_chords_df.loc[offsets_chords_df.onset_next.isnull()]) == 0

    # Check accuracy
    assert offsets_chords_df.loc[:, chords.columns].equals(chords)
    assert list(offsets_chords_df.duration.to_numpy()) == [Fraction(1, 4),
                                                           Fraction(1, 4),
                                                           Fraction(4),
                                                           Fraction(1, 20),
                                                           Fraction(79, 20),
                                                           Fraction(7, 4)]
    assert list(offsets_chords_df.mc_next.to_numpy()) == [1, 2, 7, 2, 5, 7]
    assert list(offsets_chords_df.onset_next.to_numpy()) == [Fraction(3, 4),
                                                             Fraction(0),
                                                             Fraction(1, 2),
                                                             Fraction(4, 5),
                                                             Fraction(1, 2),
                                                             Fraction(1)]


def test_add_note_offsets():
    def check_result(note: pd.DataFrame, target_offset_mc: int, target_offset_beat: Fraction):
        """
        Check the result of a call to cu.add_note_offsets with assertions.

        Parameters
        ----------
        note : pd.DataFrame
            A DataFrame with a single note, to be passed to add_note_offsets.
        target_offset_mc : int
            The correct offset_mc for the note.
        target_offset_beat : Fraction
            The correct offset_beat for the note.
        """
        note_offset = cu.add_note_offsets(note, removed_repeats)

        # Check well-formedness, structure, size, etc
        assert note_offset.offset_mc.dtype == 'Int64'
        assert isinstance(note_offset.offset_beat.values[0], Fraction)
        assert all(note_offset.index == note.index)
        assert set(note_offset.columns) - set(note.columns) == set(['offset_beat',
                                                                    'offset_mc'])
        assert len(set(note.columns) - set(note_offset.columns)) == 0
        assert len(note_offset.loc[note_offset.offset_mc.isnull()]) == 0
        assert len(note_offset.loc[note_offset.offset_beat.isnull()]) == 0

        # Check values
        assert note.equals(note_offset.loc[:, ['mc', 'onset', 'duration']])
        assert target_offset_mc == note_offset.offset_mc.values[0]
        assert target_offset_beat == note_offset.offset_beat.values[0]

        # Check with rhythmic utils
        assert ru.get_range_length(
            (note.mc.values[0], note.onset.values[0]), (target_offset_mc, target_offset_beat),
            removed_repeats.loc[note.index.get_level_values('file_id')]
        ) == note.duration.values[0]

        # Check types
        assert isinstance(note_offset.offset_beat.values[0], Fraction)
        assert note_offset.offset_mc.dtype == 'Int64'
        assert note_offset.offset_beat.dtype == Fraction

        # Check structure of result
        assert set(note_offset.columns) == set(
            ['mc', 'onset', 'duration', 'offset_mc', 'offset_beat']
        )
        assert note_offset.index.names == ['file_id', 'note_id']
        assert len(note_offset) == len(note)

    # Tests without offset
    # In same measure
    note = pd.DataFrame({
        'file_id': [0],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(0)],
        'duration': [Fraction(3, 4)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 1
    target_offset_beat = Fraction(3, 4)

    check_result(note, target_offset_mc, target_offset_beat)

    # One measure long
    note = pd.DataFrame({
        'file_id': [0],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(3, 4)],
        'duration': [Fraction(1, 4)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 2
    target_offset_beat = Fraction(0)

    check_result(note, target_offset_mc, target_offset_beat)

    # Multiple measures
    note = pd.DataFrame({
        'file_id': [0],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(0)],
        'duration': [Fraction(20)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 7
    target_offset_beat = Fraction(31, 2)

    check_result(note, target_offset_mc, target_offset_beat)

    # Tests with offsets
    # In same measure
    note = pd.DataFrame({
        'file_id': [1],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(1, 2)],
        'duration': [Fraction(1, 4)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 1
    target_offset_beat = Fraction(3, 4)

    check_result(note, target_offset_mc, target_offset_beat)

    # One measure long
    note = pd.DataFrame({
        'file_id': [1],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(1, 2)],
        'duration': [Fraction(1, 2)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 2
    target_offset_beat = Fraction(1, 2)

    check_result(note, target_offset_mc, target_offset_beat)

    # One+ measure
    note = pd.DataFrame({
        'file_id': [1],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(1, 2)],
        'duration': [Fraction(3, 4)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 2
    target_offset_beat = Fraction(3, 4)

    check_result(note, target_offset_mc, target_offset_beat)

    # Many measures
    note = pd.DataFrame({
        'file_id': [1],
        'note_id': [0],
        'mc': [1],
        'onset': [Fraction(1, 2)],
        'duration': [Fraction(20)]
    }).set_index(['file_id', 'note_id'])
    target_offset_mc = 7
    target_offset_beat = Fraction(35, 2)

    check_result(note, target_offset_mc, target_offset_beat)


def test_get_notes_during_chord():
    chords = pd.DataFrame({
        'file_id': [0, 0, 1, 2, 2],
        'onset': [Fraction(3, 4),
                  Fraction(1, 2),
                  Fraction(0),
                  Fraction(1, 4),
                  Fraction(1, 2)],
        'chord_id': [0, 1, 0, 0, 1],
        'mc': [2, 4, 2, 3, 4],
        'extra': Fraction(2, 3)
    }).set_index(['file_id', 'chord_id'])
    offsets_chords_df = cu.add_chord_metrical_data(chords, removed_repeats_with_unreachable)

    # In order: before, tied_into, tied_both, inside, tied_out, after
    notes = pd.DataFrame({
        'file_id': [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
        'note_id': [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        'extra': Fraction(2, 3),
        'mc': [1, 1, 1, 2, 2, 4, 1, 1, 1, 3, 3, 4],
        'onset': [Fraction(0),
                  Fraction(0),
                  Fraction(0),
                  Fraction(3, 4),
                  Fraction(3, 4),
                  Fraction(1, 2),
                  Fraction(0),
                  Fraction(0),
                  Fraction(0),
                  Fraction(1, 4),
                  Fraction(1, 4),
                  Fraction(1, 2)],
        'duration': [Fraction(7, 4),
                     Fraction(15, 8),
                     Fraction(4),
                     Fraction(5, 4),
                     Fraction(6, 4),
                     Fraction(2),
                     Fraction(1),
                     Fraction(5, 4),
                     Fraction(7, 4),
                     Fraction(1, 2),
                     Fraction(3, 4),
                     Fraction(2)]
    }).set_index(['file_id', 'note_id'])
    offsets_notes_df = cu.add_note_offsets(notes, removed_repeats_with_unreachable)

    for chord_id, chord in offsets_chords_df.iloc[0::3].iterrows():
        notes = cu.get_notes_during_chord(chord, offsets_notes_df)

        notes_onsets = cu.get_notes_during_chord(chord, offsets_notes_df, onsets_only=True)
        assert notes_onsets.equals(notes.loc[notes.overlap.isin([pd.NA, 1])])

        assert set(notes.columns) == set(list(offsets_notes_df.columns) + ['overlap'])
        assert notes.index.names == offsets_notes_df.index.names
        if len(notes) > 0:
            assert notes.loc[:, offsets_notes_df.columns].equals(offsets_notes_df.loc[notes.index])
        assert notes.overlap.dtype == 'Int64'
        assert len(notes) == 4
        assert list(notes.overlap) == [-1, 0, pd.NA, 1]
        assert list(notes.index.get_level_values('note_id')) == list(range(1, 5))

    # Make sure no match is fine
    no_matches = pd.DataFrame({'file_id': [1, 2],
                               'chord_id': 0,
                               'mc': 7,
                               'onset': Fraction(1),
                               'mc_next': 8,
                               'onset_next': Fraction(1),
                               'duration': Fraction(1)}).set_index(['file_id', 'chord_id'])
    for chord_id, chord in no_matches.iterrows():
        notes = cu.get_notes_during_chord(chord, offsets_notes_df)

        assert len(notes) == 0, f'{chord}\n{notes}'
        assert notes.index.names == offsets_notes_df.index.names
        assert set(notes.columns) == set(list(offsets_notes_df.columns) + ['overlap'])
        assert notes.overlap.dtype == 'Int64'


def test_merge_ties():
    notes_list = []

    # TESTS:
    #  -an unmatched note
    #  -long chained match
    #  -doubly-matched disambiguated by voice and staff
    #  -two matches at the same time
    #  -gracenotes are skipped
    #  -un-ending tie
    notes_list.append(pd.DataFrame({
        'mc': [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8],
        'onset': Fraction(0),
        'duration': [Fraction(1)] * 5 + [Fraction(1, 2)] * 5 + [Fraction(0)],
        'offset_mc': [1, 2, 3, 4, 5, 5, 6, 6, 8, 9, 8],
        'offset_beat': [Fraction(0)] * 7 + [Fraction(1, 2)] + [Fraction(0)] * 3,
        'midi': 40,
        'staff': 2,
        'voice': [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        'gracenote': [None] * 10 + ['grace'],
        'tied': [1, 0, 0, 0, -1, 0, -1, pd.NA, 1, 0, -1]
    }))

    # TESTS:
    #  -midi is taken into account
    #  -offset beat is take into account
    #  -multiple identically-matched with voice and staff still works
    notes_list.append(pd.DataFrame({
        'mc': [0, 1, 1, 1, 1, 2, 2, 3, 4, 5],
        'onset': [Fraction(0)] * 7 + [Fraction(1, 2)] * 3,
        'duration': [Fraction(1)] * 6 + [Fraction(1, 2)] + [Fraction(1)] * 3,
        'offset_mc': [1, 2, 2, 2, 2, 3, 3, 4, 5, 6],
        'offset_beat': Fraction(0),
        'midi': [40] * 5 + [50] * 5,
        'staff': 0,
        'voice': 0,
        'gracenote': None,
        'tied': [1, -1, -1, -1, 1, -1, 1, -1, 1, -1]
    }))

    # TESTS:
    #  -ties beginning with tied=0
    #  -multiply-matched cases where neither matches staff or voice
    notes_list.append(pd.DataFrame({
        'mc': [0, 1, 1, 1, 2, 3, 3, 3],
        'onset': Fraction(0),
        'duration': Fraction(1),
        'offset_mc': [1, 2, 2, 2, 3, 4, 4, 4],
        'offset_beat': Fraction(0),
        'midi': 50,
        'staff': [0, 1, 1, 1, 0, 1, 1, 1],
        'voice': [0, 1, 1, 1, 0, 1, 1, 1],
        'gracenote': None,
        'tied': [1, 0, 0, 0, 0, -1, -1, -1]
    }))

    # Create notes in good format
    notes = pd.concat(
        notes_list, keys=list(range(len(notes_list))), axis=0, names=['file_id', 'note_id']
    )
    notes = notes.assign(extra=Fraction(3, 2))
    for column in ['mc', 'offset_mc', 'midi', 'staff', 'voice', 'tied']:
        notes.loc[:, column] = notes[column].astype('Int64')

    merged = cu.merge_ties(notes)

    # First, check form of result
    assert notes.equals(notes)
    assert merged.index.names == notes.index.names
    assert len(set(merged.columns) - set(notes.columns)) == 0
    assert len(set(notes.columns) - set(merged.columns)) == 0

    unchanged = list(set(notes.columns) - set(['offset_mc', 'offset_beat', 'duration', 'tied']))
    assert notes.loc[merged.index, unchanged].equals(merged.loc[:, unchanged])

    # Now, check accuracy
    merged_single = merged.loc[0]
    assert len(merged_single) == 5
    assert all(merged_single.index == [0, 4, 7, 8, 10])
    assert all(merged_single.offset_mc == [6, 5, 6, 9, 8])
    assert all(merged_single.offset_beat == [0, 0, Fraction(1, 2), 0, 0])
    assert all(merged_single.duration == [Fraction(5), Fraction(1), Fraction(1, 2), 1, 0])
    # Fix for pd.NA == pd.NA returns False
    assert all(merged_single.tied.fillna(100) == [100, -1, 100, 1, -1])

    # Now, check accuracy
    merged_single = merged.loc[1]
    assert len(merged_single) == 9
    assert all(merged_single.index == [0, 2, 3, 4, 5, 6, 7, 8, 9])
    assert all(merged_single.offset_mc == [2, 2, 2, 2, 3, 3, 4, 5, 6])
    assert all(merged_single.offset_beat == [0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert all(merged_single.duration == [2, 1, 1, 1, 1, Fraction(1, 2), 1, 1, 1])
    # Fix for pd.NA == pd.NA returns False
    assert all(merged_single.tied.fillna(100) == [100, -1, -1, 1, -1, 1, -1, 1, -1])

    # Now, check accuracy
    merged_single = merged.loc[2]
    assert len(merged_single) == 3
    assert all(merged_single.index == [0, 2, 3])
    assert all(merged_single.offset_mc == [4, 4, 2])
    assert all(merged_single.offset_beat == [0, 0, 0])
    assert all(merged_single.duration == [4, 3, 1])
    # Fix for pd.NA == pd.NA returns False
    assert all(merged_single.tied.fillna(100) == [100, 100, 0])
