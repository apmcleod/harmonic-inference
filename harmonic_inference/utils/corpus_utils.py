"""Utility functions for working with the corpus data."""
from typing import List
from fractions import Fraction
import warnings
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

import harmonic_inference.utils.rhythmic_utils as ru


def remove_unmatched(dataframe: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows from the given dataframe which do not have an associated measure in the given
    measures dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DataFrame from which we will remove rows. Must have at least columns 'file_id'
        (int, index) and 'mc' (int).
    mesures : pd.DataFrame
        A DataFrame of valid measures. Must have at least columns 'file_id' (int, index)
        and 'mc' (int).

    Returns
    -------
    new_dataframe : pd.DataFrame
        A copy of the given dataframe, with any row which does not correspond to a measure
        from the measures DataFrame removed.
    """
    # List of columns that are indexes for the given dataframe, in order to re-index after merge
    index_list = list(dataframe.index.names)

    # Inner merge removes unmatched rows from the given dataframe
    merged = pd.merge(dataframe.loc[:, 'mc'].reset_index(), measures.loc[:, 'mc'], how='inner',
                      on=['file_id', 'mc'])
    merged_index = merged.set_index(index_list).index

    return dataframe.loc[merged_index].copy()


def remove_repeats(measures: pd.DataFrame, remove_unreachable: bool = True) -> pd.DataFrame:
    """
    Remove repeats from the given measures DataFrame.

    Parameters
    ----------
    measures : pd.DataFrame
        The measures data for the entire corpus.

    remove_unreachable : bool
        Remove rows for measures which are now unreachable.

    Returns
    -------
    measures : pd.DataFrame
        A copy of the given measures data frame with repeats removed.
    """
    measures = measures.copy()
    next_lengths = measures.next.apply(len)

    # Find potential last measures of each piece
    last_measures = (next_lengths == 0) | [-1 in n for n in measures.next] # Default case
    last_measures |= (np.roll(measures.index.get_level_values('file_id').to_numpy(), -1) !=
                      measures.index.get_level_values('file_id'))

    next_lengths[last_measures] = 0

    # Always hop to the latest measure that might come next
    measures.loc[next_lengths > 1, 'next'] = [
        max(next_mc) for next_mc in measures.loc[next_lengths > 1, 'next'].to_numpy()
    ]

    # Only 1 option
    measures.loc[next_lengths == 1, 'next'] = [
        next_mc[0] for next_mc in measures.loc[next_lengths == 1, 'next'].to_numpy()
    ]

    # Fix the last measures and typing
    measures.loc[last_measures, 'next'] = pd.NA
    measures.next = measures.next.astype('Int64')

    # Remove measures which are unreachable
    if remove_unreachable:
        # Start measures will not be in any next list, but they need to be saved as a special case
        start_measures = (np.roll(measures.index.get_level_values('file_id').to_numpy(), 1) !=
                        measures.index.get_level_values('file_id'))
        start_indices = list(measures.index.to_numpy()[start_measures])

        # Save to reset type later. Needs to be nullable for the merge to work
        mc_type = measures.mc.dtype
        measures.mc = measures.mc.astype('Int64')

        while True:
            # Merge each measure to the next measure
            merged = pd.merge(
                measures.reset_index(), measures.reset_index(), how='inner',
                left_on=['file_id', 'next'], right_on=['file_id', 'mc'], suffixes=['', '_next']
            ).set_index(['file_id', 'measure_id_next'])

            # Valid indexes have successfully been merged or are a start index
            idx = set(list(merged.index.to_numpy()) + start_indices)

            # Nothing will change: break
            if len(idx) == len(measures):
                break

            # Remove measures not in the index list from measures
            measures = measures.loc[list(idx)].sort_index()

        # Reset type
        measures.mc = measures.mc.astype(mc_type)

    return measures



def add_chord_metrical_data(chords: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'mc_next' (int), 'onset_next' (Fraction), and 'duration' (Fraction) columns to the given
    chords DataFrame, indicating the metrical position of the next chord as well as the duration
    for each chord. The last chord of each piece will be assigned 'mc_next' of the last measure
    of the piece, and 'onset_next' as the 'act_dur' of that measure. Its 'duration' will be
    calculated to that position.

    Parameters
    ----------
    chords : pd.DataFrame
        A DataFrame, indexed by 'file_id' (int), including at least 'mc' (int) and 'onset'
        (Fraction) columns.
    measures : pd.DataFrame
        A DataFrame, indexed by 'file_id' (int), including at least 'mc' (int) and 'act_dur'
        (Fraction) columns.

    Returns
    -------
    chords : pd.DataFrame
        A copy of the given chords DataFrame, with 'mc_next' (int), 'onset_next' (Fraction),
        and 'duration' (Fraction) columns added.
    """
    if isinstance(measures.iloc[0].next, list) or isinstance(measures.iloc[0].next, tuple):
        warnings.warn("Repeats have not been unrolled or removed. They will be unrolled here, but "
                     "not saved. Unrolling or removing repeats first is faster.")
        measures = remove_repeats(measures)

    # Fix for measure offsets
    full_merge = pd.merge(
        chords.reset_index(), measures, how='left', left_on=['file_id', 'mc'],
        right_on=['file_id', 'mc']
    )
    full_merge = full_merge.set_index(['file_id', 'chord_id'])
    chords = chords.assign(on_fixed=full_merge.onset - full_merge.offset)

    # In most cases, next is a simple shift
    chords = chords.assign(mc_next=chords.mc.shift(-1).astype('Int64'),
                           onset_next=chords.onset.shift(-1),
                           on_next_fixed=chords.on_fixed.shift(-1))

    # For the last chord of each piece, it is more complicated
    last_chords = chords.loc[
        chords.index.get_level_values('file_id') !=
        np.roll(chords.index.get_level_values('file_id'), -1)
    ]
    last_measures = measures.loc[measures.next.isnull()]
    last_merged = pd.merge(
        last_chords.reset_index(), last_measures,
        how='left', left_on=['file_id'], right_on=['file_id']
    )
    last_merged = last_merged.set_index(['file_id', 'chord_id'])

    # Last chord "next" pointer is end of last measure in piece
    chords.loc[last_chords.index, 'mc_next'] = last_merged.mc_y
    chords.loc[last_chords.index, 'onset_next'] = last_merged.act_dur + last_merged.offset
    chords.loc[last_chords.index, 'on_next_fixed'] = last_merged.act_dur

    # Naive duration calculation works if no measure change
    chords.loc[:, 'duration'] = chords.on_next_fixed - chords.on_fixed

    # Boolean mask for which chord durations to still check
    # Sometimes, the "mc_next" is unreachable due to removing repeats
    # The is_null call checks for those cases
    to_check = ~((chords.mc == chords.mc_next) | chords.mc.isnull())

    # Tracking location for the current mc (we will iteratively advance by measure)
    chords.loc[:, 'mc_current'] = chords.mc

    # Fix remaining durations iteratively by measure
    while to_check.any():
        # Merge remaining incorrect chords with the current measures
        full_merge = pd.merge(
            chords.loc[to_check].reset_index(), measures, how='left',
            left_on=['file_id', 'mc_current'], right_on=['file_id', 'mc']
        )
        full_merge = full_merge.set_index(['file_id', 'chord_id'])

        # Advance 1 measure in chords
        chords.loc[to_check, ['duration', 'mc_current']] = list(zip(
            *[full_merge.duration + full_merge.act_dur, full_merge.next]
        ))

        # Update to_check mask
        to_check.loc[to_check] = ~((full_merge.next == full_merge.mc_next) |
                                   full_merge.next.isnull())

    return chords.drop(['mc_current', 'on_fixed', 'on_next_fixed'], axis='columns')


def add_note_offsets(notes: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns 'offset_mc' (int) and 'offset_beat' (Fraction) to the given notes df denoting
    the offset position of each note.

    Parameters
    ----------
    notes : pd.DataFrame
        The notes whose offset positions we want. Must have at least these columns:
            'file_id' (index, int): The file id from which this note comes.
            'mc' (int): The 'measure count' index of the onset of each note. This is used
                to index into the given measures DataFrame.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of the given mc.
            'duration' (Fraction): The duration of each note, in whole notes.

    measures : pd.DataFrame
        A DataFrame containing information about each measure. Must have at least these columns:
            'file_id' (index, int): The file id from which this measure comes.
            'mc' (index, int): The 'measure count' index of this measure. Used by notes to index
                into this DataFrame.
            'act_dur' (Fraction): The duration of this measure, in whole notes. Note that this
                can be different from 'timesig', because of, e.g., partial measures near repeats.
            'next' (tuple(int) or int): The 'mc' of the measure that follows this one. This may
                contain multiple 'mc's in the case of a repeat, but it is recommended to either
                unroll or eliminate repeats before running get_offsets, which will result in
                only ints or Nones in this column. In the case of a longer list, the last
                'mc' in the list (measures['next'][-1]) is treated as the next mc and a warning is
                printed. This functionality is similar to eliminating repeats, although the
                underlying measures DataFrame is not changed.
            'offset' (Fraction): The beat, in whole notes, of the beginning of this measure.

    Returns
    -------
    notes_with_offsets : pd.DataFrame
        A copy of the given notes DataFrame with two additional columns: 'offset_mc' (int) and
        'offset_beat' (Fraction) denoting the offset position of each note.
    """
    if isinstance(measures.iloc[0].next, list) or isinstance(measures.iloc[0].next, tuple):
        warnings.warn("Repeats have not been unrolled or removed. They will be unrolled here, but "
                     "not saved. Unrolling or removing repeats first is faster.")
        measures = remove_repeats(measures)

    # Keep only the columns we want in measures for simplicity
    measures = measures.loc[:, ['mc', 'act_dur', 'next', 'offset']]
    measures.loc[:, 'end'] = measures.act_dur + measures.offset # For faster computation

    # Merge "next" to also have next_offset to potentially skip a loop iteration
    measures = pd.merge(measures.reset_index(), measures, how='left', left_on=['file_id', 'next'],
                        right_on=['file_id', 'mc'], suffixes=('', '_next'))
    measures = measures.set_index(['file_id', 'measure_id'])

    # Join notes to their onset measures. Computation will be performed on this combined df
    note_measures = pd.merge(
        notes.loc[:, ['mc', 'onset', 'duration']].reset_index(), measures, how='left',
        on=['file_id', 'mc']
    )
    note_measures = note_measures.set_index(['file_id', 'note_id'])

    # Default offset position calculation
    note_measures = note_measures.assign(
        offset_mc=note_measures.mc, offset_beat=note_measures.onset + note_measures.duration
    )

    # Find which notes are in the last measure of their piece
    last_measures = note_measures.next.isnull()

    # Fix offsets exactly at end of measure
    at_end = (note_measures.offset_beat == note_measures.end) & ~last_measures
    note_measures.loc[at_end, ['offset_mc', 'offset_beat']] = (
        note_measures.loc[at_end, ['next', 'offset_next']].to_numpy()
    )
    note_measures.offset_mc = note_measures.offset_mc.astype('Int64')

    # Find which offsets go beyond the end of their current measure
    past_end = (
        (note_measures.offset_beat > note_measures.end) & ~last_measures & ~at_end
    ).to_numpy()
    to_change_note_measures = note_measures.loc[past_end].copy()

    # Loop through, fixing those notes that still go beyond the end of their measure
    while len(to_change_note_measures) > 0:
        # Update offset positions to the next measure
        note_measures.loc[past_end, 'offset_beat'] = (
            to_change_note_measures.offset_beat - to_change_note_measures.act_dur -
            to_change_note_measures.offset + to_change_note_measures.offset_next
        )
        note_measures.loc[past_end, 'offset_mc'] = to_change_note_measures.next

        # Updated measure info for new "note measures"
        changed_note_measures = pd.merge(
            note_measures.loc[past_end, ['offset_beat', 'offset_mc']].reset_index(),
            measures, how='left', left_on=['file_id', 'offset_mc'], right_on=['file_id', 'mc'])
        changed_note_measures = changed_note_measures.set_index(['file_id', 'note_id'])

        # Update last measures with new measures
        last_measures = changed_note_measures.next.isnull()

        # Fix for notes exactly at measure ends
        at_end = (
            (changed_note_measures.offset_beat == changed_note_measures.end) & ~last_measures
        )
        note_measures.loc[at_end.loc[at_end].index, ['offset_mc', 'offset_beat']] = (
            changed_note_measures.loc[at_end, ['next', 'offset_next']].to_numpy()
        )

        # Check for any notes which still go beyond the end of a measure
        changed_past_end = (
            (changed_note_measures.offset_beat > changed_note_measures.end)
            & ~last_measures & ~at_end
        ).to_numpy()
        past_end[past_end] = changed_past_end

        to_change_note_measures = changed_note_measures.loc[changed_past_end].copy()

    notes = notes.assign(offset_mc=note_measures.offset_mc, offset_beat=note_measures.offset_beat)
    notes.offset_mc = notes.offset_mc.astype('Int64')
    return notes



def get_notes_during_chord(chord: pd.Series, notes: pd.DataFrame,
                           onsets_only: bool = False) -> pd.DataFrame:
    """
    Get all of the notes that occur during the given chord.

    Parameters
    ----------
    chord : pd.Series
        The chord whose notes to return.

    notes : pd.DataFrame
        The possible notes.

    Returns
    -------
    selected_notes : pd.DataFrame
        A DataFrame containing the notes that occur during the gievn chord, with an
        additional column 'overlap', which is:
            NA  if the note occurs entirely within the given chord.
            -1  if the note's onset is in a previous chord.
            1   if the note's offset is in a following chord.
            0   if both -1 and 1 apply.
    """
    # First, check for offset information
    assert all([column in notes.columns for column in ['offset_beat', 'offset_mc']]), (
        "Notes must contain offset information. Run `notes = corpus_utils.add_note_offsets` first."
    )

    # Search only through notes in the correct file
    if chord.name[0] in notes.index:
        file_notes = notes.loc[chord.name[0]]
    else:
        file_notes = notes.loc[~notes.mc.isin(notes.mc)]

    # Get note and chord onset and offset bounds

    # Workaround to get numpy array of tuples: www.stackoverflow.com/questions/46569100
    note_onsets = np.empty((len(file_notes),), dtype=object)
    note_offsets = np.empty((len(file_notes),), dtype=object)
    chord_onset = np.empty((1,), dtype=object)
    chord_offset = np.empty((1,), dtype=object)

    note_onsets[:] = list(zip(file_notes.mc, file_notes.onset))
    note_offsets[:] = list(zip(file_notes.offset_mc, file_notes.offset_beat))

    chord_onset[0] = (chord.mc, chord.onset)
    chord_offset[0] = (chord.mc_next, chord.onset_next)

    # Get notes that overlap the chord's range
    selection = (note_onsets < chord_offset) & (note_offsets > chord_onset)
    if onsets_only:
        selection &= note_onsets >= chord_onset

    selected_notes = file_notes.loc[selection]

    # Update note onset/offset position to only selected notes
    note_onsets = note_onsets[selection]
    note_offsets = note_offsets[selection]

    # Add overlap column
    ties_in = note_onsets < chord_onset
    ties_out = note_offsets > chord_offset
    ties_both = ties_in & ties_out

    overlap = np.array([pd.NA] * len(ties_in))
    overlap[ties_in] = -1
    overlap[ties_out] = 1
    overlap[ties_both] = 0

    # Add column
    selected_notes = selected_notes.assign(overlap=overlap)
    selected_notes.loc[:, 'overlap'] = selected_notes.overlap.astype('Int64')

    # Add file_id key back into notes
    if 'file_id' not in selected_notes.index.names:
        selected_notes = pd.concat([selected_notes], keys=[selected_notes.index.name],
                                   names=['file_id'])
        selected_notes.index.set_levels([chord.name[0]], level='file_id', inplace=True)

    return selected_notes


def merge_ties(notes: pd.DataFrame) -> pd.DataFrame:
    """
    Return a new notes DataFrame, with tied notes removed and replaced by a single note with
    longer duration. Grace notes are ignored during ties and are returned as is.

    Parameters
    ----------
    notes : pd.DataFrame
        A pandas DataFrame containing the notes to be merged together. This should include at least
        the following columns:
            'file_id' (index, int): The piece id from which each note comes.
            'note_id' (index, int): The index of each note.
            'mc' (int): The 'measure count' index of the onset of each note.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of its measure.
            'duration' (Fraction): The duration of each note, in whole notes.
            'midi' (int): The MIDI pitch of each note.
            'voice' (int): The voice of each note. Used to disambiguate ties when multiple
                notes of the same midi pitch have the same onset time.
            'staff' (int): The staff of each note. Used to disambiguate ties when multiple
                notes of the same midi pitch have the same onset time.
            'gracenote' (string): What type of grace-note each note is, or pd.NA if it is not a
                gracenote. This is used because we ignore grace notes and simply return them as is
                during note merging.
            'tied' (int): The tied status of each note:
                pd.NA if the note is not tied.
                1 if the note is tied out of (i.e., it is an onset).
                -1 if the note is tied into (i.e., it is an offset).
                0 if the note is tied into and out of (i.e., it is neither an onset nor an offset).
            'offset_beat' (Fraction): The offset beat of each note.
            'offset_mc' (int): The offset 'mc' of each note.

    Returns
    -------
    merged_notes : pd.DataFrame
        A pandas DataFrame containing all of the notes from the input DataFrame, but with
        merged notes removed and replaced by a single note, spanning their entire duration with
        the correct offset_beat and offset_mc.
        The resulting note's 'tied' value depends on whether it was matched.
        - A 1 which eventually matches a -1 will have tied=pd.NA. Otherwise, it will have tied=1.
          Any interveing tied=0 notes and the ending tied=-1 note will be removed.
        - A non-matched 0 which eventually reaches a -1 will have tied=-1. Otherwise, it will have
          tied=0. Any intervening tied=0 notes and the ending tied=-1 note will be removed.
        - A non-matched tied=-1 will have tied=-1.
        Gracenotes are returned as is.
    """
    # First, check for offset information
    assert all([column in notes.columns for column in ['offset_beat', 'offset_mc']]), (
        "Notes must contain offset information. Run `notes = corpus_utils.add_note_offsets` first."
    )

    # Split notes into those that will change and those that will not
    changing_notes_mask = notes.gracenote.isnull() & ~notes.tied.isnull()
    changing_notes = notes.loc[changing_notes_mask]
    unchanging_notes = notes.loc[~changing_notes_mask].copy()

    # Tracking dfs for tied out and in notes. These will be kept up to date while iterating
    tied_out_notes = changing_notes.loc[changing_notes.tied == 1].copy()
    tied_in_notes = changing_notes.loc[changing_notes.tied.isin([0, -1])].copy()

    # This will track all notes that are finished, or will no longer be matched ever
    finished_notes_dfs = []

    # Merge notes beginning with a tied=1 note iteratively
    merged = pd.merge(tied_out_notes.reset_index(), tied_in_notes.reset_index(), how='inner',
                      left_on=['file_id', 'midi', 'offset_mc', 'offset_beat'],
                      right_on=['file_id', 'midi', 'mc', 'onset'],
                      suffixes=('_out', '_in'))

    changes = True
    while changes and len(merged) > 0:
        changes = False

        # Get views of merged df using both tied_out and tied_out indexing
        merged_out_indexed = merged.set_index(['file_id', 'note_id_out'])
        merged_in_indexed = merged.set_index(['file_id', 'note_id_in'])

        # Find unmatched tied_out_notes, remove them from tied_out_notes, and add them to finished
        unmatched_mask = ~tied_out_notes.index.isin(merged_out_indexed.index)
        finished = tied_out_notes.loc[unmatched_mask].copy()
        if len(finished) > 0:
            finished_notes_dfs.append(finished)
            tied_out_notes = tied_out_notes.drop(finished.index)

        # Get new duration, tied values
        merged.loc[:, 'new_dur'] = merged.duration_out + merged.duration_in
        merged.loc[:, 'new_tied'] = pd.NA
        merged.loc[merged.tied_in == 0, 'new_tied'] = 1

        # Find pairs of notes where each is only in merged_notes once
        single_match_out_mask = merged_out_indexed.index.isin(
            merged_out_indexed.loc[merged_out_indexed.index.value_counts() == 1].index
        )
        single_match_in_mask = merged_in_indexed.index.isin(
            merged_in_indexed.loc[merged_in_indexed.index.value_counts() == 1].index
        )
        single_match_both = single_match_out_mask & single_match_in_mask

        # Update global tied_notes_out df with new matched values
        tied_out_notes.loc[merged_out_indexed.index[single_match_both],
                           ['offset_mc', 'offset_beat', 'duration', 'tied']] = (
            merged.loc[single_match_both,
                       ['offset_mc_in', 'offset_beat_in', 'new_dur', 'new_tied']].values
        )

        # Detect updates -- if anything has changed we should keep iterating
        if any(single_match_both):
            changes = True

        # Move finished notes from tied_out into finished
        finished = tied_out_notes.loc[tied_out_notes.tied.isnull()].copy()
        if len(finished) > 0:
            finished_notes_dfs.append(finished)
            tied_out_notes = tied_out_notes.drop(finished.index)

        # Remove any used notes from tied_in_notes
        tied_in_notes = tied_in_notes.drop(merged_in_indexed.index[single_match_both])

        # Re-match doubly-matched notes using staff and voice
        doubly_matched = ~single_match_both

        if any(doubly_matched):
            merged_doubly = merged.loc[doubly_matched]
            merged_more = merged_doubly.loc[(merged_doubly.staff_out == merged_doubly.staff_in) &
                                            (merged_doubly.voice_out == merged_doubly.voice_in)]

            # Naively drop notes that are still duplicates -- there is no other way to disambiguate
            merged_more = merged_more.drop_duplicates(subset=['file_id', 'note_id_out'])
            merged_more = merged_more.drop_duplicates(subset=['file_id', 'note_id_in'])
            # TODO: WARN

            # Get views of multiply-matched df using both out and in indexing
            merged_more_out = merged_more.set_index(['file_id', 'note_id_out'])
            merged_more_in = merged_more.set_index(['file_id', 'note_id_in'])

            # Update global tied_out_notes df with new values
            tied_out_notes.loc[merged_more_out.index,
                               ['offset_mc', 'offset_beat', 'duration', 'tied']] = (
                merged_more_out.loc[
                    :, ['offset_mc_in', 'offset_beat_in', 'new_dur', 'new_tied']
                ].values
            )

            # If anything has changed, we should iterate again
            if any(merged_more):
                changes = True

            # Move finished notes from tied_out into finished
            finished = tied_out_notes.loc[tied_out_notes.tied.isnull()].copy()
            if len(finished) > 0:
                finished_notes_dfs.append(finished)
                tied_out_notes = tied_out_notes.drop(finished.index)

            # Remove now-matched in notes
            tied_in_notes = tied_in_notes.drop(merged_more_in.index)

        # Re-merge for next iteration
        merged = pd.merge(tied_out_notes.reset_index(), tied_in_notes.reset_index(), how='inner',
                          left_on=['file_id', 'midi', 'offset_mc', 'offset_beat'],
                          right_on=['file_id', 'midi', 'mc', 'onset'],
                          suffixes=('_out', '_in'))

    finished_df = pd.concat(finished_notes_dfs)
    return pd.concat([unchanging_notes, tied_out_notes, tied_in_notes, finished_df]).sort_index()