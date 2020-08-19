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
    # First, check for offset information, and calculate it if necessary
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


def find_matching_tie(note: pd.Series = None, note_file_id: int = None,
                      note_note_id: int = None, note_midi: int = None, note_voice: int = None,
                      note_staff: int = None, note_offset_mc: int = None,
                      note_offset_beat: Fraction = None, note_duration: Fraction = None,
                      midi_masks: np.ndarray = None, prefiltered: bool = False,
                      tied_in_notes: pd.DataFrame = None, tied_in_notes_file_id: np.ndarray = None,
                      tied_in_notes_midi: np.ndarray = None,
                      tied_in_notes_voice: np.ndarray = None,
                      tied_in_notes_staff: np.ndarray = None,
                      tied_in_notes_mc: np.ndarray = None,
                      tied_in_notes_onset: np.ndarray = None) -> int:
    """
    Find the note which a given note is tied to. The matching note must have an onset beat
    and mc equal to the given note's offset_beat and offset_mc, as well as equal midi pitch.
    If multiple matching notes are found, they are disambiguated by using the notes' voice.

    By default, the matching note must also match the given note's id and section (the first
    two values of its multi-index tuple). If prefiltered is given as True, the given
    tied_in_notes DataFrame is assumed to be already filtered by id and section, and this
    requirement is not checked.

    Either note or all of the note_X parameters are required. Likewise, either tied_in_notes
    or all of the tied_in_notes_X parameters are required.

    Parameters
    ----------
    note : pd.Series
        Either this or all of the note_X parameters are required.
        A note that is tied out of. The function will return the note which this note is
        tied into. The series must have at least the columns:
            'midi' (int): The MIDI pitch of the note.
            'voice' (int): The voice of the note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time (and might potentially be
                the resulting tied note).
            'staff' (int): The staff of the note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time (and might potentially be
                the resulting tied note).
            'offset_beat' (Fraction): The offset beat of the note (see get_offsets).
            'offset_mc' (int): The offset 'mc' of the note (see get_offsets).
            'duration' (Fraction): The duration of the note, in whole notes. Used for
                warning printing.
        Additionally, if prefiltered is False, the note's name attribute must be a tuple
        where the first, second, and third values correspond to the note's id, section,
        and note_id values.

    note_file_id : int
        The file_id of the note.

    note_note_id : int
        The note_id of the note. Used for warning printing.

    note_midi : int
        The MIDI pitch of the note.

    note_voice : int
        The voice of the note. Used to disambiguate ties when multiple notes of the same
        pitch have the same onset time (and might potentially be the resulting tied note).

    note_staff : int
        The staff of the note. Used to disambiguate ties when multiple notes of the same
        pitch have the same onset time (and might potentially be the resulting tied note).

    note_offset_mc : int
        The offset 'mc' of the note (see get_offsets).

    note_offset_beat : Fraction
        The offset beat of the note (see get_offsets).

    note_duration : Fraction
        The duration of the note, in whole notes. Used for warning printing.

    midi_masks : np.ndarray
        An nd-array of precomputed midi pitch masks for the filtered tied_in_notes DataFrame.
        midi_masks[p] should be the mask for tied_in_notes.midi == p. This can be None, in which
        case this mask is calculated in this function. If the mask is not None, but the value
        in the pitch of the given note is None, the calculated mask is saved in the given
        midi_masks array.

    prefiltered : boolean
        False if the given tied_in_notes DataFrame has not yet been filtered by id and section.
        In this case, the note's name attribute must be a tuple whose first two values
        correspond to its id and section, and tied_in_notes must use a MultiIndex whose first
        two values correspond to its id and section. If True, the id and section values
        are not checked, and are not even required to be present.

    tied_in_notes : pd.DataFrame
        Either this or all of the tied_in_notes_X parameters are required.
        A DataFrame in which to search for the matching tied note of the given note.
        It must have at least the following columns:
            'midi' (int): The MIDI pitch of each note.
            'voice' (int): The voice of each note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time (and might potentially be
                the resulting tied note).
            'staff' (int): The staff of each note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time (and might potentially be
                the resulting tied note).
            'mc' (int): The 'measure count' index of the onset of each note.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of the given mc.
        Additionally, if prefiltered is False, its first two index columns must be:
            'id' (index, int): The piece id from which each note comes; and
            'section' (index, int): The section of the piece from which each note comes.

    tied_in_notes_file_id : np.ndarray(int)
        The piece id's of the searched notes.

    tied_in_notes_midi : np.ndarray(int)
        The MIDI pitches of the searched notes.

    tied_in_notes_voice : np.ndarray(int)
        The voices of the searched notes.

    tied_in_notes_staff : np.ndarray(int)
        The staffs of the searched notes.

    tied_in_notes_mc : np.ndarray(int)
        The onset 'mc's of the searched notes.

    tied_in_notes_onset : np.ndarray(Fraction)
        The onset time of each searched note, measured in whole notes since the beginning
        of its mc.

    Returns
    -------
    matched_note_index : int
        The index (from tied_in_notes) of the matching tied note to the given one, or None
        if none was found.

        If multiple matches are found in the basic search (across midi pitch, onset/offset beat
        and measure, as well as id and section if prefiltered is False), the matches are then
        filtered by voice. If one remains, it is returned as the match. If multiple remain,
        the first one (in the order of the given tied_in_notes) is returned as the match.
        If none remain after voice filtering, the first one returned by the basic search is
        returned.
    """
    # Save note fields into individual variables if they weren't given
    if None in [note_file_id, note_note_id, note_midi, note_voice, note_offset_mc,
                note_offset_beat, note_duration]:
        assert note is not None, "Either note or all note_x parameters are required."
        if note_file_id is None:
            note_file_id = note.name[0]
        if note_note_id is None:
            note_note_id = note.name[1]
        if note_midi is None:
            note_midi = note.midi
        if note_voice is None:
            note_voice = note.voice
        if note_offset_mc is None:
            note_offset_mc = note.offset_mc
        if note_offset_beat is None:
            note_offset_beat = note.offset_beat
        if note_duration is None:
            note_duration = note.duration

    # Save tied notes into numpy lists if they weren't given
    if (tied_in_notes_midi is None or tied_in_notes_mc is None or tied_in_notes_voice is None or
            tied_in_notes_onset is None or tied_in_notes_staff is None):

        assert tied_in_notes is not None, ("Either tied_in_notes or all tied_in_notes_x params "
                                           "(except _file_id) are required.")
        if tied_in_notes_midi is None:
            tied_in_notes_midi = tied_in_notes.midi.to_numpy()
        if tied_in_notes_mc is None:
            tied_in_notes_mc = tied_in_notes.mc.to_numpy()
        if tied_in_notes_voice is None:
            tied_in_notes_voice = tied_in_notes.voice.to_numpy()
        if tied_in_notes_staff is None:
            tied_in_notes_staff = tied_in_notes.staff.to_numpy()
        if tied_in_notes_onset is None:
            tied_in_notes_onset = tied_in_notes.onset.to_numpy()

    # Filter by file_id if not prefiltered
    if not prefiltered:
        if tied_in_notes_file_id is None:
            assert tied_in_notes is not None, ("Either tied_in_notes or tied_in_notes_id required "
                                               "if prefiltered is False.")
            if tied_in_notes_file_id is None:
                tied_in_notes_file_id = tied_in_notes.index.get_level_values('file_id').to_numpy()
        unfiltered_tied_in_notes_mask = tied_in_notes_file_id == note_file_id
        tied_in_notes_midi = tied_in_notes_midi[unfiltered_tied_in_notes_mask]
        tied_in_notes_mc = tied_in_notes_mc[unfiltered_tied_in_notes_mask]
        tied_in_notes_voice = tied_in_notes_voice[unfiltered_tied_in_notes_mask]
        tied_in_notes_staff = tied_in_notes_staff[unfiltered_tied_in_notes_mask]
        tied_in_notes_onset = tied_in_notes_onset[unfiltered_tied_in_notes_mask]

    # Filter by midi pitch, and save midi mask if desired
    if midi_masks is not None:
        # Save midi mask
        midi_mask = midi_masks[note_midi]
        if midi_mask is None:
            midi_mask = tied_in_notes_midi == note_midi
            midi_masks[note_midi] = midi_mask
    else:
        # Don't save
        midi_mask = tied_in_notes_midi == note_midi

    # Find matches, disregarding voice
    matching_notes_mask = np.logical_and.reduce((
        midi_mask,
        tied_in_notes_mc == note_offset_mc,
        tied_in_notes_onset == note_offset_beat
    ))

    if np.sum(matching_notes_mask) > 1:
        # More than 1 match -- filter by voice and staff
        voice_mask = np.logical_and.reduce((matching_notes_mask,
                                            tied_in_notes_voice == note_voice,
                                            tied_in_notes_staff == note_staff))

        # Replace matches if voice filtering was successful (or at least, didn't remove all matches)
        if np.any(voice_mask):
            matching_notes_mask = voice_mask

    # Error -- no match found
    if not np.any(matching_notes_mask):
        logging.warning(f"No matching tied note found for note index ({note_file_id}, "
                        f"{note_note_id}) and duration {note_duration}. Returning None.")
        return None

    # Error -- multiple matches found
    if np.sum(matching_notes_mask) > 1:
        logging.warning(f"Multiple matching tied notes found for note index ({note_file_id}, "
                        f"{note_note_id}) and duration {note_duration}. "
                        "Returning the first.")

    # Return the first note on success or matches > 1
    if not prefiltered:
        return np.argwhere(unfiltered_tied_in_notes_mask)[np.argmax(matching_notes_mask)]
    return np.argmax(matching_notes_mask)


def merge_ties(notes: pd.DataFrame, measures: pd.DataFrame = None) -> pd.DataFrame:
    """
    Return a new notes DataFrame, with tied notes removed and replaced by a single note with
    longer duration. If 'offset_beat' and 'offset_mc' columns are not in the given notes DataFrame,
    they will be calculated, so measure must be given. Grace notes are ignored during ties and
    are just returned as is.

    Parameters
    ----------
    notes : pd.DataFrame
        A pandas DataFrame containing the notes to be merged together. This should include at least
        the following columns:
            'id' (index, int): The piece id from which each note comes.
            'mc' (int): The 'measure count' index of the onset of each note.
            'onset' (Fraction): The onset time of each note, in whole notes, relative to the
                beginning of the given mc.
            'duration' (Fraction): The duration of each note, in whole notes. Notes whose
                duration goes beyond the end of their mc (e.g., after merging ties) are
                handled correctly.
            'midi' (int): The MIDI pitch of each note.
            'voice' (int): The voice of each note. Used to disambiguate ties when multiple
                notes of the same pitch have the same onset time.
            'gracenote' (string): What type of grace-note each note is, or pd.NA if it is not a
                gracenote. This is used because we ignore grace notes and simply return them as is
                during note merging.
            'tied' (int): The tied status of each note:
                pd.nan if the note is not tied.
                1 if the note is tied out of (i.e., it is an onset).
                -1 if the note is tied into (i.e., it is an offset).
                0 if the note is tied into and out of (i.e., it is neither an onset nor an offset).
        The following columns will be calculated with get_offsets if not present (so measure must
        be given):
            'offset_beat' (Fraction): The offset beat of each note (see get_offsets).
            'offset_mc' (int): The offset 'mc' of each note (see get_offsets).

    measures : pd.DataFrame
        Data about the measures in the corpus. Required if notes does not contain the columns
        'offset_beat' and 'offset_mc'. See get_offsets for more information.

    Returns
    -------
    merged_notes : pd.DataFrame
        A pandas DataFrame containing all of the notes from the input DataFrame, but with the
        merged notes removed and replaced by a single note, spanning their entire duration,
        with tied = 1.
    """
    # First, check for offset information, and calculate it if necessary
    if not all([column in notes.columns for column in ['offset_beat', 'offset_mc']]):
        assert measures is not None, ("measures must be given if offset_beat and offset_mc "
                                      "are not in notes")
        notes = add_note_offsets(notes, measures)

    # Tied in and out notes
    tied_out_mask = notes.tied == 1
    tied_out_notes = notes.loc[tied_out_mask & notes.gracenote.isna()]
    tied_in_notes = notes.loc[notes.tied.isin([-1, 0]) & notes.gracenote.isna()]

    # For all initial gracenote ties, add the notes they are tied into to the tied_out_notes
    to_add = [] # Add to tied_out_notes
    to_remove = [] # Remove from tied_in_notes
    for _, note in notes.loc[tied_out_mask & ~notes.gracenote.isna()].iterrows():
        tied_note_index = find_matching_tie(note=note, tied_in_notes=tied_in_notes)

        if tied_note_index is not None:
            tied_note_df = tied_in_notes.iloc[tied_note_index]
            to_remove.append(tied_note_df.index[0])
            if tied_note_df.tied[0] == 0:
                to_add.append(tied_note_df.index[0])

    # Move found notes from tied_in to tied_out
    tied_out_notes = tied_out_notes.append(notes.loc[to_add])
    tied_in_notes = tied_in_notes.drop(index=to_remove)

    # Get important columns into numpy arrays for MUCH faster processing
    tied_in_notes_file_id = tied_in_notes.index.get_level_values('file_id').to_numpy()
    tied_in_notes_midi = tied_in_notes.midi.to_numpy()
    tied_in_notes_mc = tied_in_notes.mc.to_numpy()
    tied_in_notes_voice = tied_in_notes.voice.to_numpy()
    tied_in_notes_staff = tied_in_notes.staff.to_numpy()
    tied_in_notes_offset_mc = tied_in_notes.offset_mc.to_numpy()
    tied_in_notes_onset = tied_in_notes.onset.to_numpy()
    tied_in_notes_duration = tied_in_notes.duration.to_numpy()
    tied_in_notes_offset_beat = tied_in_notes.offset_beat.to_numpy()
    tied_in_notes_tied = tied_in_notes.tied.to_numpy()

    tied_out_notes_file_id = tied_out_notes.index.get_level_values('file_id').to_numpy()
    tied_out_notes_note_id = tied_out_notes.index.get_level_values('note_id').to_numpy()
    tied_out_notes_midi = tied_out_notes.midi.to_numpy()
    tied_out_notes_voice = tied_out_notes.voice.to_numpy()
    tied_out_notes_staff = tied_out_notes.staff.to_numpy()
    tied_out_notes_offset_mc = tied_out_notes.offset_mc.to_numpy()
    tied_out_notes_duration = tied_out_notes.duration.to_numpy()
    tied_out_notes_offset_beat = tied_out_notes.offset_beat.to_numpy()

    # Some iteration tracking and other helper variables
    prev_index = -1
    max_midi = tied_out_notes.midi.max()

    # Loop through and fix the duration and offset every tied out note
    for iloc_idx, (note_file_id, note_note_id, note_midi, note_voice, note_staff) in (
            enumerate(zip(tied_out_notes_file_id, tied_out_notes_note_id,
                          tied_out_notes_midi, tied_out_notes_voice, tied_out_notes_staff))):

        # These are not in the iterator because they will be updated in the loop
        note_offset_mc = tied_out_notes_offset_mc[iloc_idx]
        note_offset_beat = tied_out_notes_offset_beat[iloc_idx]
        note_duration = tied_out_notes_duration[iloc_idx]

        # Pre-filter notes within (file_id) for speed
        if prev_index != note_file_id:
            prev_index = note_file_id
            tied_in_notes_mask = tied_in_notes_file_id == note_file_id
            midi_masks = np.full(max_midi + 1, None)

        # Add new notes until an end tie is reached (where tied == -1)
        while True:
            tied_note_index = find_matching_tie(
                note_midi=note_midi, note_voice=note_voice, note_offset_mc=note_offset_mc,
                note_offset_beat=note_offset_beat, note_duration=note_duration,
                note_file_id=note_file_id, note_note_id=note_note_id,
                note_staff=note_staff,
                tied_in_notes=None, prefiltered=True, midi_masks=midi_masks,
                tied_in_notes_midi=tied_in_notes_midi[tied_in_notes_mask],
                tied_in_notes_voice=tied_in_notes_voice[tied_in_notes_mask],
                tied_in_notes_mc=tied_in_notes_mc[tied_in_notes_mask],
                tied_in_notes_onset=tied_in_notes_onset[tied_in_notes_mask],
                tied_in_notes_staff=tied_in_notes_staff[tied_in_notes_mask]
            )

            # Error -- no matching tie found.
            if tied_note_index is None:
                break

            # Update duration and offset
            note_duration += tied_in_notes_duration[tied_in_notes_mask][tied_note_index]
            note_offset_mc = tied_in_notes_offset_mc[tied_in_notes_mask][tied_note_index]
            note_offset_beat = tied_in_notes_offset_beat[tied_in_notes_mask][tied_note_index]

            # Break if tie has ended
            if tied_in_notes_tied[tied_in_notes_mask][tied_note_index] == -1:
                break

        # Update arrays to final tied values
        tied_out_notes_duration[iloc_idx] = note_duration
        tied_out_notes_offset_mc[iloc_idx] = note_offset_mc
        tied_out_notes_offset_beat[iloc_idx] = note_offset_beat

    # Update dataframe from tracking arrays
    tied_out_notes.duration = tied_out_notes_duration
    tied_out_notes.offset_mc = tied_out_notes_offset_mc
    tied_out_notes.offset_beat = tied_out_notes_offset_beat

    # Return final results (unchanged notes and updated notes)
    new_notes = notes.copy()
    new_notes.loc[tied_out_notes.index] = tied_out_notes
    return pd.DataFrame(new_notes.loc[notes.tied.isna() | (notes.index.isin(tied_out_notes.index))])
