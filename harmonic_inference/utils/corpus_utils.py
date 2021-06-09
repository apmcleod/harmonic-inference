"""Utility functions for working with the corpus DataFrames."""
import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from harmonic_inference.data.corpus_constants import (
    CHORD_ONSET_BEAT,
    MEASURE_OFFSET,
    NOTE_ONSET_BEAT,
)


def remove_unmatched(dataframe: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows from the given dataframe which do not have an associated measure in the given
    measures dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DataFrame from which we will remove rows. Must have at least columns 'file_id'
        (int, index) and 'mc' (int).
    measures : pd.DataFrame
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
    merged = pd.merge(
        dataframe.loc[:, "mc"].reset_index(),
        measures.loc[:, "mc"],
        how="inner",
        on=["file_id", "mc"],
    )
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
    next_lengths = measures["next"].apply(len)

    # Find potential last measures of each piece
    last_measures = (next_lengths == 0) | [-1 in n for n in measures["next"]]  # Default case
    last_measures |= np.roll(
        measures.index.get_level_values("file_id").to_numpy(), -1
    ) != measures.index.get_level_values("file_id")

    next_lengths[last_measures] = 0

    # Always hop to the latest measure that might come next
    measures.loc[next_lengths > 1, "next"] = [
        max(next_mc) for next_mc in measures.loc[next_lengths > 1, "next"].to_numpy()
    ]

    # Only 1 option
    measures.loc[next_lengths == 1, "next"] = [
        next_mc[0] for next_mc in measures.loc[next_lengths == 1, "next"].to_numpy()
    ]

    # Fix the last measures and typing
    measures.loc[last_measures, "next"] = pd.NA
    measures["next"] = measures["next"].astype("Int64")

    # Remove measures which are unreachable
    if remove_unreachable:
        # Start measures will not be in any next list, but they need to be saved as a special case
        start_measures = np.roll(
            measures.index.get_level_values("file_id").to_numpy(), 1
        ) != measures.index.get_level_values("file_id")
        start_indices = list(measures.index.to_numpy()[start_measures])

        # Save to reset type later. Needs to be nullable for the merge to work
        mc_type = measures["mc"].dtype
        measures["mc"] = measures["mc"].astype("Int64")

        while True:
            # Merge each measure to the next measure
            merged = pd.merge(
                measures.reset_index(),
                measures.reset_index(),
                how="inner",
                left_on=["file_id", "next"],
                right_on=["file_id", "mc"],
                suffixes=["", "_next"],
            ).set_index(["file_id", "measure_id_next"])

            # Valid indexes have successfully been merged or are a start index
            idx = set(list(merged.index.to_numpy()) + start_indices)

            # Nothing will change: break
            if len(idx) == len(measures):
                break

            # Remove measures not in the index list from measures
            measures = measures.loc[list(idx)].sort_index()

        # Reset type
        measures["mc"] = measures["mc"].astype(mc_type)

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
        warnings.warn(
            "Repeats have not been unrolled or removed. They will be unrolled here, but "
            "not saved. Unrolling or removing repeats first is faster."
        )
        measures = remove_repeats(measures)

    # Fix for measure offsets
    full_merge = pd.merge(
        chords.reset_index(),
        measures,
        how="left",
        left_on=["file_id", "mc"],
        right_on=["file_id", "mc"],
    )
    full_merge = full_merge.set_index(["file_id", "chord_id"])
    chords = chords.assign(on_fixed=full_merge[CHORD_ONSET_BEAT] - full_merge[MEASURE_OFFSET])

    # In most cases, next is a simple shift
    chords = chords.assign(
        mc_next=chords["mc"].shift(-1).astype("Int64"),
        onset_next=chords[CHORD_ONSET_BEAT].shift(-1),
        on_next_fixed=chords["on_fixed"].shift(-1),
    )

    # For the last chord of each piece, it is more complicated
    last_chords = chords.loc[
        chords.index.get_level_values("file_id")
        != np.roll(chords.index.get_level_values("file_id"), -1)
    ]
    last_measures = measures.loc[measures.next.isnull()]
    last_merged = pd.merge(
        last_chords.reset_index(),
        last_measures,
        how="left",
        left_on=["file_id"],
        right_on=["file_id"],
    )
    last_merged = last_merged.set_index(["file_id", "chord_id"])

    # Last chord "next" pointer is end of last measure in piece
    chords.loc[last_chords.index, "mc_next"] = last_merged["mc_y"]
    chords.loc[last_chords.index, f"{CHORD_ONSET_BEAT}_next"] = (
        last_merged["act_dur"] + last_merged[MEASURE_OFFSET]
    )
    chords.loc[last_chords.index, "on_next_fixed"] = last_merged["act_dur"]

    # Naive duration calculation works if no measure change
    chords.loc[:, "duration"] = chords["on_next_fixed"] - chords["on_fixed"]

    # Boolean mask for which chord durations to still check
    # Sometimes, the "mc_next" is unreachable due to removing repeats
    # The is_null call checks for those cases
    to_check = ~((chords["mc"] == chords["mc_next"]) | chords["mc"].isnull())

    # Tracking location for the current mc (we will iteratively advance by measure)
    chords.loc[:, "mc_current"] = chords["mc"]

    # Fix remaining durations iteratively by measure
    while to_check.any():
        # Merge remaining incorrect chords with the current measures
        full_merge = pd.merge(
            chords.loc[to_check].reset_index(),
            measures,
            how="left",
            left_on=["file_id", "mc_current"],
            right_on=["file_id", "mc"],
        )
        full_merge = full_merge.set_index(["file_id", "chord_id"])

        # Advance 1 measure in chords
        chords.loc[to_check, ["duration", "mc_current"]] = list(
            zip(*[full_merge["duration"] + full_merge["act_dur"], full_merge["next"]])
        )

        # Update to_check mask
        to_check.loc[to_check] = ~(
            (full_merge["next"] == full_merge["mc_next"]) | full_merge["next"].isnull()
        )

    return chords.drop(["mc_current", "on_fixed", "on_next_fixed"], axis="columns")


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
    if isinstance(measures.iloc[0]["next"], list) or isinstance(measures.iloc[0]["next"], tuple):
        warnings.warn(
            "Repeats have not been unrolled or removed. They will be unrolled here, but "
            "not saved. Unrolling or removing repeats first is faster."
        )
        measures = remove_repeats(measures)

    # Keep only the columns we want in measures for simplicity
    measures = measures.loc[:, ["mc", "act_dur", "next", MEASURE_OFFSET]]
    measures.loc[:, "end"] = (
        measures["act_dur"] + measures[MEASURE_OFFSET]
    )  # For faster computation

    # Merge "next" to also have next_offset to potentially skip a loop iteration
    measures = pd.merge(
        measures.reset_index(),
        measures,
        how="left",
        left_on=["file_id", "next"],
        right_on=["file_id", "mc"],
        suffixes=("", "_next"),
    )
    measures = measures.set_index(["file_id", "measure_id"])

    # Join notes to their onset measures. Computation will be performed on this combined df
    note_measures = pd.merge(
        notes.loc[:, ["mc", NOTE_ONSET_BEAT, "duration"]].reset_index(),
        measures,
        how="left",
        on=["file_id", "mc"],
    )
    note_measures = note_measures.set_index(["file_id", "note_id"])

    # Default offset position calculation
    note_measures = note_measures.assign(
        offset_mc=note_measures["mc"],
        offset_beat=note_measures[NOTE_ONSET_BEAT] + note_measures["duration"],
    )

    # Find which notes are in the last measure of their piece
    last_measures = note_measures["next"].isnull()

    # Fix offsets exactly at end of measure
    at_end = (note_measures["offset_beat"] == note_measures["end"]) & ~last_measures
    note_measures.loc[at_end, ["offset_mc", "offset_beat"]] = note_measures.loc[
        at_end, ["next", f"{MEASURE_OFFSET}_next"]
    ].to_numpy()
    note_measures["offset_mc"] = note_measures["offset_mc"].astype("Int64")

    # Find which offsets go beyond the end of their current measure
    past_end = (
        (note_measures["offset_beat"] > note_measures["end"]) & ~last_measures & ~at_end
    ).to_numpy()
    to_change_note_measures = note_measures.loc[past_end].copy()

    # Loop through, fixing those notes that still go beyond the end of their measure
    while len(to_change_note_measures) > 0:
        # Update offset positions to the next measure
        note_measures.loc[past_end, "offset_beat"] = (
            to_change_note_measures["offset_beat"]
            - to_change_note_measures["act_dur"]
            - to_change_note_measures[MEASURE_OFFSET]
            + to_change_note_measures[f"{MEASURE_OFFSET}_next"]
        )
        note_measures.loc[past_end, "offset_mc"] = to_change_note_measures["next"]

        # Updated measure info for new "note measures"
        changed_note_measures = pd.merge(
            note_measures.loc[past_end, ["offset_beat", "offset_mc"]].reset_index(),
            measures,
            how="left",
            left_on=["file_id", "offset_mc"],
            right_on=["file_id", "mc"],
        )
        changed_note_measures = changed_note_measures.set_index(["file_id", "note_id"])

        # Update last measures with new measures
        last_measures = changed_note_measures["next"].isnull()

        # Fix for notes exactly at measure ends
        at_end = (
            changed_note_measures["offset_beat"] == changed_note_measures["end"]
        ) & ~last_measures
        note_measures.loc[
            at_end.loc[at_end].index, ["offset_mc", "offset_beat"]
        ] = changed_note_measures.loc[at_end, ["next", f"{MEASURE_OFFSET}_next"]].to_numpy()

        # Check for any notes which still go beyond the end of a measure
        changed_past_end = (
            (changed_note_measures["offset_beat"] > changed_note_measures["end"])
            & ~last_measures
            & ~at_end
        ).to_numpy()
        past_end[past_end] = changed_past_end

        to_change_note_measures = changed_note_measures.loc[changed_past_end].copy()

    notes = notes.assign(
        offset_mc=note_measures["offset_mc"], offset_beat=note_measures["offset_beat"]
    )
    notes["offset_mc"] = notes["offset_mc"].astype("Int64")
    return notes


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
            'onset_mc' (Fraction): The onset time of each note, in whole notes, relative to the
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
    assert all(
        [column in notes.columns for column in ["offset_beat", "offset_mc"]]
    ), "Notes must contain offset information. Run `notes = corpus_utils.add_note_offsets` first."

    def repopulate_tied_out_notes(
        tied_in_notes: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Repopulate tied_out_notes from the given tied_in_notes. Any note where tied==0 can become
        tied_out. This function first merges those tied=0 notes with all tied_in notes (including
        those tied==0 notes). Then, notes which are only tied out, and not tied in are moved to
        become new tied_out_notes.

        Parameters
        ----------
        tied_in_notes : pd.DataFrame
            The tied_in_notes. We will draw new tied_out_notes from this dataframe.

        Returns
        -------
        tied_out_notes : pd.DataFrame
            A dataframe containing those notes that have become tied_out.

        tied_in_notes : pd.DataFrame
            The copy of the given tied_in_notes dataframe but with those notes that have become
            tied out removed.
        """
        in_merged = pd.merge(
            tied_in_notes.loc[tied_in_notes["tied"] == 0].reset_index(),
            tied_in_notes.reset_index(),
            how="inner",
            left_on=["file_id", "midi", "offset_mc", "offset_beat"],
            right_on=["file_id", "midi", "mc", NOTE_ONSET_BEAT],
            suffixes=("_out", "_in"),
        )
        in_merged_out_indexed = in_merged.set_index(["file_id", "note_id_out"])
        in_merged_in_indexed = in_merged.set_index(["file_id", "note_id_in"])

        # Move notes that are only tied out into the tied_out_notes df
        to_move = list(set(in_merged_out_indexed.index) - set(in_merged_in_indexed.index))
        tied_out_notes = tied_in_notes.loc[to_move].copy()
        tied_in_notes = tied_in_notes.drop(to_move)

        return tied_out_notes, tied_in_notes

    def add_new_tied_values(merged: pd.DataFrame) -> None:
        """
        Add new_dur (Fraction) and new_tied (Int64) columns in place into the given df,
        with the correct values:
            - new_dur = duration_out + duration_in
            - new_tied = tied_out + tied_in if either is 0. Otherwise, pd.NA

        Parameters
        ----------
        merged : pd.DataFrame
            The dataframe to which we will add the new columns. Must have duration_out (Fraction),
            duration_in (Fraction), tied_in (Int64), and tied_out (Int64) columns.
        """
        if len(merged) > 0:
            merged.loc[:, "new_dur"] = merged["duration_out"] + merged["duration_in"]
            merged.loc[:, "new_tied"] = pd.NA
            merged.loc[(merged["tied_out"] == 0) | (merged["tied_in"] == 0), "new_tied"] = (
                merged["tied_in"] + merged["tied_out"]
            )

    def remove_finished(
        tied_out_notes: pd.DataFrame, finished_mask: List, finished_notes_dfs: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Remove the finished notes (those where the given mask is True) from the given
        tied_out_notes dataframe. If any were removed, append the removed ones (as a dataframe)
        to the given finished_notes_dfs list. Return the resulting tied_out_notes dataframe.

        Parameters
        ----------
        tied_out_notes : pd.DataFrame
            A dataframe of notes. Some of them may be removed.
        finished_mask : List
            A mask, equal in length to tied_out_notes, marking True those rows which should be
            removed.
        finished_notes_dfs : List[pd.DataFrame]
            A List in which to add the removed notes (as a DataFrame).

        Returns
        -------
        tied_out_notes : pd.DataFrame
            The given tied_out_notes dataframe with the notes indicated by the finished_mask
            removed.
        """
        if any(finished_mask):
            finished = tied_out_notes.loc[finished_mask]
            finished_notes_dfs.append(finished)
            tied_out_notes = tied_out_notes.drop(finished.index)

        return tied_out_notes

    def get_out_and_in_views(merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get views of the given merged dataframe from the perspective of both tied_out_notes and
        tied_in_notes.

        Parameters
        ----------
        merged : pd.DataFrame
            A merged dataframe between tied_out_notes and tied_in_notes. Should have columns
            file_id (int), note_id_out (int), and note_id_in (int).

        Returns
        -------
        merged_out : pd.DataFrame
            The given merged dataframe, re-indexed with (file_id, note_id_out) as the index.
        merged_in : pd.DataFrame
            The given merged dataframe, re-indexed with (file_id, note_id_in) as the index.
        """
        merged_out = merged.set_index(["file_id", "note_id_out"])
        merged_in = merged.set_index(["file_id", "note_id_in"])
        return merged_out, merged_in

    def naive_duplicate_drop(merged: pd.DataFrame) -> pd.DataFrame:
        """
        Naively drop duplicate in and out notes from the given merged dataframe. It is naive
        because no checks are performed on the duplicate notes to decide which to keep. The
        first of each index is arbitrarily kept.

        Parameters
        ----------
        merged : pd.DataFrame
            A dataframe containing merged notes, with at least columns file_id (int),
            note_id_in (int), and note_id_out (int).

        Returns
        -------
        merged : pd.DataFrame
            The given dataframe, but with duplicate notes in and out removed.
        """
        merged_out_dups = merged.set_index(["file_id", "note_id_out"]).index.duplicated(keep=False)
        cols_to_log = [
            "file_id",
            "note_id_out",
            "offset_mc_out",
            "offset_beat_out",
            "staff_out",
            "voice_out",
            "note_id_in",
            "staff_in",
            "voice_in",
        ]
        if any(merged_out_dups):
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                logging.debug(
                    "Some tied_out notes matched multiple tied_in notes and could not"
                    " be disambiguated with voice and staff. Choosing arbitrarily:\n"
                    f"{merged.loc[merged_out_dups, cols_to_log]}"
                )

        merged = merged.drop_duplicates(subset=["file_id", "note_id_out"])

        merged_in_dups = merged.set_index(["file_id", "note_id_in"]).index.duplicated(keep=False)
        if any(merged_in_dups):
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                logging.debug(
                    "Some tied_in notes matched multiple tied_out notes and could not"
                    " be disambiguated with voice and staff. Choosing arbitrarily:\n"
                    f"{merged.loc[merged_in_dups, cols_to_log]}"
                )

        return merged.drop_duplicates(subset=["file_id", "note_id_in"])

    def update_tied_out_notes(tied_out_notes: pd.DataFrame, new_values: pd.DataFrame):
        """
        Update the given tied_out_notes columns with the values from the new_values dataframe.

        Parameters
        ----------
        tied_out_notes : pd.DataFrame
            The tied_out_notes dataframe which will be updated (in place) with new values.
            Must have at least columns offset_mc (int), offset_beat (Fraction), new_dur (Fraction),
            and tied (Int64).
        new_values : pd.DataFrame
            A dataframe to draw new values from. Must include at least columns offset_mc_in (int),
            offset_beat_in (Fraction), new_dur (Fraction), new_tied (Int64).
        """
        if len(new_values) > 0:
            tied_out_notes.loc[
                new_values.index, ["offset_mc", "offset_beat", "duration", "tied"]
            ] = new_values.loc[:, ["offset_mc_in", "offset_beat_in", "new_dur", "new_tied"]].values

    def update_step(
        tied_out_notes: pd.DataFrame,
        new_tied_out_values_df: pd.DataFrame,
        finished_notes_dfs: List[pd.DataFrame],
        tied_in_notes: pd.DataFrame,
        tied_in_drop_indexes: pd.Index,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a single update step in the merging process. That is:
        1. Update tied_out_notes with the new values from new_tied_out_values_df.
        2. Remove now finished tied_out_notes from the tied_out_notes dataframe.
        3. Drop now-matched notes (given as indexes) from the tied_in_notes dataframe.

        Parameters
        ----------
        tied_out_notes : pd.DataFrame
            The current tied out notes df. An updated version of this df will be returned.
        new_tied_out_values_df : pd.DataFrame
            The df to get new values for the tied_out_notes df.
        finished_notes_dfs : List[pd.DataFrame]
            A List to store dfs containing the finished notes removed from tied_out_notes.
        tied_in_notes : pd.DataFrame
            The tied in notes df. An updated version of this df will be returned.
        tied_in_drop_indexes : pd.Index
            Indexes of the rows to drop from tied_in_notes.

        Returns
        -------
        tied_out_notes : pd.DataFrame
            A new new tied_out_notes dataframe, which has been updated with new values, and from
            which finished notes have been removed.
        tied_in_notes : pd.DataFrame
            A new tied_in_notes dataframe, from which used notes have been dropped.
        """
        update_tied_out_notes(tied_out_notes, new_tied_out_values_df)
        tied_out_notes = remove_finished(
            tied_out_notes, tied_out_notes["tied"].isnull(), finished_notes_dfs
        )
        tied_in_notes = tied_in_notes.drop(tied_in_drop_indexes)

        return tied_out_notes, tied_in_notes

    # Split notes into those that will change and those that will not
    changing_notes_mask = notes["gracenote"].isnull() & ~notes["tied"].isnull()
    changing_notes = notes.loc[changing_notes_mask]
    unchanging_notes = notes.loc[~changing_notes_mask].copy()

    # Tracking dfs for tied out and in notes. These will be kept up to date while iterating
    tied_out_notes = changing_notes.loc[changing_notes["tied"] == 1].copy()
    tied_in_notes = changing_notes.loc[changing_notes["tied"].isin([0, -1])].copy()

    # This will track all notes that are finished, i.e. either fully-matched or unmatchable
    finished_notes_dfs = []

    if len(tied_out_notes) == 0:
        tied_out_notes, tied_in_notes = repopulate_tied_out_notes(tied_in_notes)

    while len(tied_out_notes) > 0:
        merged = pd.merge(
            tied_out_notes.reset_index(),
            tied_in_notes.reset_index(),
            how="inner",
            left_on=["file_id", "midi", "offset_mc", "offset_beat"],
            right_on=["file_id", "midi", "mc", NOTE_ONSET_BEAT],
            suffixes=("_out", "_in"),
        )

        add_new_tied_values(merged)
        merged_out_indexed, merged_in_indexed = get_out_and_in_views(merged)

        # Find unmatched tied_out_notes, remove them from tied_out_notes, and add them to finished
        unmatched = ~tied_out_notes.index.isin(merged_out_indexed.index)
        if any(unmatched):
            tied_out_notes = remove_finished(tied_out_notes, unmatched, finished_notes_dfs)

        # Find pairs of notes where each is only in merged_notes once
        single_match_out_mask = merged_out_indexed.index.isin(
            merged_out_indexed.loc[merged_out_indexed.index.value_counts() == 1].index
        )
        single_match_in_mask = merged_in_indexed.index.isin(
            merged_in_indexed.loc[merged_in_indexed.index.value_counts() == 1].index
        )
        single_match_both = single_match_out_mask & single_match_in_mask

        # UPDATE STEP: update tied out notes, remove finished, drop tied in notes
        tied_out_notes, tied_in_notes = update_step(
            tied_out_notes,
            merged_out_indexed.loc[single_match_both],
            finished_notes_dfs,
            tied_in_notes,
            merged_in_indexed.index[single_match_both],
        )

        # Re-match doubly-matched notes using staff and voice
        doubly_matched = ~single_match_both

        if any(doubly_matched):
            merged_doubly = merged.loc[doubly_matched]
            merged_more = merged_doubly.loc[
                (merged_doubly["staff_out"] == merged_doubly["staff_in"])
                & (merged_doubly["voice_out"] == merged_doubly["voice_in"])
            ]

            merged_more_no_dup = naive_duplicate_drop(merged_more)
            merged_more_out, merged_more_in = get_out_and_in_views(merged_more_no_dup)

            # UPDATE STEP: update tied out notes, remove finished, drop tied in notes
            tied_out_notes, tied_in_notes = update_step(
                tied_out_notes,
                merged_more_out,
                finished_notes_dfs,
                tied_in_notes,
                merged_more_in.index,
            )

            # Get notes that were doubly matched, but the staff, voice matching eliminated them all
            merged_doubly_out_indexes_orig = merged_doubly.set_index(
                ["file_id", "note_id_out"]
            ).index
            need_to_force_out_indexes = set(merged_doubly_out_indexes_orig) - set(
                merged_more_out.index
            )

            # If some exist, force-match them to the first match
            if len(need_to_force_out_indexes) > 0:
                to_force = pd.merge(
                    tied_out_notes.loc[list(need_to_force_out_indexes)].reset_index(),
                    tied_in_notes.reset_index(),
                    how="inner",
                    left_on=["file_id", "midi", "offset_mc", "offset_beat"],
                    right_on=["file_id", "midi", "mc", NOTE_ONSET_BEAT],
                    suffixes=("_out", "_in"),
                )
                to_force = naive_duplicate_drop(to_force)

                # Get new duration, tied values
                if len(to_force) > 0:
                    add_new_tied_values(to_force)
                    to_force_out, to_force_in = get_out_and_in_views(to_force)

                    # UPDATE STEP: update tied out notes, remove finished, drop tied in notes
                    tied_out_notes, tied_in_notes = update_step(
                        tied_out_notes,
                        to_force_out,
                        finished_notes_dfs,
                        tied_in_notes,
                        to_force_in.index,
                    )

        # Reset tied_out_notes if it is empty
        if len(tied_out_notes) == 0:
            tied_out_notes, tied_in_notes = repopulate_tied_out_notes(tied_in_notes)

    merged_df = pd.concat([unchanging_notes, tied_in_notes] + finished_notes_dfs).sort_index()

    error_notes = merged_df.loc[merged_df["gracenote"].isnull() & ~merged_df["tied"].isnull()]
    if len(error_notes) > 0:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            columns = ["mc", NOTE_ONSET_BEAT, "offset_mc", "offset_beat", "midi", "tied"]
            tied_out_errors = error_notes.loc[error_notes["tied"] == 1, columns]
            tied_in_errors = error_notes.loc[error_notes["tied"] == -1, columns]
            tied_both_errors = error_notes.loc[error_notes["tied"] == 0, columns]

            if len(tied_out_errors) > 0:
                logging.debug(
                    "The following merged notes are tied out but matched with no tie"
                    f" ending:\n{tied_out_errors}"
                )
            if len(tied_in_errors) > 0:
                logging.debug(
                    "The following merged notes are tied in, but matched with no tie"
                    f" beginning:\n{tied_in_errors}"
                )
            if len(tied_both_errors) > 0:
                logging.debug(
                    "The following merged notes are tied in and out, but matched with"
                    f" no tie ending or beginning:\n{tied_both_errors}"
                )

    return merged_df
