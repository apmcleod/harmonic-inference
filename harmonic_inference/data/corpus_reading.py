"""Utilities for parsing corpus tsv files into pandas DataFrames."""
import logging
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Union

import pandas as pd
from tqdm import tqdm

import harmonic_inference.utils.corpus_utils as cu
from harmonic_inference.data.corpus_constants import (
    CHORD_ONSET_BEAT,
    CONVERTERS,
    DTYPES,
    MEASURE_OFFSET,
    NOTE_ONSET_BEAT,
)


def read_dump(
    file: str,
    index_col: Union[int, Iterable] = (0, 1),
    converters: Dict = None,
    dtypes: Dict = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a corpus tsv file into a pandas DataFrame.

    Parameters
    ----------
    file : string
        The tsv file to parse.

    index_col : int or list(int)
        The index (or indices) of column(s) to use as the index. For files.tsv, use 0.

    converters : dict
        Converters which will be passed to the pandas read_csv function. These will
        overwrite/be added to the default list of CONVERTERS.

    dtypes : dict
        Dtypes which will be passed to the pandas read_csv function. These will
        overwrite/be added to the default list of DTYPES.

    Returns
    -------
    df : pd.DataFrame
        The pandas DataFrame, parsed from the given tsv file.
    """
    conv = CONVERTERS.copy()
    types = DTYPES.copy()
    if dtypes is not None:
        types.update(dtypes)
    if converters is not None:
        conv.update(converters)
    return pd.read_csv(file, sep="\t", index_col=index_col, dtype=types, converters=conv, **kwargs)


def load_clean_corpus_dfs(dir_path: Union[str, Path], count: int = None):
    """
    Return cleaned DataFrames from the corpus data in the given directory. The DataFrames will
    be read from files: 'files.tsv', 'measures.tsv', 'chords.tsv', and 'notes.df'.

    They will undergo the following cleaning procedure:
        1. Remove repeats from measures.
        2. Drop note and chords corresponding to removed measures.
        3. Drop chords with numeral '@none' or pd.NAN.
        4. Add offsets to notes.
        5. Merge tied notes.
        6. Add offsets to chords.

    Parameters
    ----------
    dir_path : str or Path
        The path to a directory containing the files: 'files.tsv', 'measures.tsv', 'chords.tsv',
        and 'notes.df'.

    count : int
        If given, the number of pieces to read in. Else, read all of them.

    Returns
    -------
    files_df : pd.DataFrame
        The files data frame.
    measures_df : pd.DataFrame
        The measures data frame with repeats removed.
    chords_df : pd.DataFrame
        The chords data frame, cleaned as described.
    notes_df : pd.DataFrame
        The notes data frame, cleaned as described.
    """
    files_df = read_dump(Path(dir_path, "files.tsv"), index_col=0)
    measures_df = read_dump(Path(dir_path, "measures.tsv"))
    notes_df = read_dump(Path(dir_path, "notes.tsv"))
    try:
        chords_df = read_dump(Path(dir_path, "chords.tsv"), low_memory=False)
    except Exception:
        # Enable loading with no annotations
        chords_df = None

    if count is not None:
        files_df = files_df.iloc[:count]
        measures_df = measures_df.loc[files_df.index]
        notes_df = notes_df.loc[files_df.index]
        if chords_df is not None:
            chords_df = chords_df.loc[files_df.index]

    # Bugfix for Couperin piece "next" error
    files_df = files_df.loc[~(files_df["file_name"] == "c11n08_Rondeau.tsv")]
    measures_df = measures_df.loc[files_df.index]
    notes_df = notes_df.loc[files_df.index]
    if chords_df is not None:
        chords_df = chords_df.loc[chords_df.index.get_level_values(0).isin(files_df.index)]
    # End bugfix

    # Incomplete column renaming
    if "offset" in measures_df.columns:
        measures_df[MEASURE_OFFSET] = measures_df[MEASURE_OFFSET].fillna(measures_df["offset"])
        measures_df = measures_df.drop("offset", axis=1)

    if chords_df is not None:
        if "onset" in chords_df.columns:
            chords_df[CHORD_ONSET_BEAT] = chords_df[CHORD_ONSET_BEAT].fillna(chords_df["onset"])
            chords_df = chords_df.drop("onset", axis=1)

    if "onset" in notes_df.columns:
        notes_df[NOTE_ONSET_BEAT] = notes_df[NOTE_ONSET_BEAT].fillna(notes_df["onset"])
        notes_df = notes_df.drop("onset", axis=1)

    # Remove measure repeats
    if isinstance(measures_df.iloc[0].next, tuple):
        measures_df = cu.remove_repeats(measures_df)

    # Remove unmatched
    notes_df = cu.remove_unmatched(notes_df, measures_df)
    if chords_df is not None:
        chords_df = cu.remove_unmatched(chords_df, measures_df)
        chords_df = chords_df.drop(
            chords_df.loc[(chords_df.numeral == "@none") | chords_df.numeral.isnull()].index
        )

    # Remove notes with invalid onset times
    note_measures = pd.merge(
        notes_df.reset_index(),
        measures_df.reset_index(),
        how="left",
        on=["file_id", "mc"],
    )

    valid_onsets = (note_measures[MEASURE_OFFSET] <= note_measures[NOTE_ONSET_BEAT]) & (
        note_measures[NOTE_ONSET_BEAT] < note_measures["act_dur"] + note_measures[MEASURE_OFFSET]
    )
    if not valid_onsets.all():
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            invalid_string = note_measures.loc[
                ~valid_onsets,
                ["file_id", "note_id", "mc", NOTE_ONSET_BEAT, MEASURE_OFFSET, "act_dur"],
            ]
            logging.debug(
                f"{(~valid_onsets).sum()} notes have invalid onset times:\n{invalid_string}"
            )
        notes_df = notes_df.loc[valid_onsets.values]

    # Remove chords with invalid onset times
    if chords_df is not None:
        chord_measures = pd.merge(
            chords_df.reset_index(),
            measures_df.reset_index(),
            how="left",
            on=["file_id", "mc"],
        )

        valid_onsets = (chord_measures[MEASURE_OFFSET] <= chord_measures[CHORD_ONSET_BEAT]) & (
            chord_measures[CHORD_ONSET_BEAT]
            < chord_measures["act_dur"] + chord_measures[MEASURE_OFFSET]
        )
        if not valid_onsets.all():
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                invalid_string = chord_measures.loc[
                    ~valid_onsets,
                    ["file_id", "chord_id", "mc", CHORD_ONSET_BEAT, MEASURE_OFFSET, "act_dur"],
                ]
                logging.debug(
                    f"{(~valid_onsets).sum()} chords have invalid onset times:\n{invalid_string}"
                )
            chords_df = chords_df.loc[valid_onsets.values]

    # Add offsets
    if not all([column in notes_df.columns for column in ["offset_beat", "offset_mc"]]):
        notes_df = cu.add_note_offsets(notes_df, measures_df)

    # Merge ties
    notes_df = cu.merge_ties(notes_df)

    # Add chord metrical info
    if chords_df is not None:
        chords_df = cu.add_chord_metrical_data(chords_df, measures_df)

    # Remove chords with dur 0
    if chords_df is not None:
        invalid_dur = chords_df["duration"] <= 0
        if invalid_dur.any():
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                invalid_string = chords_df.loc[
                    invalid_dur,
                    ["mc", CHORD_ONSET_BEAT, "mc_next", f"{CHORD_ONSET_BEAT}_next", "duration"],
                ]
                logging.debug(
                    f"{(invalid_dur).sum()} chords have invalid durations:\n{invalid_string}"
                )
            chords_df = chords_df.loc[~invalid_dur].copy()

    return files_df, measures_df, chords_df, notes_df


def aggregate_annotation_dfs(
    annotations_path: Union[Path, str],
    out_dir: Union[Path, str],
    notes_only: bool = False,
):
    """
    Aggregate all annotations from a corpus directory into 4 combined tsvs in an out directory.
    The resulting tsv will be: files.tsv, chords.tsv, notes.tsv, and measures.tsv.

    Parameters
    ----------
    annotations_path : Union[Path, str]
        The path of the corpus annotations. Annotations should lie in directories:
        annotations_path/*/{harmonies/notes/measures}/*.tsv
    out_dir : Union[Path, str]
        The directory to write the output tsvs into.
    notes_only : bool
        If True, the resulting aggregation will include pieces without annotations,
        but which have a notes tsv entry. If False, only those with annotations (a harmonies
        directory) will be included.
    """
    if isinstance(annotations_path, str):
        annotations_path = Path(annotations_path)
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    files_dict = {"corpus_name": [], "file_name": []}

    chord_df_list = []
    note_df_list = []
    measure_df_list = []

    chord_indexes = []
    note_indexes = []
    measure_indexes = []

    dir_name = "notes" if notes_only else "harmonies"

    for file_string in tqdm(
        glob(str(Path(annotations_path, f"**/{dir_name}/*.tsv")), recursive=True)
    ):
        file_path = Path(file_string)
        base_path = file_path.parent.parent
        corpus_name = base_path.name
        file_name = file_path.name

        try:
            chord_df = pd.read_csv(Path(base_path, "harmonies", file_name), dtype=str, sep="\t")
            chord_indexes.append(len(files_dict["file_name"]))
        except Exception:
            logging.info("Error parsing harmonies for file %s", file_name)
            chord_df = None

        try:
            note_df = pd.read_csv(Path(base_path, "notes", file_name), dtype=str, sep="\t")
            note_indexes.append(len(files_dict["file_name"]))
        except Exception:
            logging.info("Error parsing notes for file %s", file_name)
            note_df = None

        try:
            measure_df = pd.read_csv(Path(base_path, "measures", file_name), dtype=str, sep="\t")
            measure_indexes.append(len(files_dict["file_name"]))
        except Exception:
            logging.info("Error parsing measures for file %s", file_name)
            measure_df = None

        files_dict["corpus_name"].append(corpus_name)
        files_dict["file_name"].append(file_name)

        if chord_df is not None:
            chord_df_list.append(chord_df)
        if note_df is not None:
            note_df_list.append(note_df)
        if measure_df is not None:
            measure_df_list.append(measure_df)

    if not files_dict["file_name"]:
        logging.warning("No files found to aggregate.")
        return

    # Write out aggregated tsvs
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    files_df = pd.DataFrame(files_dict)
    files_df.to_csv(Path(out_dir, "files.tsv"), sep="\t")

    if chord_df_list:
        chords_df = pd.concat(
            chord_df_list, keys=chord_indexes, axis=0, names=["file_id", "chord_id"]
        )
        chords_df.to_csv(Path(out_dir, "chords.tsv"), sep="\t")

    if note_df_list:
        notes_df = pd.concat(note_df_list, keys=note_indexes, axis=0, names=["file_id", "note_id"])
        notes_df.to_csv(Path(out_dir, "notes.tsv"), sep="\t")

    if measure_df_list:
        measures_df = pd.concat(
            measure_df_list, keys=measure_indexes, axis=0, names=["file_id", "measure_id"]
        )
        measures_df.to_csv(Path(out_dir, "measures.tsv"), sep="\t")
