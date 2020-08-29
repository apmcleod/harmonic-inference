"""Utilities for parsing corpus tsv files into pandas DataFrames."""
import re
from typing import Iterable, Dict, Union
from fractions import Fraction
from glob import glob
from tqdm import tqdm
from pathlib import Path
import logging

import pandas as pd

import harmonic_inference.utils.corpus_utils as cu


# Helper functions to be used as converters, handlling empty strings
str2inttuple = lambda l: tuple() if l == '' else tuple(int(s) for s in l.split(', '))
str2strtuple = lambda l: tuple() if l == '' else tuple(str(s) for s in l.split(', '))
iterable2str = lambda iterable: ', '.join(str(s) for s in iterable)
def int2bool(s):
    try:
        return bool(int(s))
    except:
        return pd.NA


CONVERTERS = {
    'added_tones': str2inttuple,
    'act_dur': Fraction,
    'chord_tones': str2inttuple,
    'globalkey_is_minor': int2bool,
    'localkey_is_minor': int2bool,
    'next': str2inttuple,
    'nominal_duration': Fraction,
    'offset': Fraction,
    'onset': Fraction,
    'duration': Fraction,
    'scalar': Fraction
}


DTYPES = {
    'alt_label': str,
    'barline': str,
    'bass_note': 'Int64',
    'cadence': str,
    'cadences_id': 'Int64',
    'changes': str,
    'chord': str,
    'chord_type': str,
    'dont_count': 'Int64',
    'figbass': str,
    'form': str,
    'globalkey': str,
    'gracenote': str,
    'harmonies_id': 'Int64',
    'keysig': int,
    'label': str,
    'localkey': str,
    'mc': 'Int64',
    'midi': int,
    'mn': int,
    'notes_id': 'Int64',
    'numbering_offset': 'Int64',
    'numeral': str,
    'pedal': str,
    'playthrough': int,
    'phraseend': str,
    'relativeroot': str,
    'repeats': str,
    'root': 'Int64',
    'special': str,
    'staff': int,
    'tied': 'Int64',
    'timesig': str,
    'tpc': int,
    'voice': int,
    'voices': int,
    'volta': 'Int64'
}


def read_dump(file: str, index_col: Union[int, Iterable] = (0, 1), converters: Dict = None,
              dtypes: Dict = None, **kwargs) -> pd.DataFrame:
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
    return pd.read_csv(file, sep='\t', index_col=index_col, dtype=types, converters=conv,
                       **kwargs)


def load_clean_corpus_dfs(dir_path: Union[str, Path]):
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
    files_df = read_dump(Path(dir_path, 'files.tsv'), index_col=0)
    measures_df = read_dump(Path(dir_path, 'measures.tsv'))
    chords_df = read_dump(Path(dir_path, 'chords.tsv'), low_memory=False)
    notes_df = read_dump(Path(dir_path, 'notes.tsv'))

    # Remove measure repeats
    if isinstance(measures_df.iloc[0].next, tuple):
        measures_df = cu.remove_repeats(measures_df)

    # Remove unmatched
    notes_df = cu.remove_unmatched(notes_df, measures_df)
    chords_df = cu.remove_unmatched(chords_df, measures_df)
    chords_df = chords_df.drop(chords_df.loc[(chords_df.numeral == '@none') | chords_df.numeral.isnull()].index)

    # Remove notes with invalid onset times
    note_measures = pd.merge(
        notes_df.reset_index(),
        measures_df.reset_index(),
        how='left',
        on=['file_id', 'mc'],
    )

    valid_onsets = (
        (note_measures["offset"] <= note_measures["onset"]) &
        (note_measures["onset"] < note_measures["act_dur"] + note_measures["offset"])
    )
    if not valid_onsets.all():
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            invalid_string = note_measures.loc[
                ~valid_onsets,
                ['file_id', 'note_id', 'mc', 'onset', 'offset', 'act_dur'],
            ]
            logging.warning(
                f"{(~valid_onsets).sum()} notes have invalid onset times:\n{invalid_string}"
            )
        notes_df = notes_df.loc[valid_onsets.values]

    # Remove chords with invalid onset times
    chord_measures = pd.merge(
        chords_df.reset_index(),
        measures_df.reset_index(),
        how='left',
        on=['file_id', 'mc'],
    )

    valid_onsets = (
        (chord_measures["offset"] <= chord_measures["onset"]) &
        (chord_measures["onset"] < chord_measures["act_dur"] + chord_measures["offset"])
    )
    if not valid_onsets.all():
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            invalid_string = chord_measures.loc[
                ~valid_onsets,
                ['file_id', 'chord_id', 'mc', 'onset', 'offset', 'act_dur'],
            ]
            logging.warning(
                f"{(~valid_onsets).sum()} chords have invalid onset times:\n{invalid_string}"
            )
        chords_df = chords_df.loc[valid_onsets.values]

    # Add offsets
    if not all([column in notes_df.columns for column in ['offset_beat', 'offset_mc']]):
        notes_df = cu.add_note_offsets(notes_df, measures_df)

    # Merge ties
    notes_df = cu.merge_ties(notes_df)

    # Add chord metrical info
    chords_df = cu.add_chord_metrical_data(chords_df, measures_df)

    # Remove chords with dur 0
    invalid_dur = chords_df["duration"] <= 0
    if invalid_dur.any():
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            invalid_string = chords_df.loc[
                invalid_dur,
                ['mc', 'onset', 'mc_next', 'onset_next', 'duration'],
            ]
            logging.warning(
                f"{(invalid_dur).sum()} chords have invalid durations:\n{invalid_string}"
            )
        chords_df = chords_df.loc[~invalid_dur].copy()

    return files_df, measures_df, chords_df, notes_df


def aggregate_annotation_dfs(annotations_path: Union[Path, str], out_dir: Union[Path, str]):
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
    """
    if isinstance(annotations_path, str):
        annotations_path = Path(annotations_path)
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    files_dict = {
        'corpus_name': [],
        'file_name': []
    }

    chord_df_list = []
    note_df_list = []
    measure_df_list = []

    for file_string in tqdm(glob(str(Path(annotations_path, '*/harmonies/*.tsv')))):
        file_path = Path(file_string)
        base_path = file_path.parent.parent
        corpus_name = base_path.name
        file_name = file_path.name

        try:
            chord_df = pd.read_csv(Path(base_path, 'harmonies', file_name), dtype=str, sep='\t')
            note_df = pd.read_csv(Path(base_path, 'notes', file_name), dtype=str, sep='\t')
            measure_df = pd.read_csv(Path(base_path, 'measures', file_name), dtype=str, sep='\t')
        except:
            print("Error parsing file " + file_name)
            continue

        files_dict['corpus_name'].append(corpus_name)
        files_dict['file_name'].append(file_name)

        chord_df_list.append(chord_df)
        note_df_list.append(note_df)
        measure_df_list.append(measure_df)

    # Write out aggregated tsvs
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exists_ok=True)

    files_df = pd.DataFrame(files_dict)
    files_df.to_csv(Path(out_dir, 'files.tsv'), sep='\t')

    chords_df = pd.concat(
        chord_df_list, keys=files_df.index, axis=0, names=['file_id', 'chord_id']
    )
    chords_df.to_csv(Path(out_dir, 'chords.tsv'), sep='\t')

    notes_df = pd.concat(
        note_df_list, keys=files_df.index, axis=0, names=['file_id', 'note_id']
    )
    notes_df.to_csv(Path(out_dir, 'notes.tsv'), sep='\t')

    measures_df = pd.concat(
        measure_df_list, keys=list(files_df.index), axis=0, names=['file_id', 'measure_id']
    )
    measures_df.to_csv(Path(out_dir, 'measures.tsv'), sep='\t')
