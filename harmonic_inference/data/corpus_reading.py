"""Utilities for parsing corpus tsv files into pandas DataFrames."""
import re
from typing import Iterable, Dict, Union
from fractions import Fraction
from glob import glob
from tqdm import tqdm
from pathlib import Path

import pandas as pd


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
        The index (or indices) of column(s) to use as the index. For note_list.tsv,
        use [0, 1, 2].

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
