import json
import os
import pickle
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union

from music21.converter import parse
from tqdm import tqdm

from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.piece import (
    Piece,
    get_measures_df_from_music21_score,
    get_score_piece_from_data_frames,
    get_score_piece_from_dict,
    get_score_piece_from_music_xml,
)


def load_kwargs_from_json(json_path: Union[Path, str, None]) -> Dict[str, Any]:
    """
    Load keyword arguments from a given json file. Fields may be anything json can
    handle. Additionally, this function will load enums as strings in appropriate
    fields. For example, and "ChordType.*" will load correctly in any dict field ending
    in "reduction". Also, "PitchType.*" will load correctly in any non-nested field.

    Parameters
    ----------
    json_path : Union[Path, str, None]
        A path to load a json file from. If None, an empty dict is returned.

    Returns
    -------
    parsed_kwargs : Dict[str, Any]
        The keyword arguments for an init method, which can be used like
        Class(**kwargs).
    """
    if json_path is None:
        return {}

    with open(json_path, "r") as json_file:
        parsed_kwargs = json.load(json_file)

    for key, value in parsed_kwargs.items():
        if isinstance(value, str):
            if value.startswith("PitchType") or value.startswith("PieceType"):
                parsed_kwargs[key] = PitchType[value[10:]]

        elif isinstance(value, dict) and "reduction" in key:
            parsed_kwargs[key] = {
                ChordType[chord_key[10:]]: ChordType[chord_val[10:]]
                for chord_key, chord_val in value.items()
            }

    return parsed_kwargs


def load_pieces_from_xml(input_path: Union[str, Path]):
    xmls = []
    csvs = []

    for file_path in sorted(glob(os.path.join(str(input_path), "**", "*.mxl"), recursive=True)):
        music_xml_path = Path(file_path)
        label_csv_path = (
            music_xml_path.parent.parent / "chords" / Path(str(music_xml_path.stem) + ".csv")
        )

        if music_xml_path.exists() and label_csv_path.exists():
            xmls.append(music_xml_path)
            csvs.append(label_csv_path)


def load_pieces(
    xml: bool = False,
    input_path: Union[str, Path] = "corpus_data",
    piece_dicts_path: Union[str, Path] = None,
    file_ids: List[int] = None,
    specific_id: int = None,
) -> List[Piece]:
    """
    A utility function for loading pieces from xml, dfs, or pickled dicts.

    Parameters
    ----------
    xml : bool
        True to load from musicXML files. False from dfs.
    input_path : Union[str, Path]
        A path to the raw data location (music_xml directory or tsv directory).
    piece_dicts_path : Union[str, Path]
        An optional path to pre-loaded dicts of pieces. Makes loading much faster.
    file_ids : List[int]
        A List of file_ids for the pieces to be loaded.
    specific_id : int
        A specific id if only 1 file should be loaded (must be found in the file_ids list).

    Returns
    -------
    pieces : List[Piece]
        A List of loaded Piece objects.

    Raises
    ------
    ValueError
        If a specific_id is specified, but no file_ids are given, or the specific_id is not found in
        the given file_ids.
    """
    # Load raw data
    if xml:
        xmls = []
        csvs = []

        for file_path in sorted(glob(os.path.join(str(input_path), "**", "*.mxl"), recursive=True)):
            music_xml_path = Path(file_path)
            label_csv_path = (
                music_xml_path.parent.parent / "chords" / Path(str(music_xml_path.stem) + ".csv")
            )

            if music_xml_path.exists() and label_csv_path.exists():
                xmls.append(music_xml_path)
                csvs.append(label_csv_path)

    else:
        files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(input_path)

    # Load from pkl if available
    pkl_path = Path(piece_dicts_path)
    if pkl_path.exists():
        with open(pkl_path, "rb") as pkl_file:
            piece_dicts = pickle.load(pkl_file)
    else:
        piece_dicts = [None] * len(file_ids)

    # Return only one specific file
    if specific_id is not None:
        if file_ids is None or specific_id not in file_ids:
            raise ValueError("Given id not in the set of file_ids to load.")

        index = file_ids.index(specific_id)
        file_ids = [file_ids[index]]
        piece_dicts = [piece_dicts[index]]

    # Load pieces (from dicts or from source files)
    if xml:
        pieces = [
            get_score_piece_from_music_xml(
                xmls[file_id],
                csvs[file_id],
                name=str(xmls[file_id]),
            )
            if piece_dict is None
            else get_score_piece_from_dict(
                get_measures_df_from_music21_score(parse(xmls[file_id])),
                piece_dict,
                name=str(xmls[file_id]),
            )
            for file_id, piece_dict in tqdm(
                zip(file_ids, piece_dicts),
                total=len(file_ids),
                desc="Loading pieces",
            )
        ]
    else:
        pieces = [
            get_score_piece_from_data_frames(
                notes_df.loc[file_id],
                chords_df.loc[file_id],
                measures_df.loc[file_id],
                name=(
                    f"{file_id}: {files_df.loc[file_id, 'corpus_name']}/"
                    f"{files_df.loc[file_id, 'file_name']}"
                ),
            )
            if piece_dict is None
            else get_score_piece_from_dict(
                measures_df.loc[file_id],
                piece_dict,
                name=(
                    f"{file_id}: {files_df.loc[file_id, 'corpus_name']}/"
                    f"{files_df.loc[file_id, 'file_name']}"
                ),
            )
            for file_id, piece_dict in tqdm(
                zip(file_ids, piece_dicts),
                total=len(file_ids),
                desc="Loading pieces",
            )
        ]

    return pieces
