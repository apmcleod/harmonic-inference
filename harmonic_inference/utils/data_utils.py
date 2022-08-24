import json
import logging
import os
import pickle
import sys
from argparse import Namespace
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union

from music21.converter import parse
from tqdm import tqdm

import harmonic_inference.models.initial_chord_models as icm
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.piece import (
    Piece,
    get_measures_df_from_music21_score,
    get_score_piece_from_data_frames,
    get_score_piece_from_dict,
    get_score_piece_from_music_xml,
)
from harmonic_inference.models.joint_model import MODEL_CLASSES


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

        if Path(input_path).is_file():
            xmls = [input_path]
            csvs = [None]
            input_path = Path(input_path).parent

        else:
            for file_path in sorted(
                glob(os.path.join(str(input_path), "**", "*.mxl"), recursive=True)
            ):
                music_xml_path = Path(file_path)
                label_csv_path = (
                    music_xml_path.parent.parent
                    / "chords"
                    / Path(str(music_xml_path.stem) + ".csv")
                )

                if music_xml_path.exists():
                    xmls.append(music_xml_path)
                    csvs.append(label_csv_path) if label_csv_path.exists() else None

    else:
        files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(input_path)

    if not file_ids:
        file_ids = list(files_df.index) if not xml else list(range(len(xmls)))

    # Load from pkl if available
    if piece_dicts_path and Path(piece_dicts_path).exists():
        with open(Path(piece_dicts_path), "rb") as pkl_file:
            piece_dicts = pickle.load(pkl_file)
    elif xml:
        piece_dicts = [None] * len(xmls)
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
                name=f"{file_id}: {xmls[file_id].relative_to(input_path)}",
            )
            if piece_dict is None
            else get_score_piece_from_dict(
                get_measures_df_from_music21_score(parse(xmls[file_id])),
                piece_dict,
                name=f"{file_id}: {xmls[file_id].relative_to(input_path)}",
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
                (
                    chords_df.loc[file_id]
                    if chords_df is not None and file_id in chords_df.index.get_level_values(0)
                    else None
                ),
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


def load_models_from_argparse(ARGS: Namespace, model_type: str = None) -> Dict:
    """
    Get a Dictionary of loaded models from command line arguments.

    Parameters
    ----------
    ARGS : Namespace
        The parsed argparse Namespace from command line arguments including model information.

    model_type : str
        The model name to load (e.g., "ccm"), if only one should be loaded.

    Returns
    -------
    models : Dict
        A dictionary mapping each model abbreviation (csm, ccm, etc.) to a loaded
        instance of a model of that type.
    """
    models = {}
    for model_name, model_classes in MODEL_CLASSES.items():
        if model_type is not None and model_name != model_type:
            continue

        if model_name == "cpm" and hasattr(ARGS, "no_cpm") and ARGS.no_cpm:
            continue

        if model_name == "kppm" and hasattr(ARGS, "no_kppm") and ARGS.no_kppm:
            continue

        if model_name == "icm":
            continue

        DEFAULT_PATH = os.path.join(
            "`--checkpoint`", model_name, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
        )
        try:
            checkpoint_arg = getattr(ARGS, model_name)
        except AttributeError:
            # From create_sched_sampling_data.py
            checkpoint_arg = getattr(ARGS, "model_path")
            DEFAULT_PATH = os.path.join(
                "`--checkpoint`", "model", "lightning_logs", "version_*", "checkpoints", "*.ckpt"
            )
        try:
            version_arg = getattr(ARGS, f"{model_name}_version")
        except AttributeError:
            # From create_sched_sampling_data.py
            version_arg = getattr(ARGS, "model_version")

        if checkpoint_arg == DEFAULT_PATH or version_arg is not None:
            checkpoint_arg = DEFAULT_PATH
            checkpoint_arg = checkpoint_arg.replace("`--checkpoint`", ARGS.checkpoint)
            checkpoint_arg = checkpoint_arg.replace("model", model_name)

            if version_arg is not None:
                checkpoint_arg = checkpoint_arg.replace("_*", f"_{version_arg}")

        possible_checkpoints = sorted(glob(checkpoint_arg))
        if len(possible_checkpoints) == 0:
            logging.error("No checkpoints found for %s in %s.", model_name, checkpoint_arg)
            sys.exit(2)

        if len(possible_checkpoints) == 1:
            checkpoint = possible_checkpoints[0]
            logging.info("Loading checkpoint %s for %s.", checkpoint, model_name)

        else:
            checkpoint = possible_checkpoints[-1]
            logging.info("Multiple checkpoints found for %s. Loading %s.", model_name, checkpoint)

        for model_class in model_classes:
            try:
                models[model_name] = model_class.load_from_checkpoint(checkpoint)
            except Exception:
                continue
            else:
                models[model_name].freeze()
                break

        assert model_name in models, f"Couldn't load {model_name} from checkpoint {checkpoint}."

    if model_type is None or model_type == "icm":
        # Load icm json differently
        icm_path = ARGS.icm_json.replace("`--checkpoint`", ARGS.checkpoint)
        logging.info("Loading checkpoint %s for icm.", icm_path)
        models["icm"] = icm.SimpleInitialChordModel(load_kwargs_from_json(icm_path))

    return models
