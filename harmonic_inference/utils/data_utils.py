import json
from pathlib import Path
from typing import Any, Dict, Union

from harmonic_inference.data.data_types import ChordType, PitchType


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
