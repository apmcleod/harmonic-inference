"""An ICM outputs a prior distribution over the initial (relative) chord in a key."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from harmonic_inference.data.chord import Chord, get_chord_vector_length
from harmonic_inference.data.data_types import NO_REDUCTION, ChordType, KeyMode
from harmonic_inference.data.piece import Piece
from harmonic_inference.utils.data_utils import load_kwargs_from_json
from harmonic_inference.utils.harmonic_utils import (
    get_chord_from_one_hot_index,
    get_chord_label_list,
    get_chord_one_hot_index,
)


class SimpleInitialChordModel:
    """
    The most simple ICM will output a prior over the initial (relative) chord symbol, given
    whether the key is major or minor.
    """

    def __init__(
        self,
        json_path: Union[Path, str],
        use_inversions: bool = True,
        reduction: Dict[ChordType, ChordType] = None,
    ):
        """
        Create a new InitialChordModel by loading it from a json file.

        Parameters
        ----------
        json_path : Union[Path, str]
            The path of a json config file written using the InitialChordModel.train method.
        use_inversion : bool
            True to use inversions. False to collaps all inversions of a chord to root position.
        reduction : Dict[ChordType, ChordType]
            A reduction mapping for chord types.
        """
        data = load_kwargs_from_json(json_path)

        self.PITCH_TYPE = data["pitch_type"]

        self.major_prior = data["major"]
        self.minor_prior = data["minor"]

        inversions_present = data["inversions_present"]
        reduction_present = (
            NO_REDUCTION if data["reduction_present"] is None else data["reduction_present"]
        )

        self.use_inversions = use_inversions
        if "use_inversions" in data:
            self.use_inversions = data["use_inversions"]

        self.reduction = reduction
        if "reduction" in data:
            self.reduction = data["reduction"]
        if self.reduction is None:
            self.reduction = NO_REDUCTION

        if self.use_inversions and not inversions_present:
            logging.warning(
                "Inversions not present in ICM json, but use_inversions is True.\n"
                "Setting use_inversions to False."
            )
            self.use_inversions = False

        if reduction_present is not None:
            try:
                new_reduction = {
                    chord_type: self.reduction[reduction_present[chord_type]]
                    for chord_type in ChordType
                }
            except KeyError:
                logging.warning(
                    "Requested reduction invalid for the reduction that is already present:\n"
                    "All values in the reduction_present dict must be keys in the reduction dict.\n"
                    "Setting reduction to reduction_present."
                )
                self.reduction = reduction_present
            else:
                for chord_type in ChordType:
                    if (
                        chord_type in self.reduction
                        and self.reduction[chord_type] != new_reduction[chord_type]
                    ):
                        logging.warning(
                            "Mismatch between given reduction and nested reduction for %s:\n"
                            "  Given: %s\n"
                            "  Nested (will be used instead): %s",
                            chord_type,
                            self.reduction[chord_type],
                            new_reduction[chord_type],
                        )
                    elif chord_type not in self.reduction and chord_type in reduction_present:
                        logging.warning(
                            "Chord reduction inferred from nested reduction for %s:\n"
                            "  Given: None\n"
                            "  Nested (will be used): %s",
                            chord_type,
                            new_reduction[chord_type],
                        )

                self.reduction = new_reduction

        if reduction is not None or not use_inversions:
            num_labels = get_chord_label_list(
                self.PITCH_TYPE,
                use_inversions,
                relative=True,
                reduction=reduction,
            )
            self.major_prior = np.zeros(num_labels)
            self.minor_prior = np.zeros(num_labels)

            for major_prior, minor_prior, (root, chord_type, inversion) in zip(
                data["major"],
                data["minor"],
                get_chord_from_one_hot_index(
                    slice(None, None),
                    self.PITCH_TYPE,
                    relative=True,
                    reduction=reduction_present,
                    use_inversions=inversions_present,
                ),
            ):
                index = get_chord_one_hot_index(
                    chord_type,
                    root,
                    self.PITCH_TYPE,
                    inversion=inversion,
                    use_inversion=use_inversions,
                    relative=True,
                    reduction=reduction,
                )

                self.major_prior[index] += major_prior
                self.minor_prior[index] += minor_prior

        self.major_log_prior = np.log(self.major_prior)
        self.minor_log_prior = np.log(self.minor_prior)

    def get_prior(self, is_minor: bool, log: bool = True) -> List[float]:
        """
        Return the prior (or log prior) over all initial chords.

        Parameters
        ----------
        is_minor : bool
            True if the current key is minor. False for major.
        log : bool
            True to return the log prior. False for probability values.

        Returns
        -------
        prior : List[float]
            The prior (or log prior) over all (relative) chords for the given key mode.
        """
        if is_minor:
            return self.minor_log_prior if log else self.minor_prior
        return self.major_log_prior if log else self.major_prior

    def evaluate(self, pieces: List[Piece]) -> Dict[str, float]:
        """
        Evaluate a loaded ICM over a List of pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to evaluate over.

        Returns
        -------
        results : Dict[str, float]
            A dictionary of the model's accuracy (key "acc") and loss (key "loss") over the
            given pieces.
        """
        correct = 0
        total_loss = 0

        major_max_index = np.argmax(self.get_prior(False, log=True))
        minor_max_index = np.argmax(self.get_prior(True, log=True))

        for piece in pieces:
            first_chord_one_hot = piece.get_chords()[0].get_one_hot_index(
                relative=True, use_inversion=True, pad=False
            )

            if piece.get_keys()[0].relative_mode == KeyMode.MAJOR:
                correct_index = major_max_index
                log_prior = self.get_prior(False, log=True)
            else:
                correct_index = minor_max_index
                log_prior = self.get_prior(True, log=True)

            if first_chord_one_hot == correct_index:
                correct += 1

            total_loss -= log_prior[first_chord_one_hot]

        return {
            "acc": 100 * correct / len(pieces),
            "loss": total_loss / len(pieces),
        }


def train_icm(
    chords: List[Chord],
    json_path: Union[Path, str],
    add_n_smoothing: float = 1.0,
):
    """
    Train a new InitialChordModel and write out the results to a json file.

    Parameters
    ----------
    chords : List[Chord]
        All of the initial chords in the dataset.
    json_path : Union[Path, str]
        The path to write the json output to.
    add_n_smoothing : float
        Add a total of this amount of probability mass, uniformly over all chords.
        For example, 1 (default), will add `1 / num_chords` to each prior bin before
        counting.
    """
    pitch_type = chords[0].pitch_type
    one_hot_length = get_chord_vector_length(
        pitch_type,
        one_hot=True,
        relative=True,
        use_inversions=True,
        pad=False,
        reduction=None,
    )

    # Initialize with smoothing
    smoothing_factor = add_n_smoothing / one_hot_length
    major_key_chords_one_hots = np.ones(one_hot_length) * smoothing_factor
    minor_key_chords_one_hots = np.ones(one_hot_length) * smoothing_factor

    # Count chords
    for chord in chords:
        one_hot_index = chord.get_one_hot_index(
            relative=True, use_inversion=True, pad=False, reduction=None
        )

        if chord.key_mode == KeyMode.MAJOR:
            major_key_chords_one_hots[one_hot_index] += 1
        else:
            minor_key_chords_one_hots[one_hot_index] += 1

    # Normalize
    major_key_chords_one_hots /= np.sum(major_key_chords_one_hots)
    minor_key_chords_one_hots /= np.sum(minor_key_chords_one_hots)

    # Write out result to json
    with open(json_path, "w") as json_file:
        json.dump(
            {
                "inversions_present": True,
                "reduction_present": None,
                "pitch_type": str(pitch_type),
                "major": list(major_key_chords_one_hots),
                "minor": list(minor_key_chords_one_hots),
            },
            json_file,
            indent=4,
        )
