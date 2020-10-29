"""An ICM outputs a prior distribution over the initial (relative) chord in a key."""
import json
from pathlib import Path
from typing import List, Union

import numpy as np

from harmonic_inference.data.data_types import KeyMode, PitchType
from harmonic_inference.data.piece import Chord, Piece


class SimpleInitialChordModel:
    """
    The most simple ICM will output a prior over the initial (relative) chord symbol, given
    whether the key is major or minor.
    """

    def __init__(self, json_path: Union[Path, str]):
        """
        Create a new InitialChordModel by loading it from a json file.

        Parameters
        ----------
        json_path : Union[Path, str]
            The path of a json config file written using the InitialChordModel.train method.
        """
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        self.CHORD_TYPE = PitchType[data["pitch_type"]]
        self.use_inversions = data["use_inversions"]

        self.major_prior = data["major"]
        self.minor_prior = data["minor"]

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

    def evaluate(self, pieces: List[Piece]):
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

    @staticmethod
    def train(
        chords: List[Chord],
        json_path: Union[Path, str],
        use_inversions: bool = True,
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
        use_inversions : bool
            True to use inversions when calculating the initial priors.
        add_n_smoothing : float
            Add a total of this amount of probability mass, uniformly over all chords.
            For example, 1 (default), will add `1 / num_chords` to each prior bin before
            counting.
        """
        pitch_type = chords[0].pitch_type
        one_hot_length = Chord.get_chord_vector_length(
            pitch_type,
            one_hot=True,
            relative=True,
            use_inversions=use_inversions,
            pad=False,
        )

        # Initialize with smoothing
        smoothing_factor = add_n_smoothing / one_hot_length
        major_key_chords_one_hots = np.ones(one_hot_length) * smoothing_factor
        minor_key_chords_one_hots = np.ones(one_hot_length) * smoothing_factor

        # Count chords
        for chord in chords:
            one_hot_index = chord.get_one_hot_index(
                relative=True, use_inversion=use_inversions, pad=False
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
                    "pitch_type": str(pitch_type).split(".")[1],
                    "use_inversions": use_inversions,
                    "major": list(major_key_chords_one_hots),
                    "minor": list(minor_key_chords_one_hots),
                },
                json_file,
                indent=4,
            )
