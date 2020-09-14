"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from typing import Iterable, List, Tuple, Dict
import heapq

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
from harmonic_inference.data.piece import Chord, Key, Piece
import harmonic_inference.data.datasets as ds


MODEL_CLASSES = {
    'ccm': ccm.SimpleChordClassifier,
    'ctm': ctm.SimpleChordTransitionModel,
    'csm': csm.SimpleChordSequenceModel,
    'ktm': ktm.SimpleKeyTransitionModel,
    'ksm': ksm.SimpleKeySequenceModel,
}

class Beam():
    """
    A Beam to perform beam search.
    """
    def __init__(self, beam_size: int):
        self.beam_size = beam_size
        self.beam = []

    def __len__(self):
        return len(self.beam)

    def __iter__(self):
        return self.beam.__iter__()

    def add(self, state):
        """
        Add the given state to this Beam.

        Parameters
        ----------
        state
            The state of a HarmonicInferenceModel to add to this Beam.
        """
        self.beam.append(state)

    def get_top_state(self):
        """
        Get the most probable state from this beam.

        Returns
        -------
        top_state : State
            The most probable state from the beam.
        """
        best_state = None
        best_prob = float("-infinity")

        for state in self.beam:
            if state['log_prob'] > best_prob:
                best_state = state
                best_prob = state['log_prob']

        return best_state

    def cut_to_size(self):
        """
        Cut this beam down to the desired beam_size (set in __init__()).
        """
        self.beam = sorted(self.beam, key=lambda s: s['log_prob'], reverse=True)[:self.beam_size]


class HarmonicInferenceModel():
    """
    A model to perform harmonic inference on an input score, midi, or audio piece.
    """
    def __init__(
        self,
        models: Dict,
        min_change_prob: float = 0.5,
        max_no_change_prob: float = 0.5
    ):
        """
        Create a new HarmonicInferenceModel from a set of pre-loaded models.

        Parameters
        ----------
        models : Dict
            A dictionary mapping of model components:
                'ccm': A ChordClassifier
                'ctm': A ChordTransitionModel
                'csm': A ChordSequenceModel
                'ktm': A KeyTransitionModel
                'ksm': A KeySequenceModel
        min_change_prob : float
            The minimum probability (from the CTM) on which a chord change can occur.
        max_no_change_prob : float
            The maximum probability (from the CTM) on which a chord is allowed not
            to change.
        """
        for model, model_class in MODEL_CLASSES.keys():
            assert model in models.keys(), f"`{model}` not in models dict."
            assert isinstance(models[model], model_class), (
                f"`{model}` in models dict is not of type {model_class.__name__}."
            )

        self.chord_classifier = models['ccm']
        self.chord_sequence_model = models['csm']
        self.chord_transition_model = models['ctm']
        self.key_sequence_model = models['ksm']
        self.key_transition_model = models['ktm']

        # Ensure all types match
        # TODO: This block is removed now because of a bug in train.py. Once the ccm is
        #       retrained, uncomment this assertion.
        # assert self.chord_classifier.INPUT_TYPE == self.chord_transition_model.INPUT_TYPE, (
        #     "Chord Classifier input type does not match Chord Transition Model input type"
        # )
        assert self.chord_classifier.OUTPUT_TYPE == self.chord_sequence_model.CHORD_TYPE, (
            "Chord Classifier output type does not match Chord Sequence Model chord type"
        )
        assert self.chord_sequence_model.CHORD_TYPE == self.key_transition_model.INPUT_TYPE, (
            "Chord Sequence Model chord type does not match Key Transition Model input type"
        )
        assert self.chord_sequence_model.CHORD_TYPE == self.key_sequence_model.INPUT_TYPE, (
            "Chord Sequence Model chord type does not match Key Transition Model input type"
        )

        # Set joint model types
        self.INPUT_TYPE = self.chord_classifier.INPUT_TYPE
        self.CHORD_OUTPUT_TYPE = self.chord_sequence_model.CHORD_TYPE
        self.KEY_OUTPUT_TYPE = self.key_sequence_model.KEY_TYPE

        # Save other params
        assert min_change_prob <= max_no_change_prob, (
            "Undefined chord change behavior on probability range "
            f"({max_no_change_prob}, {min_change_prob})"
        )
        self.min_change_prob = min_change_prob
        self.max_no_change_prob = max_no_change_prob

    def get_harmonies(
        self,
        pieces: Iterable[Piece]
    ) -> Tuple[List[List[Tuple[int, Chord]]], List[List[Tuple[int, Chord]]]]:
        """
        Run the model on the given pieces and output the harmony of them.

        Parameters
        ----------
        pieces : Iterable[Piece]
            A List of Pieces to perform harmonic inference on.

        Returns
        -------
        chords : List[List[Tuple[int, Chord]]]
            For each Piece, a List of (index, Chord) tuples containing the Chords of the piece
            and the indexes at which they start.
        keys : List[List[Tuple[int, Key]]]
            For each Piece, a List of (index, Key) tuples containing the Keys of the piece
            and the indexes at which they start.
        """
        # Get chord change probabilities (batched, with CTM)
        change_probs = self.get_chord_change_probs(pieces)

        # Calculate valid chord change locations and their probabilities
        chord_ranges = []
        chord_log_probs = []
        for piece_change_log_probs in tqdm(
            change_probs,
            desc="Calculating valid chord change locations"
        ):
            ranges, log_probs = self.get_possible_chord_indexes(piece_change_log_probs)
            chord_ranges.append(ranges)
            chord_log_probs.append(log_probs)

        # TODO: Calculate chord priors for each possible chord range (batched, with CCM)


        # TODO: The remainder of the search must be done one piece at a time


        return chord_ranges, chord_log_probs

    def get_possible_chord_indexes(
        self,
        change_probs: List[float]
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Get the possible chord changes given a list of chord change probabilities.

        Parameters
        ----------
        change_probs : List[float]
            The probability of a chord change on each input of a single Piece.

        Returns
        -------
        chord_ranges : List[Tuple[int, int]]
            A List of every possible chord location given the change probabilities, as
            (start, end) tuples, where start is inclusive and end is exclusive.
        chord_log_probs : List[float]
            The log probability of each chord range occurring, according to the input
            change_probs.
        """
        change_log_probs = np.log(change_probs)
        no_change_log_probs = np.log(1 - change_probs)

        chord_ranges = []
        chord_log_probs = []

        # Starts is a priority queue so that we don't double-check any intervals
        starts = [0]
        heapq.heapify(starts)

        # Efficient checking if an index exists in the priority queue already
        in_starts = np.full(len(change_log_probs), False, dtype=bool)
        in_starts[0] = True

        while starts:
            start = heapq.heappop(starts)

            # Detect any next chord change positions
            running_log_prob = 0.0
            for index, change_prob, change_log_prob, no_change_log_prob in enumerate(
                zip(
                    change_probs[start + 1:],
                    change_log_probs[start + 1:],
                    no_change_log_probs[start + 1:]
                ),
                start=start + 1,
            ):

                if change_prob > self.min_change_prob:
                    # Chord change can occur
                    chord_ranges.append((start, index))
                    chord_log_probs.append(running_log_prob + change_log_prob)

                    if not in_starts[index]:
                        heapq.heappush(starts, index)
                        in_starts[index] = True

                    if change_prob > self.max_no_change_prob:
                        # Chord change must occur
                        break

                # No change can occur
                running_log_prob += no_change_log_prob

        return chord_ranges, chord_log_probs



    def get_chord_change_probs(self, pieces: Iterable[Piece]) -> List[List[float]]:
        """
        Get the Chord Transition Model's outputs for the given pieces.

        Parameters
        ----------
        pieces : Iterable[Piece]
            The Pieces whose CTM outputs to return.

        Returns
        -------
        change_probs : List[List[float]]
            A List of the chord change probability on each input of each of the given Pieces.
        """
        ctm_dataset = ds.ChordTransitionDataset(pieces)
        ctm_loader = DataLoader(
            ctm_dataset,
            batch_size=ds.ChordTransitionDataset.valid_batch_size,
            shuffle=False,
        )

        outputs = []
        for batch in ctm_loader:
            batch_outputs, batch_lengths = self.chord_transition_model.get_output(batch)
            outputs.extend(
                [output[:length].numpy() for output, length in zip(batch_outputs, batch_lengths)]
            )

        return outputs
