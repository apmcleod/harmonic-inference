"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from typing import Iterable, List, Tuple, Dict

from torch.utils.data.dataloader import DataLoader

from harmonic_inference.models.chord_classifier_models import ChordClassifierModel
from harmonic_inference.models.chord_sequence_models import ChordSequenceModel
from harmonic_inference.models.chord_transition_models import ChordTransitionModel
from harmonic_inference.models.key_sequence_models import KeySequenceModel
from harmonic_inference.models.key_transition_models import KeyTransitionModel
from harmonic_inference.data.piece import Chord, Key, Piece
import harmonic_inference.data.datasets as ds


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
    def __init__(self, models: Dict):
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
        """
        for model in ['ccm', 'ctm', 'csm', 'ktm', 'ksm']:
            assert model in models.keys(), f"`{model}` not in models dict."

        self.chord_classifier = models['ccm']
        self.chord_sequence_model = models['csm']
        self.chord_transition_model = models['ctm']
        self.key_sequence_model = models['ksm']
        self.key_transition_model = models['ktm']

        # Ensure all types match
        assert self.chord_classifier.INPUT_TYPE == self.chord_transition_model.INPUT_TYPE, (
            "Chord Classifier input type does not match Chord Transition Model input type"
        )
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

    def get_harmonies(self, pieces: Iterable[Piece]) -> Iterable[Iterable[Tuple[Chord, Key]]]:
        """
        Run the model on the given pieces and output the harmony of them.

        Parameters
        ----------
        pieces : Iterable[Piece]
            A List of Pieces to perform harmonic inference on.

        Returns
        -------
        harmonies : Iterable[Iterable[Tuple[Chord, Key]]]
            A (Chord, Key) tuple for each input in each given Piece.
        """
        transition_probs = self.get_transition_probs(pieces)
        return transition_probs

    def get_transition_probs(self, pieces: Iterable[Piece]) -> List[List[float]]:
        """
        Get the Chord Transition Model's outputs for the given pieces.

        Parameters
        ----------
        pieces : Iterable[Piece]
            The Pieces whose CTM outputs to return.

        Returns
        -------
        transition_probs : List[List[float]]
            A List of the transition probability on each note in each of the given Piece.
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
