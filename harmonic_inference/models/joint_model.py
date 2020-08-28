"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from typing import List, Tuple

from harmonic_inference.models.chord_classifier_models import ChordClassifierModel
from harmonic_inference.models.chord_sequence_models import ChordSequenceModel
from harmonic_inference.models.chord_transition_models import ChordTransitionModel
from harmonic_inference.models.key_sequence_models import KeySequenceModel
from harmonic_inference.models.key_transition_models import KeyTransitionModel
from harmonic_inference.data.piece import Piece


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
    def __init__(self,
                 chord_classifier: ChordClassifierModel,
                 chord_sequence_model: ChordSequenceModel,
                 chord_transition_model: ChordTransitionModel,
                 key_sequence_model: KeySequenceModel,
                 key_transition_model: KeyTransitionModel):
        """
        Create a new HarmonicInferenceModel from a set of pre-defined model interfaces.

        Parameters
        ----------
        chord_classifier : ChordClassifierModel
            A model that takes as input notes, midi notes, or audio frames, and outputs chord
            probabilities for each.

        chord_sequence_model : ChordSequenceModel
            A model that outputs a probability distribution over the next chord in a sequence of
            chord symbols.

        chord_transition_model : ChordTransitionModel
            A model that outputs a probability at each input step of a chord change.

        key_sequence_model : KeySequenceModel
            A model that outputs a probability distribution over the next key in a sequence of
            keys.

        key_transition_model : KeyTransitionModel
            A model that outputs a probability at each input step of a chord change.
        """
        self.chord_classifier = chord_classifier
        self.chord_sequence_model = chord_sequence_model
        self.chord_transition_model = chord_transition_model
        self.key_sequence_model = key_sequence_model
        self.key_transition_model = key_transition_model

        # Ensure all types match
        assert self.chord_classifier.INPUT_TYPE == self.chord_transition_model.DATA_TYPE, (
            "Chord Classifier input type does not match Chord Transition Model data type"
        )
        assert self.chord_classifier.OUTPUT_TYPE == self.chord_sequence_model.DATA_TYPE, (
            "Chord Classifier output type does not match Chord Sequence Model data type"
        )
        assert self.chord_sequence_model.DATA_TYPE == self.key_transition_model.DATA_TYPE, (
            "Chord Sequence Model data type does not match Key Transition Model data type"
        )
        assert self.chord_sequence_model.DATA_TYPE == self.key_sequence_model.INPUT_TYPE, (
            "Chord Sequence Model data type does not match Key Transition Model input type"
        )

        # pylint: disable=invalid-name
        self.INPUT_TYPE = self.chord_classifier.INPUT_TYPE
        # pylint: disable=invalid-name
        self.CHORD_OUTPUT_TYPE = self.chord_sequence_model.DATA_TYPE
        # pylint: disable=invalid-name
        self.KEY_OUTPUT_TYPE = self.key_sequence_model.OUTPUT_TYPE

    def get_all_chord_boundaries(self, piece: Piece,
                                 threshold: float = 0.1) -> List[Tuple[int, int]]:
        """
        Get a list of all chord boundaries that the Chord Transition Model believes to be possible
        given the input Piece.

        Parameters
        ----------
        piece : Piece
            An input musical piece, either score, midi, or audio.
        threshold : float
            The threshold to allow for a branch during decoding. If the model's output is on the
            range [ctm_threshold, 1 - ctm_threshold], it will branch.

        Returns
        -------
        List[Tuple[int, int, float]]
            A list of chord boundaries, each represented as a tuple (start, end) of indexes into
            piece, where the chord lies on the range [start, end).
        """

    def perform_inference(self, piece: Piece, ctm_threshold: float = 0.1):
        """
        Perform harmonic inference on an input musical piece.

        Parameters
        ----------
        input_piece : Piece
            An input musical piece, either score, midi, or audio.
        ctm_threshold : float
            The threshold to allow for a branch during Chord Transition Model decoding. If the
            model's output is on the range [ctm_threshold, 1 - ctm_threshold], it will branch.

        Returns
        -------
        harmony : list
            The inferred harmonic structure of the piece.
        """
        assert piece.DATA_TYPE == self.INPUT_TYPE

        all_chord_boundaries = self.chord_transition_model.get_all_chord_boundaries(
            piece, threshold=ctm_threshold
        )
