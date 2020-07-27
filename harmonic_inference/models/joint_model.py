"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from chord_classifier_models import ChordClassifierModel
from chord_sequence_models import ChordSequenceModel
from chord_transition_models import ChordTransitionModel
from key_sequence_models import KeySequenceModel
from key_transition_models import KeyTransitionModel


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

    def perform_inference(self, input_piece):
        """
        Perform harmonic inference on an input musical piece.

        Parameters
        ----------
        input_piece : Piece
            An input musical piece, either score, midi, or audio.

        Returns
        -------
        harmony : list
            The inferred harmonic structure of the piece.
        """
        # TODO
