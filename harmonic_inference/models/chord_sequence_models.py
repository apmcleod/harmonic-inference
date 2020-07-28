"""Models that generate probability distributions over the next chord in a sequence."""
from model_interface import Model
from harmonic_inference.data.data_types import PitchType


class ChordSequenceModel(Model):
    """
    The base class for all Chord Sequence Models, which model the sequence of chords of a Piece.
    """
    def __init__(self, chord_type: PitchType):
        """
        Create a new base KeySequenceModel with the given output and input data types.

        Parameters
        ----------
        chord_type : PitchType
            The way a given model will output its chords.
        """
        # pylint: disable=invalid-name
        self.CHORD_TYPE = chord_type

    def get_next_chord_prior(self, input_data=None):
        """
        Get a distribution over the next chord given an optional input.

        Parameters
        ----------
        input_data
            Observed data (if required) to predict the next chord.
        """
        raise NotImplementedError

    def next_step(self, chord):
        """
        Tell the model to take a step, giving it the next chord.

        Parameters
        ----------
        chord
            The chord for the model's next step.
        """
        raise NotImplementedError
