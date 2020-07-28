"""Models that generate probability distributions over the next key in a sequence."""
from model_interface import Model
from harmonic_inference.data.data_types import PitchType


class KeySequenceModel(Model):
    """
    The base class for all Key Sequence Models, which model the sequence of keys of a Piece.
    """
    def __init__(self, key_type: PitchType, input_type: PitchType = None):
        """
        Create a new base KeySequenceModel with the given output and input data types.

        Parameters
        ----------
        key_type : PitchType
            The way a given model will output its key tonics.
        input_type : PitchType, optional
            If a model will take input data, the format of that data.
        """
        # pylint: disable=invalid-name
        self.INPUT_TYPE = input_type
        # pylint: disable=invalid-name
        self.KEY_TYPE = key_type

    def get_next_key_prior(self, input_data=None):
        """
        Get a distribution over the next key given an optional input.

        Parameters
        ----------
        input_data
            Observed data (if required) to predict the next key.
        """
        raise NotImplementedError

    def next_step(self, key):
        """
        Tell the model to take a step, giving it the next key.

        Parameters
        ----------
        key
            The key for the model's next step.
        """
        raise NotImplementedError
