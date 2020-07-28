"""Models that output the probability of a key change occurring on a given input."""
from model_interface import Model
from harmonic_inference.data.data_types import PitchType


class KeyTransitionModel(Model):
    """
    The base class for all Key Transition Models which model when a key change will occur.
    """
    def __init__(self, input_type: PitchType = None):
        """
        Create a new base model.

        Parameters
        ----------
        input_type : PitchType, optional
            What type of input the model is expecting in get_change_prob(input_data).
        """
        # pylint: disable=invalid-name
        self.INPUT_TYPE = input_type

    def get_change_prob(self, input_data) -> float:
        """
        Get the probability that a key change will occur given the input data.

        Parameters
        ----------
        input_data
            The input data for the next step.

        Returns
        -------
        probability : float
            The probability that a key change will occur, given the input data.
        """
        raise NotImplementedError

    def next_step(self, change: bool):
        """
        Tell the model to take a step, either with or without a key change.

        Parameters
        ----------
        change : bool
            True if a key change was made on this step. False otherwise.
        """
        raise NotImplementedError
