"""Models that output the probability of a chord change occurring on a given input."""
from model_interface import Model
from harmonic_inference.data.piece import PieceType


class ChordTransitionModel(Model):
    """
    The base class for all Chord Transition Models which model when a chord change will occur.
    """
    def __init__(self, input_type: PieceType = None):
        """
        Create a new base model.

        Parameters
        ----------
        input_type : PieceType, optional
            What type of input the model is expecting in get_change_prob(input_data).
        """
        # pylint: disable=invalid-name
        self.INPUT_TYPE = input_type

    def get_change_prob(self, input_data) -> float:
        """
        Get the probability that a chord change will occur given the input data.

        Parameters
        ----------
        input_data
            The input data for the next step.

        Returns
        -------
        probability : float
            The probability that a chord change will occur, given the input data.
        """
        raise NotImplementedError

    def next_step(self, change: bool):
        """
        Tell the model to take a step, either with or without a chord change.

        Parameters
        ----------
        change : bool
            True if a chord change was made on this step. False otherwise.
        """
        raise NotImplementedError
