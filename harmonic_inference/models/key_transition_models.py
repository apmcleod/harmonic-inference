"""Models that output the probability of a key change occurring on a given input."""
import pytorch_lightning as pl

from harmonic_inference.data.data_types import PitchType


class KeyTransitionModel(pl.LightningModule):
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
        super().__init__()
        # pylint: disable=invalid-name
        self.INPUT_TYPE = input_type
