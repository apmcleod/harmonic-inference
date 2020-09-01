"""Models that generate probability distributions over the next key in a sequence."""
import pytorch_lightning as pl

from harmonic_inference.data.data_types import PitchType


class KeySequenceModel(pl.LightningModule):
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
        super().__init__()
        # pylint: disable=invalid-name
        self.INPUT_TYPE = input_type
        # pylint: disable=invalid-name
        self.KEY_TYPE = key_type
