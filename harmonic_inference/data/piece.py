"""A class storing a musical piece from score, midi, or audio format."""
from enum import Enum

class DataType(Enum):
    """
    An enum indicating the type of data stored by a given Piece. Either score, midi, or audio.
    """
    SCORE = 0
    MIDI = 1
    AUDIO = 2


class Piece():
    """
    A single musical piece, which can be from score, midi, or audio.
    """
    def __init__(self, data_type: DataType):
        # pylint: disable=invalid-name
        self.DATA_TYPE = data_type

    def __len__(self) -> int:
        """
        Get the number of data points in this Piece.

        Returns
        -------
        length : int
            The number of data points in this Piece.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int):
        """
        Get the 'idx'th input data point from this Piece.

        Parameters
        ----------
        idx : int
            The index of the data point to return from this Piece.

        Returns
        ------
        data
            The desired data point.
        """
        raise NotImplementedError
