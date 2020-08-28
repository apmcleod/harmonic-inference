"""A file defining the generic model interface that all models should implement."""
from typing import Dict, Iterable

from harmonic_inference.data.piece import Piece
import pytorch_lightning as pl

class Model(pl.LightningModule):
    """
    A Generic Model class which all others should extend.
    """
    def __init__(self):
        super().__init__()

    def get_state(self) -> Dict:
        """
        Get the state of this Model as a dictionary of fields.

        Returns
        -------
        state : Dict
            The state of this Model as a dictionary of fields. It must contain at least log_prob.
        """
        raise NotImplementedError

    def load_state(self, state: Dict):
        """
        Load a state into this Model from a dictionary.

        Parameters
        ----------
        state : Dict
            The state to load. Like one returned by get_state() or transition(frame).
        """
        raise NotImplementedError

    def initialize(self, piece: Piece) -> Iterable[Dict]:
        """
        Initialize this Model based on a given Piece. This may include setting prior probabilities
        or any other initialization.

        Parameters
        ----------
        piece : Piece
            The Piece that will be run shortly.

        Returns
        -------
        new_states : Iterable[Dict]
            The states that may occur given the piece.
        """
        raise NotImplementedError
