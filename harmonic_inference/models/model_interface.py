"""A file defining the generic model interface that all models should implement."""
from typing import Dict

from harmonic_inference.data.piece import Piece

class Model():
    """
    A Generic Model class which all others should extend.
    """
    def get_state(self) -> Dict:
        raise NotImplementedError

    def load_state(self, state: Dict):
        raise NotImplementedError

    def set_priors(self, piece: Piece):
        raise NotImplementedError

    def transition(self, frame):
        raise NotImplementedError
