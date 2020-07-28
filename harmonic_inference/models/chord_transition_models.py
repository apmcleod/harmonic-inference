"""Models that output the probability of a chord change occurring on a given input."""
from model_interface import Model


class ChordTransitionModel(Model):
    def get_change_prob(self, input_data) -> float:
        raise NotImplementedError

    def next_step(self, change: bool):
        raise NotImplementedError
