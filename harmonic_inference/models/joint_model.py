"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from typing import Dict, Iterable
import itertools

from model_interface import Model
from chord_classifier_models import ChordClassifierModel
from chord_sequence_models import ChordSequenceModel
from chord_transition_models import ChordTransitionModel
from key_sequence_models import KeySequenceModel
from key_transition_models import KeyTransitionModel
from harmonic_inference.data.piece import Piece


class Beam():
    """
    A Beam to perform beam search.
    """
    def __init__(self, beam_size: int):
        self.beam_size = beam_size
        self.beam = []

    def __len__(self):
        return len(self.beam)

    def __iter__(self):
        return self.beam.__iter__()

    def add(self, state):
        """
        Add the given state to this Beam.

        Parameters
        ----------
        state
            The state of a HarmonicInferenceModel to add to this Beam.
        """
        self.beam.append(state)

    def get_top_state(self):
        """
        Get the most probable state from this beam.

        Returns
        -------
        top_state : State
            The most probable state from the beam.
        """
        best_state = None
        best_prob = float("-infinity")

        for state in self.beam:
            if state['log_prob'] > best_prob:
                best_state = state
                best_prob = state['log_prob']

        return best_state

    def cut_to_size(self):
        """
        Cut this beam down to the desired beam_size (set in __init__()).
        """
        self.beam = sorted(self.beam, key=lambda s: s['log_prob'], reverse=True)[:self.beam_size]


class HarmonicInferenceModel(Model):
    """
    A model to perform harmonic inference on an input score, midi, or audio piece.
    """
    def __init__(self,
                 chord_classifier: ChordClassifierModel,
                 chord_sequence_model: ChordSequenceModel,
                 chord_transition_model: ChordTransitionModel,
                 key_sequence_model: KeySequenceModel,
                 key_transition_model: KeyTransitionModel):
        """
        Create a new HarmonicInferenceModel from a set of pre-defined model interfaces.

        Parameters
        ----------
        chord_classifier : ChordClassifierModel
            A model that takes as input notes, midi notes, or audio frames, and outputs chord
            probabilities for each.

        chord_sequence_model : ChordSequenceModel
            A model that outputs a probability distribution over the next chord in a sequence of
            chord symbols.

        chord_transition_model : ChordTransitionModel
            A model that outputs a probability at each input step of a chord change.

        key_sequence_model : KeySequenceModel
            A model that outputs a probability distribution over the next key in a sequence of
            keys.

        key_transition_model : KeyTransitionModel
            A model that outputs a probability at each input step of a chord change.
        """
        # pylint: disable=invalid-name
        self.DATA_TYPE = chord_classifier.DATA_TYPE

        self.chord_classifier = chord_classifier
        self.chord_sequence_model = chord_sequence_model
        self.chord_transition_model = chord_transition_model
        self.key_sequence_model = key_sequence_model
        self.key_transition_model = key_transition_model

        self.log_prob = (self.chord_classifier.log_prob + self.chord_sequence_model.log_prob +
                         self.chord_transition_model.log_prob + self.key_sequence_model.log_prob +
                         self.key_transition_model.log_prob)

    def get_state(self) -> Dict:
        """
        Get the state of this model as a dict of the states of its component models.

        Returns
        -------
        state : Dict
            The state of this model as a dict of the states of its component models,
            plus the log_prob of the overall state.
        """
        state = {
            'chord_classifier_state': self.chord_classifier.get_state(),
            'chord_sequence_model_state': self.chord_sequence_model.get_state(),
            'chord_transition_model_state': self.chord_transition_model.get_state(),
            'key_sequence_model_state': self.key_sequence_model.get_state(),
            'key_transition_model_state': self.key_transition_model.get_state(),
            'log_prob': self.log_prob
        }
        return state

    def load_state(self, state: Dict):
        """
        Load the given state into this model.

        Parameters
        ----------
        state : Dict
            A state returned by this model's get_state() function.
        """
        self.chord_classifier.load_state(state['chord_classifier_state'])
        self.chord_sequence_model.load_state(state['chord_sequence_model_state'])
        self.chord_transition_model.load_state(state['chord_transition_model_state'])
        self.key_sequence_model.load_state(state['key_sequence_model_state'])
        self.key_transition_model.load_state(state['key_transition_model_state'])
        self.log_prob = state['log_prob']

    def initialize(self, piece: Piece) -> Iterable[Dict]:
        """
        Initialize each model based on the Piece.

        Parameters
        ----------
        piece : Piece
            The input Piece which will be decoded.
        """
        new_states_cc = self.chord_classifier.initialize(piece)
        new_states_csm = self.chord_sequence_model.initialize(piece)
        new_states_ctm = self.chord_transition_model.initialize(piece)
        new_states_ksm = self.key_sequence_model.initialize(piece)
        new_states_ktm = self.key_transition_model.initialize(piece)

        new_states = []
        for cc_state, csm_state, ctm_state, ksm_state, ktm_state in itertools.product(
                new_states_cc, new_states_csm, new_states_ctm, new_states_ksm, new_states_ktm):
            state = {
                'chord_classifier_state': cc_state,
                'chord_sequence_model_state': csm_state,
                'chord_transition_model_state': ctm_state,
                'key_sequence_model_state': ksm_state,
                'key_transition_model_state': ktm_state,
                'log_prob': (cc_state['log_prob'] + csm_state['log_prob'] + ctm_state['log_prob'] +
                             ksm_state['log_prob'] + ktm_state['log_prob'])
            }
            new_states.append(state)

        return new_states



    def process_frame(self, frame) -> Iterable[Dict]:
        """
        Get an Iterable of possible next model states given the current model and the input frame.

        Parameters
        ----------
        frame
            The next input frame of the Piece we are decoding.

        Returns
        -------
        Iterable[Dict]
            An iterable of possible states that this model will transition into given the frame.
        """
        new_states = []

        transition_prob = self.chord_transition_model.get_transition_prob(frame)

        no_transition_state = self.get_state()
        no_transition_state['chord_transition_model_state'] = no_transition
        no_transition_state['log_prob'] += no_transition.log_prob



def perform_inference(piece: Piece, model: HarmonicInferenceModel, beam_size: int = 50):
    """
    Perform harmonic inference on an input musical piece.

    Parameters
    ----------
    input_piece : Piece
        An input musical piece, either score, midi, or audio.

    model : HarmonicInferenceModel
        A joint model for performing harmonic inference.

    beam_size : int
        The beam size to use for beam search decoding.

    Returns
    -------
    harmony : list
        The inferred harmonic structure of the piece.
    """
    assert piece.DATA_TYPE == model.DATA_TYPE

    beam = Beam(beam_size)

    for state in model.initialize(piece):
        beam.add(state)

    beam.cut_to_size()

    for frame in piece:
        new_beam = Beam(beam_size)

        for state in beam:
            model.load_state(state)

            for new_state in model.process_frame(frame):
                new_beam.add(new_state)

        new_beam.cut_to_size()
        beam = new_beam

    model.log_prob(beam.get_top_state())
    return beam.get_top_state()
