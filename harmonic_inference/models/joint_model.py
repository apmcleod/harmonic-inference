"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from fractions import Fraction
from typing import List, Tuple, Dict, Union
import logging
import heapq
from collections import defaultdict
import itertools

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
from harmonic_inference.data.piece import Piece
import harmonic_inference.data.datasets as ds
from harmonic_inference.data.data_types import KeyMode, PitchType
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.utils.harmonic_constants import TPC_C


MODEL_CLASSES = {
    'ccm': ccm.SimpleChordClassifier,
    'ctm': ctm.SimpleChordTransitionModel,
    'csm': csm.SimpleChordSequenceModel,
    'ktm': ktm.SimpleKeyTransitionModel,
    'ksm': ksm.SimpleKeySequenceModel,
}


LABELS = {
    'chords': None,
    'keys': None,
}


class State:
    """
    The state used during the model's beam search, ordered by their log-probability.

    State's are reverse-linked lists, where each State holds a pointer to the previous
    state, and only contains chord and key information for the most recent of each.
    """
    def __init__(
        self,
        chord: int = None,
        key: int = None,
        change_index: int = 0,
        log_prob: float = 0.0,
        prev_state=None,
        hash_length: int = None,
        csm_hidden_state: np.array = None,
        ktm_hidden_state: np.array = None,
        csm_log_prior: np.array = None,
    ):
        """
        Create a new State.

        Parameters
        ----------
        chord : int
            This state's chord, as a one-hot-index integer.
        key : int
            This state's key, as a one-hot-index integer.
        change_index : int
            The index up to which this state is valid. This state's chord and key are
            valid from the input indexes self.prev_state.change_index -- self.change_index.
        log_prob : float
            The log probability of this state.
        prev_state : State
            The previous state.
        hash_length : int
            The length of hash to use. If prev_state is None, this State's hash_tuple
            will be self.hash_length Nones, as a tuple. Otherwise, this State's hash_tuple
            will be the last self.hash_length-1 entries in prev_state.hash_tuple, with
            this State's (key, chord) tuple appended to it.
        csm_hidden_state : np.array
            The hidden state for the CSM's next step.
        ktm_hidden_state : np.array
            The hidden state for the KTM's next step.
        csm_log_prior : np.array
            The log prior for each (relative) chord symbol, as output by the CSM.
        """
        self.valid = True

        self.chord = chord
        self.key = key
        self.change_index = change_index

        self.log_prob = log_prob

        self.prev_state = prev_state
        if hash_length is not None:
            if self.prev_state is None:
                self.hash_tuple = tuple([None] * hash_length)
            else:
                self.hash_tuple = tuple(list(prev_state.hash_tuple[1:]) + [(key, chord)])

        self.csm_hidden_state = None if csm_hidden_state is None else csm_hidden_state.copy()
        self.ktm_hidden_state = None if ktm_hidden_state is None else ktm_hidden_state.copy()
        self.csm_log_prior = None if csm_log_prior is None else csm_log_prior

    def chord_transition(self, chord: int, change_index: int, log_prob: float):
        """
        Perform a chord transition form this State, and return the new State.

        Parameters
        ----------
        chord : int
            The new state's chord.
        change_index : int
            The input index at which the new state's chord will end.
        log_prob : float
            The log-probability of the given chord transition occurring, in terms of
            absolute chord (CCM) and the chord's index bounds (CTM).

        Returns
        -------
        new_state : State
            The state resulting from the given transition.
        """
        return State(
            chord=chord,
            key=self.key,
            change_index=change_index,
            log_prob=self.log_prob + log_prob,
            prev_state=self,
            hash_length=len(self.hash_tuple) if hasattr(self, 'hash_tuple') else None,
            csm_hidden_state=self.csm_hidden_state,
            ktm_hidden_state=self.ktm_hidden_state,
            csm_log_prior=self.csm_log_prior,
        )

    def can_key_transition(self) -> bool:
        """
        Detect if this state can key transition.

        Key transitions are not allowed on the first chord (since then a different initial
        key would have been set instead).

        Returns
        -------
        can_transition : bool
            True if this state can enter a new key. False otherwise.
        """
        return self.prev_state is not None and self.prev_state.prev_state is not None

    def key_transition(self, key: int, log_prob: float):
        """
        Transition to a new key on the most recent chord.

        Parameters
        ----------
        key : int
            The new key to transition to.
        log_prob : float
            The log-probability of the given key transition, in terms of the input index (KTM)
            and the new key (KSM).

        Returns
        -------
        new_state : State
            Essentially, a replacement of this state (the new state's prev_state is the same),
            but with a new key.
        """
        return State(
            chord=self.chord,
            key=key,
            change_index=self.change_index,
            log_prob=self.prev_state.log_prob + log_prob,
            prev_state=self.prev_state,
            hash_length=len(self.hash_tuple) if hasattr(self, 'hash_tuple') else None,
            csm_hidden_state=self.csm_hidden_state,
            ktm_hidden_state=self.ktm_hidden_state,
            csm_log_prior=self.csm_log_prior,
        )

    def add_csm_prior(self, pitch_type: PitchType) -> None:
        """
        Add the log_prior for this state's current relative chord to this state's log_prob.

        This has its own method essentially to notify the State that its prior, chord, and key
        are all now valid.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root and key tonic.
        """
        self.log_prob += self.csm_log_prior[self.get_relative_chord(pitch_type)]

    def get_csm_input(self) -> np.array:
        """
        Get the input for the next step of this state's CSM.

        Returns
        -------
        csm_input : np.array
            The input for the next step of this state's CSM.
        """
        # TODO
        raise NotImplementedError()

    def get_ktm_input(self) -> np.array:
        """
        Get the input for the next step of this state's KTM.

        Returns
        -------
        ktm_input : np.array
            The input for the next step of this state's KTM.
        """
        # TODO
        raise NotImplementedError()

    def get_ksm_input(self) -> np.array:
        """
        Get the input for this state's KSM. This should be run only when the KTM has decided
        that there should be a key change, and the generated input will be from the last key
        change to the current state.

        Returns
        -------
        ktm_input : np.array
            The input for the KSM form the last key change until now.
        """
        # TODO
        raise NotImplementedError()

    def get_relative_chord(self, pitch_type: PitchType) -> int:
        """
        Get the one-hot index of the chord symbol, relative to the current key.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root and key tonic.

        Returns
        -------
        relative_chord : int
            The current chord, relative to the current key.
        """
        root, chord_type, inversion = LABELS['chord'][self.chord]
        root = hu.transpose_pitch(root, self.key - TPC_C, pitch_type)

        return hu.get_chord_one_hot_index(
            chord_type,
            root,
            pitch_type,
            inversion=inversion,
            use_inversion=True,
        )

    def get_chords(self) -> Tuple[List[int], List[int]]:
        """
        Get the chords and the chord change indexes up to this state.

        Returns
        -------
        chords : List[int]
            A List of the chord symbol indexes up to this State.
        change_indexes : List[int]
            A List of the chord transition indexes up to this State. This list will be of
            length 1 greater than chords because it includes an initial 0.
        """
        if self.prev_state is None:
            return [], [self.change_index]

        chords, changes = self.prev_state.get_chords()
        chords.append(self.chord)
        changes.append(self.change_index)

        return chords, changes

    def get_keys(self) -> Tuple[List[int], List[int]]:
        """
        Get the keys and the key change indexes up to this state.

        Returns
        -------
        keys : List[int]
            A List of the key symbol indexes up to this State.
        change_indexes : List[int]
            A List of the key transition indexes up to this State. This list will be of
            length 1 greater than keys because it includes an initial 0.
        """
        if self.prev_state is None:
            return [], [self.change_index]

        keys, changes = self.prev_state.get_keys()
        if len(keys) == 0 or self.key != keys[-1]:
            keys.append(self.key)
            changes.append(self.change_index)

        # Key is equal to the previous one -- update change index
        elif len(keys) != 0:
            changes[-1] = self.change_index

        return keys, changes

    def get_hash(self) -> Union[Tuple[Tuple[int, int]], int]:
        """
        Get the hash of this State.

        If self.hash_length is not None, this is stored in a field "hash_tuple", which is
        a tuple of (key, chord) tuples of the last hash_length states.

        If self.hash_length is None, the item's id is returned as its hash, as default.

        Returns
        -------
        hash : Union[Tuple[Tuple[int, int]], int]
            Either the last self.hash_length (key, chord) tuple, as a tuple, or this
            object's id.
        """
        try:
            return self.hash_tuple
        except AttributeError:
            return id(self)

    def __lt__(self, other):
        return self.log_prob < other.log_prob


class Beam:
    """
    Beam class for beam search, implemented as a min-heap.

    A min-heap is chosen, since the important part during the beam search is to know
    the minimum (least likely) state currently in the beam (O(1) time), and to be able
    to remove it quickly (O(log(beam_size)) time).

    Getting the maximum state (O(n) time) is only done once, at the end of the beam search.
    """
    def __init__(self, beam_size: int):
        """
        Create a new Beam of the given size.

        Parameters
        ----------
        beam_size : int
            The size of the Beam.
        """
        self.beam_size = beam_size
        self.beam = []

    def fits_in_beam(self, state: State) -> bool:
        """
        Check if the given state will fit in the beam, but do not add it.

        This should be used only to check for early exits. If you will add the
        state to the beam immediately anyways, it is faster to just use
        beam.add(state) and check its return value.

        Parameters
        ----------
        state : State
            The state to check.

        Returns
        -------
        fits : bool
            True if the state will fit in the beam both by size and by log_prob.
        """
        return len(self) < self.beam_size or self.beam[0] < state

    def add(self, state: State) -> bool:
        """
        Add the given state to the beam, if it fits, and return a boolean indicating
        if the state fit.

        Parameters
        ----------
        state : State
            The state to add to the beam.

        Returns
        -------
        added : bool
            True if the given state was added to the beam. False otherwise.
        """
        if len(self) == self.beam_size:
            if self.beam[0] < state:
                heapq.heappushpop(self.beam, state)
                return True
            return False

        # Beam was not yet full
        heapq.heappush(self.beam, state)
        return True

    def get_top_state(self) -> State:
        """
        Get the top state in this beam.

        Returns
        -------
        top_state : State
            The top state in this beam. This runs in O(beam_size) time, since it requires
            a full search of the beam.
        """
        return max(self)

    def empty(self):
        """
        Empty this beam.
        """
        self.beam = []

    def __iter__(self):
        return self.beam.__iter__()

    def __len__(self):
        return len(self.beam)


class HashedBeam(Beam):
    """
    A HashedBeam is like a Beam, but additionally has a dictionary mapping State hashes
    to states, where no two States with the same hash are allowed to be in the beam,
    regardless of the probability of other beam states.

    When a state should be removed from the beam because of the hashed beam, it is easily
    removed from the state dict, but impractical to search through the min-heap to find and
    remove it. It is instead marked as invalid (state.valid = False), and ignored when
    iterating through the states in the beam.

    Care is also taken to ensure that the state on top of the min-heap is always valid,
    so that the minimum log_prob is always known. Thus, when marking a state as invalid,
    if it is on the top of the min-heap, the head of the min-heap is repeatedly removed
    until it is valid. (See _fix_beam_min().)
    """
    def __init__(self, beam_size: int):
        """
        Create a new HashedBeam with the given overall beam size.

        Parameters
        ----------
        beam_size : int
            The size of the beam.
        """
        super().__init__(beam_size)
        self.state_dict = {}

    def fits_in_beam(self, state: State) -> bool:
        """
        Check if the given state will fit in the beam, but do not add it.

        This should be used only to check for early exits. If you will add the
        state to the beam immediately anyways, it is faster to just use
        beam.add(state) and check its return value.

        Parameters
        ----------
        state : State
            The state to check.

        Returns
        -------
        fits : bool
            True if the state will fit in the beam both by size and by log_prob.
        """
        state_hash = state.get_hash()
        return (
            (len(self) < self.beam_size or self.beam[0] < state) and
            (state_hash not in self.state_dict or self.state_dict[state_hash] < state)
        )

    def _fix_beam_min(self):
        """
        Remove all states with valid == False from the top of the min-heap until the min
        state is valid.
        """
        while not self.beam[0].valid:
            heapq.heappop(self.beam)

    def add(self, state: State) -> bool:
        """
        Add the given state to the beam, if it fits, and return a boolean indicating
        if the state fit.

        Parameters
        ----------
        state : State
            The state to add to the beam.

        Returns
        -------
        added : bool
            True if the given state was added to the beam. False otherwise.
        """
        state_hash = state.get_hash()

        if state_hash in self.state_dict:
            if self.state_dict[state_hash] < state:
                self.state_dict[state_hash].valid = False
                self.state_dict[state_hash] = state
                heapq.heappush(self.beam, state)
                self._fix_beam_min()
                return True
            return False

        # Here, the state is in a new hash
        if len(self) == self.beam_size:
            if self.beam[0] < state:
                removed_state = heapq.heappushpop(self.beam, state)
                self.state_dict[state_hash] = state
                del self.state_dict[removed_state.get_hash()]
                self._fix_beam_min()
                return True
            return False

        # Beam is not yet full
        heapq.heappush(self.beam, state)
        self.state_dict[state_hash] = state
        return True

    def empty(self):
        """
        Empty this beam.
        """
        self.beam = []
        self.state_dict = []

    def __iter__(self):
        return self.state_dict.values().__iter__()

    def __len__(self):
        return len(self.state_dict)


class HarmonicInferenceModel:
    """
    A model to perform harmonic inference on an input score, midi, or audio piece.
    """
    def __init__(
        self,
        models: Dict,
        min_chord_change_prob: float = 0.25,
        max_no_chord_change_prob: float = 0.75,
        max_chord_length: Fraction = Fraction(8),
        min_key_change_prob: float = 0.25,
        max_no_key_change_prob: float = 0.75,
        beam_size: int = 500,
        max_chord_branching_factor: int = 20,
        target_chord_branch_prob: float = 0.95,
        max_key_branching_factor: int = 5,
        target_key_branch_prob: float = 0.95,
        hash_length: int = 5,
    ):
        """
        Create a new HarmonicInferenceModel from a set of pre-loaded models.

        Parameters
        ----------
        models : Dict
            A dictionary mapping of model components:
                'ccm': A ChordClassifier
                'ctm': A ChordTransitionModel
                'csm': A ChordSequenceModel
                'ktm': A KeyTransitionModel
                'ksm': A KeySequenceModel
        min_chord_change_prob : float
            The minimum probability (from the CTM) on which a chord change can occur.
        max_no_chord_change_prob : float
            The maximum probability (from the CTM) on which a chord is allowed not to change.
        max_chord_length : Fraction
            The maximum length for a chord generated by this model.
        min_key_change_prob : float
            The minimum probability (from the KTM) on which a key change can occur.
        max_no_key_change_prob : float
            The maximum probability (from the KTM) on which a key is allowed not to change.
        beam_size : int
            The beam size to use for decoding with this model.
        max_chord_branching_factor : int
            For each state during the beam search, the maximum number of different chord
            classifications to try during branching.
        target_chord_branch_prob : float
            Once the chords transitioned into account for at least this much probability mass,
            no more chords are searched, even if the max_chord_branching_factor has not yet been
            reached.
        max_Key_branching_factor : int
            For each state during the beam search, the maximum number of different key
            classifications to try during branching.
        target_key_branch_prob : float
            Once the keys transitioned into account for at least this much probability mass,
            no more keys are searched, even if the max_key_branching_factor has not yet been
            reached.
        hash_length : int
            If not None, a hashed beam is used, where only 1 State is kept in the Beam
        """
        for model, model_class in MODEL_CLASSES.items():
            assert model in models.keys(), f"`{model}` not in models dict."
            assert isinstance(models[model], model_class), (
                f"`{model}` in models dict is not of type {model_class.__name__}."
            )

        self.chord_classifier = models['ccm']
        self.chord_sequence_model = models['csm']
        self.chord_transition_model = models['ctm']
        self.key_sequence_model = models['ksm']
        self.key_transition_model = models['ktm']

        # Ensure all types match
        assert self.chord_classifier.INPUT_TYPE == self.chord_transition_model.INPUT_TYPE, (
            "Chord Classifier input type does not match Chord Transition Model input type"
        )
        assert self.chord_classifier.OUTPUT_TYPE == self.chord_sequence_model.CHORD_TYPE, (
            "Chord Classifier output type does not match Chord Sequence Model chord type"
        )
        assert self.chord_sequence_model.CHORD_TYPE == self.key_transition_model.INPUT_TYPE, (
            "Chord Sequence Model chord type does not match Key Transition Model input type"
        )
        assert self.chord_sequence_model.CHORD_TYPE == self.key_sequence_model.INPUT_TYPE, (
            "Chord Sequence Model chord type does not match Key Transition Model input type"
        )

        # Set joint model types
        self.INPUT_TYPE = self.chord_classifier.INPUT_TYPE
        self.CHORD_OUTPUT_TYPE = self.chord_sequence_model.CHORD_TYPE
        self.KEY_OUTPUT_TYPE = self.key_sequence_model.KEY_TYPE

        # Load labels
        LABELS['chord'] = [
            (
                hu.get_pitch_from_string(root, self.CHORD_OUTPUT_TYPE),
                hu.get_chord_type_from_string(description.split(',')[0]),
                int(inv)
            )
            for root, description, inv in [
                chord.split(':')
                for chord in hu.get_chord_label_list(self.CHORD_OUTPUT_TYPE, use_inversions=True)
            ]
        ]
        LABELS['key'] = [
            (
                hu.get_pitch_from_string(tonic, self.KEY_OUTPUT_TYPE),
                KeyMode[mode.split('.')[1]]
            )
            for tonic, mode in [
                key.split(':')
                for key in hu.get_key_label_list(self.KEY_OUTPUT_TYPE)
            ]
        ]

        # CTM params
        assert min_chord_change_prob <= max_no_chord_change_prob, (
            "Undefined chord change behavior on probability range "
            f"({max_no_chord_change_prob}, {min_chord_change_prob})"
        )
        self.min_chord_change_prob = min_chord_change_prob
        self.max_no_chord_change_prob = max_no_chord_change_prob
        self.max_chord_length = max_chord_length

        # KTM params
        assert min_key_change_prob <= max_no_key_change_prob, (
            "Undefined key change behavior on probability range "
            f"({max_no_key_change_prob}, {min_key_change_prob})"
        )
        self.min_key_change_prob = min_key_change_prob
        self.max_no_key_change_prob = max_no_key_change_prob

        # Chord branching params (CCM)
        self.max_chord_branching_factor = max_chord_branching_factor
        self.target_chord_branch_prob = target_chord_branch_prob

        # Key branching params (KSM)
        self.max_key_branching_factor = max_key_branching_factor
        self.target_key_branch_prob = target_key_branch_prob

        # Beam search params
        self.beam_size = beam_size
        self.hash_length = hash_length

    def get_harmony(self, piece: Piece) -> State:
        """
        Run the model on a piece and output its harmony.

        Parameters
        ----------
        piece : Piece
            A Piece to perform harmonic inference on.

        Returns
        -------
        state : State
            The top estimated state.
        """
        # Get chord change probabilities (with CTM)
        logging.info("Getting chord change probabilities")
        change_probs = self.get_chord_change_probs(piece)

        # Calculate valid chord ranges and their probabilities
        logging.info("Calculating valid chord ranges")
        chord_ranges, chord_log_probs = self.get_chord_ranges(piece, change_probs)

        # Calculate chord priors for each possible chord range (batched, with CCM)
        logging.info("Classifying chords")
        chord_classifications = self.get_chord_classifications(piece, chord_ranges)

        # Iterative beam search for other modules
        logging.info("Performing iterative beam search")
        state = self.beam_search(
            piece,
            chord_ranges,
            chord_log_probs,
            chord_classifications,
        )

        return state

    def get_chord_change_probs(self, piece: Piece) -> List[float]:
        """
        Get the Chord Transition Model's outputs for a given piece.

        Parameters
        ----------
        piece : Piece
            A Piece whose CTM outputs to return.

        Returns
        -------
        change_probs : List[float]
            A List of the chord change probability on each input of the given Piece.
        """
        ctm_dataset = ds.ChordTransitionDataset([piece])
        ctm_loader = DataLoader(
            ctm_dataset,
            batch_size=ds.ChordTransitionDataset.valid_batch_size,
            shuffle=False,
        )

        # CTM keeps each piece as a single input, so will only have 1 batch
        for batch in ctm_loader:
            batch_output, batch_length = self.chord_transition_model.get_output(batch)
            return batch_output[0][:batch_length[0]].numpy()

    def get_chord_ranges(
        self,
        piece: Piece,
        change_probs: List[float],
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Get all possible chord ranges and their log-probability, given the chord change
        probabilities for each input in the Piece.

        Parameters
        ----------
        piece : Piece
            The Piece whose chord ranges to return.
        change_probs : List[float]
            The probability of a chord change occurring on each input of the Piece.

        Returns
        -------
        chord_ranges : List[Tuple[int, int]]
            A List of possible chord ranges, as (start, end) tuples representing the start
            (inclusive) and end (exclusive) points of each possible range.
        chord_log_probs : List[float]
            For each chord range, it's log-probability, including its end change, but not its
            start change.
        """
        chord_ranges = []
        chord_log_probs = []
        duration_cache = piece.get_duration_cache()

        # Invalid masks all but first note at each onset position
        first = 0
        invalid = np.full(len(change_probs), False, dtype=bool)
        for i, (prev_note, note) in enumerate(
            zip(piece.get_inputs()[:-1], piece.get_inputs()[1:]),
            start=1,
        ):
            if prev_note.onset == note.onset:
                invalid[i] = True
            else:
                if first != i - 1:
                    change_probs[first] = np.max(change_probs[first:i])
                first = i

        # Log everything vectorized
        change_log_probs = np.log(change_probs)
        no_change_log_probs = np.log(1 - change_probs)

        # Starts is a priority queue so that we don't double-check any intervals
        starts = [0]
        heapq.heapify(starts)

        # Efficient checking if an index exists in the priority queue already
        in_starts = np.full(len(change_log_probs), False, dtype=bool)
        in_starts[0] = True

        while starts:
            start = heapq.heappop(starts)

            running_log_prob = 0.0
            running_duration = Fraction(0.0)
            reached_end = True

            # Detect any next chord change positions
            for index, (change_prob, change_log_prob, no_change_log_prob, duration) in enumerate(
                zip(
                    change_probs[start + 1:],
                    change_log_probs[start + 1:],
                    no_change_log_probs[start + 1:],
                    duration_cache[start:]  # Off-by-one because cache is dur to next note
                ),
                start=start + 1,
            ):
                if invalid[index]:
                    continue

                running_duration += duration
                if running_duration > self.max_chord_length:
                    reached_end = False
                    break

                if change_prob >= self.min_chord_change_prob:
                    # Chord change can occur
                    chord_ranges.append((start, index))
                    chord_log_probs.append(running_log_prob + change_log_prob)

                    if not in_starts[index]:
                        heapq.heappush(starts, index)
                        in_starts[index] = True

                    if change_prob > self.max_no_chord_change_prob:
                        # Chord change must occur
                        reached_end = False
                        break

                # No change can occur
                running_log_prob += no_change_log_prob

            # Detect if a chord reaches the end of the piece and add it here if so
            if reached_end:
                chord_ranges.append((start, len(change_probs)))
                chord_log_probs.append(running_log_prob)

        return chord_ranges, chord_log_probs

    def get_chord_classifications(
        self,
        piece: Piece,
        ranges: List[Tuple[int, int]],
    ) -> List[np.array]:
        """
        Generate a chord type prior for each potential chord (from ranges).

        Parameters
        ----------
        piece : Piece
            The Piece for which we want to classify the chords.
        ranges : List[Tuple[int, int]]
            A List of all possible chord ranges as (start, end) for the Piece.

        Returns
        -------
        classifications : List[np.array]
            The prior log-probability over all chord symbols for each given range.
        """
        ccm_dataset = ds.ChordClassificationDataset([piece], ranges=[ranges], dummy_targets=True)
        ccm_loader = DataLoader(
            ccm_dataset,
            batch_size=ds.ChordClassificationDataset.valid_batch_size,
            shuffle=False,
        )

        # Get classifications
        classifications = []
        for batch in tqdm(ccm_loader, desc="Classifying chords"):
            classifications.extend(
                [output.numpy() for output in self.chord_classifier.get_output(batch)]
            )

        return np.log(classifications)

    def beam_search(
        self,
        piece: Piece,
        chord_ranges: List[Tuple[int, int]],
        chord_log_probs: List[float],
        chord_classifications: List[np.array],
    ) -> State:
        """
        Perform a beam search over the given Piece to label its Chords and Keys.

        Parameters
        ----------
        piece : Piece
            The Piece to beam search over.
        chord_ranges : List[Tuple[int, int]]
            A List of possible chord ranges, as (start, end) tuples.
        chord_log_probs : List[float]
            The log probability of each chord ranges in chord_ranges.
        chord_classifications : List[np.array]
            The prior log-probability over all chord symbols for each given range.

        Returns
        -------
        state : State
            The top state after the beam search.
        """
        # Dict mapping start of chord range to list of data tuples
        chord_ranges_dict = defaultdict(list)

        priors = np.exp(chord_classifications)
        priors_argsort = np.argsort(-priors)  # Negative to sort descending
        max_indexes = np.clip(
            np.argmax(
                np.cumsum(
                    np.take_along_axis(priors, priors_argsort, -1),
                    axis=-1,
                ) >= self.target_chord_branch_prob,
                axis=-1,
            ) + 1,
            1,
            self.max_chord_branching_factor,
        )
        for (start, end), range_log_prob, log_prior, prior_argsort, max_index in zip(
            chord_ranges,
            chord_log_probs,
            chord_classifications,
            priors_argsort,
            max_indexes,
        ):
            chord_ranges_dict[start].append(
                (
                    end,
                    range_log_prob,
                    log_prior,
                    prior_argsort,
                    max_index,
                )
            )

        beam_class = Beam if self.hash_length is None else HashedBeam
        all_states = [beam_class(self.beam_size) for _ in range(len(piece.get_inputs()) + 1)]
        all_states[0].add(State(key=0, hash_length=self.hash_length))

        for current_start, current_states in tqdm(
            enumerate(all_states[:-1]),
            desc="Beam searching through inputs",
            total=len(all_states) - 1,
        ):
            # Run CSM here to avoid running it for invalid states
            self.run_csm_batched(list(current_states))

            to_check_for_key_change = []

            # Initial branch on absolute chord symbol
            for state, range_data in itertools.product(
                current_states, chord_ranges_dict[current_start]
            ):
                (
                    range_end,
                    range_log_prob,
                    chord_log_priors,
                    chord_priors_argsort,
                    max_index,
                ) = range_data

                # Ensure each state branches at least once
                if max_index == 1 and chord_priors_argsort[0] == state.chord:
                    max_index = 2

                # Branch
                for chord_id in chord_priors_argsort[:max_index]:
                    if chord_id == state.chord:
                        # Disallow self-transitions
                        continue

                    # Calculate the new state on this absolute chord
                    new_state = state.chord_transition(
                        chord_id, range_end, range_log_prob + chord_log_priors[chord_id]
                    )

                    to_check_for_key_change.append(new_state)

            # Check for key changes
            change_probs = self.get_key_change_probs(to_check_for_key_change)
            no_change_log_probs = np.log(1 - change_probs)
            change_log_probs = np.log(change_probs)

            # Branch on key changes
            to_csm_prior_states = []
            to_ksm_states = []
            for change_prob, change_log_prob, no_change_log_prob, state in zip(
                change_probs,
                change_log_probs,
                no_change_log_probs,
                to_check_for_key_change
            ):
                can_not_change = change_prob <= self.max_no_key_change_prob
                can_change = change_prob >= self.min_key_change_prob

                # Make a copy only if necessary
                change_state = state.copy() if can_change and can_not_change else state

                if can_change:
                    change_state.log_prob += change_log_prob
                    to_ksm_states.append(change_state)

                if can_not_change:
                    state.log_prob += no_change_log_prob
                    to_csm_prior_states.append(state)

            # Change keys and put resulting states into the appropriate beam
            for state in self.get_key_change_states(to_ksm_states):
                all_states[state.change_index].add(state)

            # Add CSM prior and add to beam (CSM is run at the start of each iteration)
            for state in to_csm_prior_states:
                state.add_csm_prior(self.CHORD_OUTPUT_TYPE)

                # Add state to its beam, if it fits
                all_states[state.change_index].add(state)

            current_states.empty()

        return all_states[-1].get_top_state()

    def get_key_change_probs(self, states: List[State]) -> np.array:
        """
        Get the probability of a key change for each of the given states.

        Parameters
        ----------
        states : List[State]
            The States to check for key changes. Some internal fields of these states may change.

        Returns
        -------
        key_change_probs : np.array[float]
            The probability of a key change occurring for each given State.
        """
        key_change_probs = np.zeros(len(states), dtype=float)

        # Check for states that can key transition
        valid = np.array([state.can_key_transition() for state in states])
        valid_count = np.sum(valid)
        if valid_count == 0:
            return key_change_probs

        # Get inputs and hidden states for valid states
        ktm_hidden_states = self.key_transition_model.init_hidden(valid_count)
        ktm_inputs = [None] * valid_count
        for i, state in enumerate(states[valid]):
            if state.ktm_hidden_state is not None:
                ktm_hidden_states[0][i], ktm_hidden_states[1][i] = state.ktm_hidden_state
            ktm_inputs[i] = state.get_ktm_input()

        # Generate KTM loader
        ktm_dataset = ds.HarmonicDataset()
        ktm_dataset.inputs = ktm_inputs
        ktm_dataset.hidden_states = ktm_hidden_states
        ktm_loader = DataLoader(
            ktm_dataset,
            batch_size=ds.KeyTransitionDataset.valid_batch_size,
            shuffle=False,
        )

        # Run KTM
        outputs = []
        hidden_states = []
        for batch in ktm_loader:
            batch_output, batch_hidden = self.key_transition_model.run_one_step(batch)
            outputs.extend(batch_output)
            hidden_states.extend(batch_hidden)

        # Copy hidden states to valid states
        for state, hidden_state in zip(states[valid], hidden_states):
            state.ktm_hidden_state = hidden_state

        # Copy KTM output probabilities to probs return array
        key_change_probs[valid] = outputs

        return outputs

    def get_key_change_states(self, states: List[State]) -> List[State]:
        """
        Get all states resulting from key changes on the given states using the KSM.

        Parameters
        ----------
        states : List[State]
            The states which will change key.

        Returns
        -------
        new_states : List[State]
            A List of all states resulting from key changes of the given states.
        """
        # Get inputs and hidden states for all states
        ksm_inputs = [state.get_ksm_input() for state in states] * len(states)

        # Generate KSM loader
        ksm_dataset = ds.HarmonicDataset()
        ksm_dataset.inputs = ksm_inputs
        ksm_dataset.input_lengths = [len(ksm_input) for ksm_input in ksm_inputs]
        ksm_loader = DataLoader(
            ksm_dataset,
            batch_size=ds.KeySequenceDataset.valid_batch_size,
            shuffle=False,
        )

        # Run KSM
        key_priors = []
        for batch in ksm_loader:
            priors = self.key_sequence_model.get_output(batch)
            key_priors.extend(priors)

        priors_argsort = np.argsort(-np.array(key_priors))  # Negative to sort descending
        max_indexes = np.clip(
            np.argmax(
                np.cumsum(
                    np.take_along_axis(key_priors, priors_argsort, -1),
                    axis=-1,
                ) >= self.target_key_branch_prob,
                axis=-1,
            ) + 1,
            1,
            self.max_key_branching_factor,
        )

        new_states = []
        for state, log_prior, max_index in zip(states, np.log(key_priors), max_indexes):
            # Ensure each state branches at least once
            if max_index == 1 and priors_argsort[0] == state.key:
                max_index = 2

            # Branch
            for key_id in priors_argsort[:max_index]:
                if key_id == state.key:
                    # Disallow self-transitions
                    continue

                # Calculate the new state on this absolute chord
                new_state = state.key_transition(key_id, log_prior)

                new_states.append(new_state)

        return new_states

    def run_csm_batched(self, states: List[State]):
        """
        Run the CSM batched on the given states, which will have their csm_prior and
        csm_hidden_state fields changed.

        Parameters
        ----------
        states : List[State]
            The states to be run on the csm. These will have their csm_prior and
            csm_hidden_state fields changed.
        """
        # Get inputs and hidden states for all states
        csm_hidden_states = self.chord_sequence_model.init_hidden(len(states))
        csm_inputs = [None] * len(states)
        for i, state in enumerate(states):
            if state.csm_hidden_state is not None:
                csm_hidden_states[0][i], csm_hidden_states[1][i] = state.csm_hidden_state
            csm_inputs[i] = state.get_csm_input()

        # Generate CSM loader
        csm_dataset = ds.HarmonicDataset()
        csm_dataset.inputs = csm_inputs
        csm_dataset.hidden_states = csm_hidden_states
        csm_loader = DataLoader(
            csm_dataset,
            batch_size=ds.ChordSequenceDataset.valid_batch_size,
            shuffle=False,
        )

        # Run CSM
        priors = []
        hidden_states = []
        for batch in csm_loader:
            batch_output, batch_hidden = self.chord_sequence_model.run_one_step(batch)
            priors.extend(batch_output)
            hidden_states.extend(batch_hidden)

        # Update states with new priors and hidden states
        for state, log_prior, hidden in zip(states, np.log(priors), hidden_states):
            state.csm_log_prior = log_prior
            state.csm_hidden_state = hidden
