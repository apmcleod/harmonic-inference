"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
from fractions import Fraction
from typing import List, Tuple, Dict
import logging
import heapq
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
from harmonic_inference.data.piece import Chord, Key, Piece
import harmonic_inference.data.datasets as ds


MODEL_CLASSES = {
    'ccm': ccm.SimpleChordClassifier,
    'ctm': ctm.SimpleChordTransitionModel,
    'csm': csm.SimpleChordSequenceModel,
    'ktm': ktm.SimpleKeyTransitionModel,
    'ksm': ksm.SimpleKeySequenceModel,
}


class State:
    """
    The state used during the model's beam search.
    """
    def __init__(
        self,
        chord: int = None,
        key: int = None,
        change_index: int = 0,
        log_prob: float = 0.0,
        prev_state = None,
    ):
        """
        [summary]

        Parameters
        ----------
        chord : int
            [description]
        key : int
            [description]
        change_index : int
            [description]
        log_prob : float
            [description]
        prev_state : [type]
            [description]
        """
        self.chord = chord
        self.key = key
        self.change_index = change_index

        self.log_prob = log_prob

        self.prev_state = prev_state

    def chord_transition(self, chord: int, change_index: int, log_prob: float):
        """
        [summary]

        Parameters
        ----------
        chord : int
            [description]
        change_index : int
            [description]
        log_prob : float
            [description]

        Returns
        -------
        new_state : State
            [description]
        """
        return State(
            chord=chord,
            key=self.key,
            change_index=change_index,
            log_prob=self.log_prob + log_prob,
            prev_state=self,
        )

    def can_key_transition(self) -> bool:
        """
        [summary]

        Returns
        -------
        can_transition : bool
            [description]
        """
        return self.prev_state is not None and self.prev_state.prev_state is not None

    def key_transition(self, key: int, log_prob: float):
        """
        [summary]

        Parameters
        ----------
        key : int
            [description]
        log_prob : float
            [description]

        Returns
        -------
        new_state : State
            [description]
        """
        return State(
            chord=self.chord,
            key=key,
            change_index=self.change_index,
            log_prob=self.prev_state.log_prob + log_prob,
            prev_state=self.prev_state,
        )

    def get_chords(self) -> Tuple[List[int], List[int]]:
        """
        [summary]

        Returns
        -------
        chords : List[int]
            [description]
        change_indexes : List[int]
            [description]
        """
        if self.prev_state is None:
            return [], [self.change_index]

        chords, changes = self.prev_state.get_chords()
        chords.append(self.chord)
        changes.append(self.change_index)

        return chords, changes

    def get_keys(self) -> Tuple[List[int], List[int]]:
        """
        [summary]

        Returns
        -------
        keys : List[int]
            [description]
        change_indexes : List[int]
            [description]
        """
        if self.prev_state is None:
            return [], [self.change_index]

        keys, changes = self.prev_state.get_keys()
        if self.key != keys[-1]:
            keys.append(self.key)
            changes.append(self.change_index)

        return keys, changes

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
        return len(self.beam) < self.beam_size or self.beam[0] < state

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
        if len(self.beam) == self.beam_size:
            if self.beam[0] < state:
                heapq.heappushpop(self.beam, state)
                return True
            return False
        else:
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
        return max(self.beam)

    def empty(self):
        """
        Empty this beam.
        """
        self.beam = []

    def __iter__(self):
        return self.beam.__iter__()


class HarmonicInferenceModel:
    """
    A model to perform harmonic inference on an input score, midi, or audio piece.
    """
    def __init__(
        self,
        models: Dict,
        min_change_prob: float = 0.25,
        max_no_change_prob: float = 0.75,
        max_chord_length: Fraction = Fraction(8),
        beam_size: int = 500,
        max_branching_factor: int = 20,
        target_branch_prob: float = 0.95,
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
        min_change_prob : float
            The minimum probability (from the CTM) on which a chord change can occur.
        max_no_change_prob : float
            The maximum probability (from the CTM) on which a chord is allowed not
            to change.
        max_chord_length : Fraction
            The maximum length for a chord generated by this model.
        beam_size : int
            The beam size to use for decoding with this model.
        max_branching_factor : int
            For each state during the beam search, the maximum number of different chord
            classifications to try during branching. Each of these will be potentially checked
            for key change as well.
        target_branch_prob : float
            Once the branches transitioned into account for at least this much probability mass,
            no more branches are searched, even if the max_branching_factor has not yet been
            reached.
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

        # Save other params
        assert min_change_prob <= max_no_change_prob, (
            "Undefined chord change behavior on probability range "
            f"({max_no_change_prob}, {min_change_prob})"
        )
        self.min_change_prob = min_change_prob
        self.max_no_change_prob = max_no_change_prob
        self.max_chord_length = max_chord_length

        # Beam search params
        self.beam_size = beam_size
        self.max_branching_factor = max_branching_factor
        self.target_branch_prob = target_branch_prob

    def get_harmony(
        self,
        piece: Piece,
    ) -> Tuple[List[Tuple[int, Chord]], List[Tuple[int, Key]]]:
        """
        Run the model on a piece and output its harmony.

        Parameters
        ----------
        piece : Piece
            A Piece to perform harmonic inference on.

        Returns
        -------
        chords : List[Tuple[int, Chord]]
            A List of (index, Chord) tuples containing the Chords of the piece and the
            indexes at which each starts.
        keys : List[Tuple[int, Key]]
            A List of (index, Key) tuples containing the Keys of the piece and the
            indexes at which each starts.
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

        return state.get_chords(), state.get_keys()

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

                if change_prob > self.min_change_prob:
                    # Chord change can occur
                    chord_ranges.append((start, index))
                    chord_log_probs.append(running_log_prob + change_log_prob)

                    if not in_starts[index]:
                        heapq.heappush(starts, index)
                        in_starts[index] = True

                    if change_prob > self.max_no_change_prob:
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
                ) >= self.target_branch_prob,
                axis=-1,
            ) + 1,
            1,
            self.max_branching_factor,
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

        all_states = [Beam(self.beam_size) for _ in range(len(piece.get_inputs()) + 1)]
        all_states[0].add(State())

        for current_start, current_states in tqdm(
            enumerate(all_states[:-1]),
            desc="Beam searching through inputs",
            total=len(all_states) - 1,
        ):
            for state in current_states:
                for range_data in chord_ranges_dict[current_start]:
                    (
                        range_end,
                        range_log_prob,
                        chord_log_priors,
                        chord_priors_argsort,
                        max_index,
                     ) = range_data

                    next_beam = all_states[range_end]
                    if not next_beam.fits_in_beam(state):
                        continue

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

                        if not next_beam.fits_in_beam(new_state):
                            # No need to check for key-relative chord and/or key change
                            continue

                        # TODO: Check relative chord and/or key change
                        next_beam.add(new_state)

            current_states.empty()

        return all_states[-1].get_top_state()
