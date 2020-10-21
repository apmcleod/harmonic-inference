"""Combined models that output a key/chord sequence given an input score, midi, or audio."""
import bisect
import heapq
import itertools
import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import harmonic_inference.data.datasets as ds
import harmonic_inference.models.chord_classifier_models as ccm
import harmonic_inference.models.chord_sequence_models as csm
import harmonic_inference.models.chord_transition_models as ctm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.data_types import KeyMode, PitchType
from harmonic_inference.data.piece import Piece
from harmonic_inference.utils.beam_search_utils import Beam, HashedBeam, State

MODEL_CLASSES = {
    "ccm": ccm.SimpleChordClassifier,
    "ctm": ctm.SimpleChordTransitionModel,
    "csm": csm.SimpleChordSequenceModel,
    "ktm": ktm.SimpleKeyTransitionModel,
    "ksm": ksm.SimpleKeySequenceModel,
}


LABELS = {
    "chords": [],
    "keys": [],
}


def add_joint_model_args(parser: ArgumentParser):
    """
    Add parameters for the HarmonicInferenceModel to the given ArgumentParser.

    Parameters
    ----------
    parser : ArgumentParser
        The ArgumentParser to add the HarmonicInferenceModel arguments to.
    """
    parser.add_argument(
        "--min-chord-change-prob",
        default=0.25,
        type=float,
        help="The minimum CTM probability that can be a chord change.",
    )

    parser.add_argument(
        "--max-no-chord-change-prob",
        default=0.75,
        type=float,
        help="The maximum CTM probability that can be a non-chord change.",
    )

    parser.add_argument(
        "--max-chord-length",
        default=Fraction(8),
        type=Fraction,
        help="The maximum duration (in whole notes) of a chord.",
    )

    parser.add_argument(
        "--min-key-change-prob",
        default=0.25,
        type=float,
        help="The minimum KTM probability that can be a key change.",
    )

    parser.add_argument(
        "--max-no-key-change-prob",
        default=0.75,
        type=float,
        help="The maximum KTM probability that can be a non-key change.",
    )

    parser.add_argument(
        "--beam-size",
        default=50,
        type=int,
        help="The beam size to use during decoding.",
    )

    parser.add_argument(
        "--max-chord-branching-factor",
        default=20,
        type=int,
        help="The maximum number of different chords to branch into.",
    )

    parser.add_argument(
        "--target-chord-branch-prob",
        default=0.95,
        type=float,
        help=(
            "Once the chords branched into account for at least this much probability mass "
            "stop branching, disregarding --max-chord-branching-factor."
        ),
    )

    parser.add_argument(
        "--max-key-branching-factor",
        default=5,
        type=int,
        help="The maximum number of different keys to branch into.",
    )

    parser.add_argument(
        "--target-key-branch-prob",
        default=0.95,
        type=float,
        help=(
            "Once the keys branched into account for at least this much probability mass "
            "stop branching, disregarding --max-key-branching-factor."
        ),
    )

    parser.add_argument(
        "--hash-length",
        default=5,
        type=int,
        help=(
            "If 2 states are identical in chord and key for this many chord changes "
            "(disregarding change index), only the most likely state is kept in the beam."
        ),
    )


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
        beam_size: int = 50,
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
            assert isinstance(
                models[model], model_class
            ), f"`{model}` in models dict is not of type {model_class.__name__}."

        self.chord_classifier = models["ccm"]
        self.chord_sequence_model = models["csm"]
        self.chord_transition_model = models["ctm"]
        self.key_sequence_model = models["ksm"]
        self.key_transition_model = models["ktm"]

        # Ensure all types match
        assert (
            self.chord_classifier.INPUT_TYPE == self.chord_transition_model.INPUT_TYPE
        ), "Chord Classifier input type does not match Chord Transition Model input type"
        assert (
            self.chord_classifier.OUTPUT_TYPE == self.chord_sequence_model.CHORD_TYPE
        ), "Chord Classifier output type does not match Chord Sequence Model chord type"
        assert (
            self.chord_sequence_model.CHORD_TYPE == self.key_transition_model.INPUT_TYPE
        ), "Chord Sequence Model chord type does not match Key Transition Model input type"
        assert (
            self.chord_sequence_model.CHORD_TYPE == self.key_sequence_model.INPUT_TYPE
        ), "Chord Sequence Model chord type does not match Key Transition Model input type"

        # Set joint model types
        self.INPUT_TYPE = self.chord_classifier.INPUT_TYPE
        self.CHORD_OUTPUT_TYPE = self.chord_sequence_model.CHORD_TYPE
        self.KEY_OUTPUT_TYPE = self.key_sequence_model.KEY_TYPE

        # Load labels
        self.LABELS = {
            "chord": [
                (
                    hu.get_pitch_from_string(root, self.CHORD_OUTPUT_TYPE),
                    hu.get_chord_type_from_string(description.split(",")[0]),
                    int(inv),
                )
                for root, description, inv in [
                    chord.split(":")
                    for chord in hu.get_chord_label_list(
                        self.CHORD_OUTPUT_TYPE, use_inversions=True
                    )
                ]
            ],
            "key": [
                (hu.get_pitch_from_string(tonic, self.KEY_OUTPUT_TYPE), KeyMode[mode.split(".")[1]])
                for tonic, mode in [
                    key.split(":") for key in hu.get_key_label_list(self.KEY_OUTPUT_TYPE)
                ]
            ],
            "relative_key": list(
                itertools.product(
                    KeyMode,
                    (
                        range(hc.MIN_KEY_CHANGE_INTERVAL_TPC, hc.MAX_KEY_CHANGE_INTERVAL_TPC)
                        if self.KEY_OUTPUT_TYPE == PitchType.TPC
                        else range(hc.NUM_PITCHES[PitchType.MIDI])
                    ),
                )
            ),
        }

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

        # No piece currently
        self.current_piece = None

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
        self.current_piece = piece

        # Save caches from piece
        self.duration_cache = piece.get_duration_cache()
        self.onset_cache = [vec.onset for vec in piece.get_inputs()] + [
            piece.get_inputs()[-1].offset
        ]
        self.onset_level_cache = [vec.onset_level for vec in piece.get_inputs()] + [
            piece.get_inputs()[-1].offset_level
        ]

        # Get chord change probabilities (with CTM)
        logging.info("Getting chord change probabilities")
        change_probs = self.get_chord_change_probs(piece)

        # Debug log chord change probabilities
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debug_chord_change_probs(change_probs)

        # Calculate valid chord ranges and their probabilities
        logging.info("Calculating valid chord ranges")
        chord_ranges, chord_log_probs = self.get_chord_ranges(piece, change_probs)

        # Calculate chord priors for each possible chord range (batched, with CCM)
        logging.info("Classifying chords")
        chord_classifications = self.get_chord_classifications(piece, chord_ranges)

        # Debug log chord classifications
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debug_chord_classifications(chord_ranges, chord_classifications)

        # Iterative beam search for other modules
        logging.info("Performing iterative beam search")
        state = self.beam_search(
            piece,
            chord_ranges,
            chord_log_probs,
            chord_classifications,
        )

        self.current_piece = None

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
            return batch_output[0][: batch_length[0]].numpy()

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
                    change_probs[start + 1 :],
                    change_log_probs[start + 1 :],
                    no_change_log_probs[start + 1 :],
                    self.duration_cache[start:],  # Off-by-one because cache is dur to next note
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
                )
                >= self.target_chord_branch_prob,
                axis=-1,
            )
            + 1,
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

        # Add initial states
        for key in range(len(hu.get_key_label_list(self.KEY_OUTPUT_TYPE))):
            all_states[0].add(State(key=key, hash_length=self.hash_length))

        for current_start, current_states in tqdm(
            enumerate(all_states[:-1]),
            desc="Beam searching through inputs",
            total=len(all_states) - 1,
        ):
            if len(current_states) == 0:
                continue

            # Run CSM here to avoid running it for invalid states
            if current_start != 0:
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
                        chord_id,
                        range_end,
                        range_log_prob + chord_log_priors[chord_id],
                        self.CHORD_OUTPUT_TYPE,
                        self.LABELS,
                    )

                    if new_state is not None and all_states[range_end].fits_in_beam(
                        new_state,
                        check_hash=False,
                    ):
                        to_check_for_key_change.append(new_state)

            # Check for key changes
            change_probs = self.get_key_change_probs(to_check_for_key_change)
            no_change_log_probs = np.log(1 - change_probs)
            change_log_probs = np.log(change_probs)

            # Branch on key changes
            to_csm_prior_states = []
            to_ksm_states = []
            for change_prob, change_log_prob, no_change_log_prob, state in zip(
                change_probs, change_log_probs, no_change_log_probs, to_check_for_key_change
            ):
                can_not_change = change_prob <= self.max_no_key_change_prob and state.is_valid()
                can_change = change_prob >= self.min_key_change_prob

                # Make a copy only if necessary
                change_state = state.copy() if can_change and can_not_change else state

                if can_change:
                    change_state.log_prob += change_log_prob
                    if all_states[change_state.change_index].fits_in_beam(
                        change_state,
                        check_hash=False,
                    ):
                        to_ksm_states.append(change_state)

                if can_not_change:
                    state.log_prob += no_change_log_prob
                    if all_states[state.change_index].fits_in_beam(state, check_hash=True):
                        to_csm_prior_states.append(state)

            # Change keys and put resulting states into the appropriate beam
            for state in self.get_key_change_states(to_ksm_states):
                all_states[state.change_index].add(state)

            # Add CSM prior and add to beam (CSM is run at the start of each iteration)
            for state in to_csm_prior_states:
                if current_start != 0:
                    state.add_csm_prior(
                        self.CHORD_OUTPUT_TYPE,
                        self.duration_cache,
                        self.onset_cache,
                        self.onset_level_cache,
                        self.LABELS,
                    )

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

        if len(states) == 0:
            return key_change_probs

        # Check for states that can key transition. Only these will update output_prob
        valid_states = [state.can_key_transition() for state in states]

        # Get inputs and hidden states for valid states
        ktm_hidden_states = self.key_transition_model.init_hidden(len(states))
        ktm_inputs = [None] * len(states)
        for i, state in enumerate(states):
            if state.ktm_hidden_state is not None:
                ktm_hidden_states[0][:, i], ktm_hidden_states[1][:, i] = state.ktm_hidden_state
            ktm_inputs[i] = state.get_ktm_input(
                self.CHORD_OUTPUT_TYPE,
                self.duration_cache,
                self.onset_cache,
                self.onset_level_cache,
                self.LABELS,
            )

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
        cell_states = []
        for batch in ktm_loader:
            batch_output, batch_hidden = self.key_transition_model.run_one_step(batch)
            outputs.extend(batch_output.numpy().squeeze(axis=1))
            hidden_states.extend(torch.transpose(batch_hidden[0], 0, 1))
            cell_states.extend(torch.transpose(batch_hidden[1], 0, 1))

        # Copy hidden states to valid states
        for state, hidden, cell in zip(states, hidden_states, cell_states):
            state.ktm_hidden_state = (hidden, cell)

        # Copy KTM output probabilities to probs return array
        key_change_probs[valid_states] = np.array(outputs)[valid_states]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debug_key_transitions(
                key_change_probs[valid_states],
                np.array(states)[valid_states],
            )

        return key_change_probs

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
        if len(states) == 0:
            return []

        # Get inputs and hidden states for all states
        ksm_inputs = [
            state.get_ksm_input(
                self.CHORD_OUTPUT_TYPE,
                self.duration_cache,
                self.onset_cache,
                self.onset_level_cache,
                self.LABELS,
            )
            for state in states
        ]

        # Generate KSM loader
        ksm_dataset = ds.HarmonicDataset()
        ksm_dataset.inputs = ksm_inputs
        ksm_dataset.input_lengths = [len(ksm_input) for ksm_input in ksm_inputs]
        ksm_dataset.targets = np.zeros(len(ksm_inputs))
        ksm_loader = DataLoader(
            ksm_dataset,
            batch_size=ds.KeySequenceDataset.valid_batch_size,
            shuffle=False,
        )

        # Run KSM
        key_priors = []
        for batch in ksm_loader:
            priors = self.key_sequence_model.get_output(batch)
            key_priors.extend(priors.numpy())

        key_priors = np.array(key_priors)
        priors_argsort = np.argsort(-key_priors)  # Negative to sort descending
        max_indexes = np.clip(
            np.argmax(
                np.cumsum(
                    np.take_along_axis(np.exp(key_priors), priors_argsort, -1),
                    axis=-1,
                )
                >= self.target_key_branch_prob,
                axis=-1,
            )
            + 1,
            1,
            self.max_key_branching_factor,
        )

        new_states = []
        for state, log_priors, prior_argsort, max_index in zip(
            states,
            key_priors,
            priors_argsort,
            max_indexes,
        ):
            # Ensure each state branches at least once
            if max_index == 1 and prior_argsort[0] == state.key:
                max_index = 2

            # Branch
            for relative_key_id in prior_argsort[:max_index]:
                mode, interval = self.LABELS["relative_key"][relative_key_id]
                tonic = state.get_key(self.KEY_OUTPUT_TYPE, self.LABELS).relative_tonic + interval

                if self.KEY_OUTPUT_TYPE == PitchType.MIDI:
                    tonic %= 12
                elif tonic < 0 or tonic >= hc.NUM_PITCHES[PitchType.TPC]:
                    # Invalid key tonic
                    continue

                key_id = hu.get_key_one_hot_index(mode, tonic, self.KEY_OUTPUT_TYPE)

                if key_id == state.key:
                    # Disallow self-transitions and illegal keys
                    continue

                # Calculate the new state on this key change
                new_state = state.key_transition(
                    key_id,
                    log_priors[relative_key_id],
                    self.KEY_OUTPUT_TYPE,
                    self.LABELS,
                )

                if new_state is not None:
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
                csm_hidden_states[0][:, i], csm_hidden_states[1][:, i] = state.csm_hidden_state
            csm_inputs[i] = state.get_csm_input(
                self.CHORD_OUTPUT_TYPE,
                self.duration_cache,
                self.onset_cache,
                self.onset_level_cache,
                self.LABELS,
            )

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
        cell_states = []
        for batch in csm_loader:
            batch_output, batch_hidden = self.chord_sequence_model.run_one_step(batch)
            priors.extend(batch_output)
            hidden_states.extend(torch.transpose(batch_hidden[0], 0, 1))
            cell_states.extend(torch.transpose(batch_hidden[1], 0, 1))

        # Update states with new priors and hidden states
        for state, log_prior, hidden, cell in zip(states, priors, hidden_states, cell_states):
            state.csm_log_prior = log_prior.numpy()[0]
            state.csm_hidden_state = (hidden, cell)

    def debug_chord_change_probs(self, change_probs: List[float]):
        """
        Log chord change probabilities to as debug messages.

        Parameters
        ----------
        change_probs : List[float]
            The chord change probability for each input.
        """
        for i, change_prob in enumerate(change_probs):
            if i in self.current_piece.get_chord_change_indices():
                if change_prob < 0.5:
                    logging.debug(
                        "Piece changes chord on index %s but change_prob=%s",
                        i,
                        change_prob,
                    )
            else:
                if change_prob > 0.5:
                    logging.debug(
                        "Piece doesn't change chord on index %s but change_prob=%s",
                        i,
                        change_prob,
                    )

    def debug_chord_classifications(
        self, chord_ranges: List[Tuple[int, int]], chord_classifications: List[float]
    ):
        """
        Log chord classifications as debug messages.

        Parameters
        ----------
        chord_ranges : List[Tuple[int, int]]
            A list of the chord ranges that were classified.
        chord_classifications : List[List[float]]
            The log_probability of each chord for each given range.
        """
        change_indices = self.current_piece.get_chord_change_indices()

        for range, chord_probs in zip(chord_ranges, np.exp(chord_classifications)):
            range_start, range_end = range

            correct_chords = self.current_piece.get_chords_within_range(range_start, range_end)
            correct_chords_one_hot = [
                chord.get_one_hot_index(relative=False, use_inversion=True)
                for chord in correct_chords
            ]

            rankings = list(np.argsort(-chord_probs))
            correct_probs = [chord_probs[one_hot] for one_hot in correct_chords_one_hot]
            correct_rank = [rankings.index(one_hot) for one_hot in correct_chords_one_hot]

            is_range_correct = (
                len(correct_chords) == 1
                and range_start in change_indices
                and range_end in change_indices
            )
            is_classification_correct = is_range_correct and correct_rank[0] == 0
            is_any_classification_correct = min(correct_rank) == 0

            if is_any_classification_correct and not is_range_correct:
                continue

            correct_string = (
                "=== " if is_classification_correct else "*** " if is_range_correct else ""
            )

            logging.debug("%sChord classification results for range %s:", correct_string, range)
            logging.debug(
                "    correct chords: %s",
                "; ".join(
                    np.array(hu.get_chord_label_list(self.CHORD_OUTPUT_TYPE))[
                        correct_chords_one_hot
                    ]
                ),
            )
            for one_hot, prob, rank in zip(correct_chords_one_hot, correct_probs, correct_rank):
                logging.debug(
                    "        p(%s)=%s, rank=%s",
                    hu.get_chord_label_list(self.CHORD_OUTPUT_TYPE)[one_hot],
                    prob,
                    rank,
                )

            logging.debug("    Top chords:")
            for rank, one_hot in enumerate(
                rankings[: min(self.max_chord_branching_factor, max(correct_rank) + 1)]
            ):
                logging.debug(
                    "       %s%s: p(%s) = %s",
                    "*" if one_hot in correct_chords_one_hot else " ",
                    rank,
                    hu.get_chord_label_list(self.CHORD_OUTPUT_TYPE)[one_hot],
                    chord_probs[one_hot],
                )

    def debug_key_transitions(self, key_change_probs: List[float], states: List[State]):
        """
        Log key transitions as debug messages.

        Parameters
        ----------
        key_change_probs : List[float]
            The probability of a transition for each State.
        states : List[State]
            The States.
        """
        key_changes = self.current_piece.get_key_change_input_indices()
        keys = self.current_piece.get_keys()

        for change_prob, state in zip(key_change_probs, states):
            change_index = state.prev_state.change_index
            state_key = state.get_key(self.KEY_OUTPUT_TYPE, self.LABELS)

            new_key_index = bisect.bisect_left(key_changes, change_index)
            correct_key = keys[new_key_index - 1]
            if not state_key.equals(correct_key, use_relative=True):
                # Skip if current state's key is incorrect
                continue

            if key_changes[new_key_index] == change_index:
                logging.debug(
                    "Key change at index %s: %s -> %s",
                    change_index,
                    correct_key,
                    keys[new_key_index],
                )

            else:
                logging.debug(
                    "No key change at index %s: %s",
                    change_index,
                    correct_key,
                )

            logging.debug("   Current key: %s", state_key)
            logging.debug(
                "    Recent chords: %s",
                "; ".join(
                    [
                        str(
                            s.get_chord(
                                self.CHORD_OUTPUT_TYPE,
                                self.duration_cache,
                                self.onset_cache,
                                self.onset_level_cache,
                                self.LABELS,
                            )
                        )
                        for s in [state, state.prev_state]
                    ]
                ),
            )
            logging.debug("        p(change) = %s", change_prob)


def from_args(models: Dict, ARGS: Namespace) -> HarmonicInferenceModel:
    """
    Load a HarmonicInferenceModel from this given models and argparse parsed arguments.

    Parameters
    ----------
    models : Dict
        A dictionary mapping of model components:
            'ccm': A ChordClassifier
            'ctm': A ChordTransitionModel
            'csm': A ChordSequenceModel
            'ktm': A KeyTransitionModel
            'ksm': A KeySequenceModel
    ARGS : Namespace
        Parsed command-line arguments for the HarmonicInferenceModel's parameters.

    Returns
    -------
    model : HarmonicInferenceModel
        A HarmonicInferenceModel, with parameters taken from the parsed args.
    """
    return HarmonicInferenceModel(
        models,
        min_chord_change_prob=ARGS.min_chord_change_prob,
        max_no_chord_change_prob=ARGS.max_no_chord_change_prob,
        max_chord_length=ARGS.max_chord_length,
        min_key_change_prob=ARGS.min_key_change_prob,
        max_no_key_change_prob=ARGS.max_no_key_change_prob,
        beam_size=ARGS.beam_size,
        max_chord_branching_factor=ARGS.max_chord_branching_factor,
        target_chord_branch_prob=ARGS.target_chord_branch_prob,
        max_key_branching_factor=ARGS.max_key_branching_factor,
        target_key_branch_prob=ARGS.target_key_branch_prob,
        hash_length=ARGS.hash_length,
    )
