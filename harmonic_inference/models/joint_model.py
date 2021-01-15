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
import harmonic_inference.models.initial_chord_models as icm
import harmonic_inference.models.key_sequence_models as ksm
import harmonic_inference.models.key_transition_models as ktm
import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.data_types import KeyMode, PitchType
from harmonic_inference.data.piece import Piece, get_range_start
from harmonic_inference.utils.beam_search_utils import Beam, HashedBeam, State

MODEL_CLASSES = {
    "ccm": ccm.SimpleChordClassifier,
    "ctm": ctm.SimpleChordTransitionModel,
    "csm": csm.SimpleChordSequenceModel,
    "ktm": ktm.SimpleKeyTransitionModel,
    "ksm": ksm.SimpleKeySequenceModel,
    "icm": icm.SimpleInitialChordModel,
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

    parser.add_argument(
        "--ksm-exponent",
        default=1.0,
        type=float,
        help=(
            "An exponent to be applied to the KSM's probability outputs. Used to weight "
            "the KSM and CSM equally even given their different vocabulary sizes."
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
        ksm_exponent: float = 1.0,
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
                'icm': An InitialChordModel
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
            If not None, a hashed beam is used, where only 1 State is kept in the Beam.
        ksm_exponent : float
            An exponent to apply to the KSM's output probabilities. Used to weight the KSM
            and CSM equally, even given their different vocabulary sizes.
        """
        for model, model_class in MODEL_CLASSES.items():
            assert model in models.keys(), f"`{model}` not in models dict."
            assert isinstance(
                models[model], model_class
            ), f"`{model}` in models dict is not of type {model_class.__name__}."

        self.chord_classifier: ccm.ChordClassifierModel = models["ccm"]
        self.chord_sequence_model: csm.ChordSequenceModel = models["csm"]
        self.chord_transition_model: ctm.ChordTransitionModel = models["ctm"]
        self.key_sequence_model: ksm.KeySequenceModel = models["ksm"]
        self.key_transition_model: ktm.KeyTransitionModel = models["ktm"]
        self.initial_chord_model: icm.SimpleInitialChordModel = models["icm"]
        self.check_input_output_types()

        # Set joint model types
        self.INPUT_TYPE = self.chord_classifier.INPUT_TYPE
        self.CHORD_OUTPUT_TYPE = self.chord_sequence_model.OUTPUT_PITCH_TYPE
        self.KEY_OUTPUT_TYPE = self.key_sequence_model.OUTPUT_PITCH_TYPE

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
        self.ksm_exponent = ksm_exponent

        # Beam search params
        self.beam_size = beam_size
        self.hash_length = hash_length

        # No piece currently
        self.current_piece = None
        self.debugger = None
        self.duration_cache = None
        self.onset_cache = None
        self.onset_level_cache = None

    def check_input_output_types(self):
        """
        Check input and output types of all models to be sure they are compatible.
        """
        # Input vectors
        assert (
            self.chord_classifier.INPUT_TYPE == self.chord_transition_model.INPUT_TYPE
        ), "CCM input type does not match CTM input type"
        assert (
            self.chord_classifier.INPUT_PITCH == self.chord_transition_model.PITCH_TYPE
        ), "CCM input pitch type does not match CTM pitch type"

        # Output from CCM
        assert (
            self.chord_classifier.OUTPUT_PITCH == self.chord_sequence_model.INPUT_CHORD_PITCH_TYPE
        ), "CCM output pitch type does not match CSM chord pitch type"

        # Output from CSM
        assert (
            self.chord_sequence_model.OUTPUT_PITCH_TYPE
            == self.key_sequence_model.INPUT_CHORD_PITCH_TYPE
        ), "CSM output pitch type does not match KSM input chord pitch type"
        assert (
            self.chord_sequence_model.OUTPUT_PITCH_TYPE
            == self.key_sequence_model.INPUT_CHORD_PITCH_TYPE
        ), "CSM output pitch type does not match KSM input chord pitch type"
        assert (
            self.chord_sequence_model.OUTPUT_PITCH_TYPE
            == self.chord_sequence_model.INPUT_CHORD_PITCH_TYPE
        ), "CSM output pitch type does not match its own input chord pitch type"

        # Output from KSM
        assert (
            self.chord_sequence_model.INPUT_KEY_PITCH_TYPE
            == self.key_sequence_model.OUTPUT_PITCH_TYPE
        ), "CSM input key pitch type does not match KSM output pitch type"
        assert (
            self.key_transition_model.INPUT_KEY_PITCH_TYPE
            == self.key_sequence_model.OUTPUT_PITCH_TYPE
        ), "KTM input key pitch type does not match KSM output pitch type"
        assert (
            self.key_sequence_model.INPUT_KEY_PITCH_TYPE
            == self.key_sequence_model.OUTPUT_PITCH_TYPE
        ), "KSM input key pitch type does not match its own input pitch type"

        # Output from ICM
        assert (
            self.initial_chord_model.PITCH_TYPE == self.chord_sequence_model.OUTPUT_PITCH_TYPE
        ), "ICM pitch type does not match CSM output pitch type"

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

        self.debugger = DebugLogger(
            self.current_piece,
            self.CHORD_OUTPUT_TYPE,
            self.KEY_OUTPUT_TYPE,
            self.max_chord_branching_factor,
            self.max_key_branching_factor,
        )

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
        change_probs = self.get_chord_change_probs()

        # Debug log chord change probabilities
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debugger.debug_chord_change_probs(change_probs)

        # Calculate valid chord ranges and their probabilities
        logging.info("Calculating valid chord ranges")
        chord_ranges, range_log_probs = self.get_chord_ranges(change_probs)

        # Convert range starting points to new starts based on the note offsets
        chord_change_indices = [start for start, _ in chord_ranges]
        chord_windows = [
            (get_range_start(piece.get_inputs()[start].onset, piece.get_inputs()), end)
            for start, end in chord_ranges
        ]

        # Calculate chord priors for each possible chord range (batched, with CCM)
        logging.info("Classifying chords")
        chord_classifications = self.get_chord_classifications(chord_windows, chord_change_indices)

        # Debug log chord classifications
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debugger.debug_chord_classifications(chord_ranges, chord_classifications)

        # Iterative beam search for other modules
        logging.info("Performing iterative beam search")
        state = self.beam_search(chord_ranges, range_log_probs, chord_classifications)

        self.current_piece = None

        return state

    def get_chord_change_probs(self) -> List[float]:
        """
        Get the Chord Transition Model's outputs for the current piece.

        Returns
        -------
        change_probs : List[float]
            A List of the chord change probability on each input of the given Piece.
        """
        ctm_dataset = ds.ChordTransitionDataset([self.current_piece])
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
        change_probs: List[float],
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Get all possible chord ranges and their log-probability, given the chord change
        probabilities for each input in the Piece.

        Parameters
        ----------
        change_probs : List[float]
            The probability of a chord change occurring on each input of the Piece.

        Returns
        -------
        chord_ranges : List[Tuple[int, int]]
            A List of possible chord ranges, as (start, end) tuples representing the start
            (inclusive) and end (exclusive) points of each possible range.
        range_log_probs : List[float]
            For each chord range, it's log-probability, including its end change, but not its
            start change.
        """
        chord_ranges = []
        range_log_probs = []

        # Invalid masks all but first note at each onset position
        invalid = np.roll(self.duration_cache == 0, 1)
        invalid[0] = True

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
            # Special case for start == 0 because it is always invalid, but can have duration > 0
            running_duration = self.duration_cache[start] if start == 0 else Fraction(0)
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
                    range_log_probs.append(running_log_prob + change_log_prob)

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
                range_log_probs.append(running_log_prob)

        return chord_ranges, range_log_probs

    def get_chord_classifications(
        self,
        ranges: List[Tuple[int, int]],
        change_indices: List[int],
    ) -> List[np.array]:
        """
        Generate a chord type prior for each potential chord (from ranges).

        Parameters
        ----------
        ranges : List[Tuple[int, int]]
            A List of all possible chord ranges as (start, end) for the Piece.
        change_indices : List[int]
            The change index for each of the given ranges.

        Returns
        -------
        classifications : List[np.array]
            The prior log-probability over all chord symbols for each given range.
        """
        ccm_dataset = ds.ChordClassificationDataset(
            [self.current_piece],
            ranges=[ranges],
            change_indices=[change_indices],
            dummy_targets=True,
        )
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
        chord_ranges: List[Tuple[int, int]],
        range_log_probs: List[float],
        chord_classifications: List[np.array],
    ) -> State:
        """
        Perform a beam search over the given Piece to label its Chords and Keys.

        Parameters
        ----------
        chord_ranges : List[Tuple[int, int]]
            A List of possible chord ranges, as (start, end) tuples.
        range_log_probs : List[float]
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
            range_log_probs,
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
        all_states = [
            beam_class(self.beam_size) for _ in range(len(self.current_piece.get_inputs()) + 1)
        ]

        # Add initial states
        for key in range(len(hu.get_key_label_list(self.KEY_OUTPUT_TYPE))):
            key_mode = self.LABELS["key"][key][1]
            state = State(key=key, hash_length=self.hash_length)
            state.csm_log_prior = self.initial_chord_model.get_prior(
                key_mode == KeyMode.MINOR, log=True
            )
            all_states[0].add(state, force=True)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debugger.debug_initial_chord_prior(
                self.initial_chord_model.get_prior(
                    KeyMode.MINOR == self.current_piece.get_keys()[0].relative_mode, log=False
                )
            )

        for current_start, current_states in tqdm(
            enumerate(all_states[:-1]),
            desc="Beam searching through inputs",
            total=len(all_states) - 1,
        ):
            if len(current_states) == 0:
                continue

            logging.debug("Index = %s; num_states = %s", current_start, len(current_states))

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
                range_length = range_end - current_start

                if (
                    current_start != 0
                    and len(all_states[range_end]) > 0
                    and all_states[range_end].beam[0].log_prob >= state.log_prob + range_log_prob
                ):
                    # Quick exit if range will never get into resulting beam
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
                        chord_id,
                        range_end,
                        range_log_prob + chord_log_priors[chord_id] * range_length,
                        self.CHORD_OUTPUT_TYPE,
                        self.LABELS,
                    )

                    if new_state is not None:
                        if current_start == 0 or all_states[range_end].fits_in_beam(
                            new_state,
                            check_hash=False,
                        ):
                            to_check_for_key_change.append(new_state)
                        else:
                            # No other chord will fit in beam
                            break

            # Check for key changes
            change_probs = self.get_key_change_probs(to_check_for_key_change)
            with np.errstate(divide="ignore"):
                no_change_log_probs = np.log(1 - change_probs)
                change_log_probs = np.log(change_probs)

            # Branch on key changes
            to_csm_prior_states = []
            to_ksm_states = []
            for change_prob, change_log_prob, no_change_log_prob, state in zip(
                change_probs, change_log_probs, no_change_log_probs, to_check_for_key_change
            ):
                range_length = state.change_index - state.prev_state.change_index
                can_not_change = change_prob <= self.max_no_key_change_prob and state.is_valid(
                    check_key=True
                )
                can_change = change_prob >= self.min_key_change_prob

                # Make a copy only if necessary
                change_state = state.copy() if can_change and can_not_change else state

                if can_change:
                    change_state.log_prob += change_log_prob * range_length
                    if current_start == 0 or all_states[change_state.change_index].fits_in_beam(
                        change_state,
                        check_hash=False,
                    ):
                        to_ksm_states.append(change_state)

                if can_not_change:
                    state.log_prob += no_change_log_prob * range_length
                    if current_start == 0 or all_states[state.change_index].fits_in_beam(
                        state, check_hash=True
                    ):
                        to_csm_prior_states.append(state)

            # Change keys and put resulting states into the appropriate beam
            changed_key_states = self.get_key_change_states(to_ksm_states)
            # Run the KTM once, and KSM up to the current frame, to update hidden states
            self.get_key_change_probs(changed_key_states, hidden_only=True)
            self.get_key_change_states(changed_key_states, hidden_only=True)
            for state in changed_key_states:
                all_states[state.change_index].add(state)

            # Debug all csm priors
            if current_start != 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                self.debugger.debug_chord_sequence_priors(to_csm_prior_states)

            # Add CSM prior and add to beam (CSM is run at the start of each iteration)
            for state in to_csm_prior_states:
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

    def get_key_change_probs(self, states: List[State], hidden_only: bool = False) -> np.array:
        """
        Get the probability of a key change for each of the given states.

        Parameters
        ----------
        states : List[State]
            The States to check for key changes. Some internal fields of these states may change.
        hidden_only : bool
            True to only update the given states' ktm_hidden_states, and return nothing.

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

        # Get inputs and hidden states for all states (even invalid states need to be run)
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
        ktm_dataset.set_hidden_states(ktm_hidden_states)
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

        # Copy hidden states to all states
        for state, hidden, cell in zip(states, hidden_states, cell_states):
            state.ktm_hidden_state = (hidden, cell)

        if hidden_only:
            return None

        # Copy KTM output probabilities to probs return array
        key_change_probs[valid_states] = np.array(outputs)[valid_states]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debugger.debug_key_transitions(
                key_change_probs[valid_states],
                np.array(states)[valid_states],
            )

        return key_change_probs

    def get_key_change_states(self, states: List[State], hidden_only: bool = False) -> List[State]:
        """
        Get all states resulting from key changes on the given states using the KSM.

        Parameters
        ----------
        states : List[State]
            The states which will change key.
        hidden_only : bool
            True to only update the given states' ktm_hidden_states, and return nothing.

        Returns
        -------
        new_states : List[State]
            A List of all states resulting from key changes of the given states.
        """
        if len(states) == 0:
            return []

        # Get inputs and hidden states for valid states
        ksm_hidden_states = self.key_sequence_model.init_hidden(len(states))
        ksm_inputs = [None] * len(states)
        for i, state in enumerate(states):
            if state.ksm_hidden_state is not None:
                ksm_hidden_states[0][:, i], ksm_hidden_states[1][:, i] = state.ksm_hidden_state
            ksm_inputs[i] = state.get_ksm_input(
                self.CHORD_OUTPUT_TYPE,
                self.duration_cache,
                self.onset_cache,
                self.onset_level_cache,
                self.LABELS,
            )

        # Generate KSM loader
        ksm_dataset = ds.HarmonicDataset()
        ksm_dataset.inputs = ksm_inputs
        ksm_dataset.set_hidden_states(ksm_hidden_states)
        ksm_dataset.input_lengths = [len(ksm_input) for ksm_input in ksm_inputs]
        ksm_dataset.targets = np.zeros(len(ksm_inputs))
        ksm_loader = DataLoader(
            ksm_dataset,
            batch_size=ds.KeySequenceDataset.valid_batch_size,
            shuffle=False,
        )

        # Run KSM
        key_log_priors = []
        hidden_states = []
        cell_states = []
        for batch in ksm_loader:
            batch_output, batch_hidden = self.key_sequence_model.get_output(batch)
            key_log_priors.extend(batch_output.numpy())
            hidden_states.extend(torch.transpose(batch_hidden[0], 0, 1))
            cell_states.extend(torch.transpose(batch_hidden[1], 0, 1))

        # Copy hidden states to valid states
        for state, hidden, cell in zip(states, hidden_states, cell_states):
            state.ksm_hidden_state = (hidden, cell)

        if hidden_only:
            return None

        key_log_priors = np.array(key_log_priors)
        priors_argsort = np.argsort(-key_log_priors)  # Negative to sort descending
        max_indexes = np.clip(
            np.argmax(
                np.cumsum(
                    np.take_along_axis(np.exp(key_log_priors), priors_argsort, -1),
                    axis=-1,
                )
                >= self.target_key_branch_prob,
                axis=-1,
            )
            + 1,
            1,
            self.max_key_branching_factor,
        )

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debugger.debug_key_sequences(key_log_priors, states)

        new_states = []
        for state, log_priors, prior_argsort, max_index in zip(
            states,
            key_log_priors,
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
                range_length = state.change_index - state.prev_state.change_index
                new_state = state.key_transition(
                    key_id,
                    log_priors[relative_key_id] * self.ksm_exponent * range_length,
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
        csm_dataset.set_hidden_states(csm_hidden_states)
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


class DebugLogger:
    """
    A DebugLogger will print debug messages for various model outputs.
    """

    def __init__(
        self,
        piece: Piece,
        chord_type: PitchType,
        key_type: PitchType,
        max_chords_to_print: int = 20,
        max_keys_to_print: int = 5,
    ):
        """
        Create a new debug log printer for a particular piece.

        Parameters
        ----------
        piece : Piece
            The piece to print debug messages for.
        chord_type : PitchType
            The pitch type for chord root labels and priors.
        key_type : PitchType
            The pitch type for key tonic labels and priors.
        max_chords_to_print : int
            The maximum number of chords to print, until the correct chord.
        max_keys_to_print : int
            The maximum number of keys to print, until the correct key.
        """
        self.piece = piece
        self.CHORD_OUTPUT_TYPE = chord_type
        self.KEY_OUTPUT_TYPE = key_type
        self.max_chords_to_print = max_chords_to_print
        self.max_keys_to_print = max_keys_to_print

        self.chord_labels = np.array(hu.get_chord_label_list(self.CHORD_OUTPUT_TYPE))
        self.key_labels = np.array(hu.get_key_label_list(self.KEY_OUTPUT_TYPE))

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
        if len(states) == 0:
            return

        key_changes = self.piece.get_key_change_input_indices()
        keys = self.piece.get_keys()
        chord_changes = self.piece.get_chord_change_indices()
        chords = self.piece.get_chords()

        # Previous change is the same for all states here
        change_index = states[0].prev_state.change_index
        new_key_index = bisect.bisect_left(key_changes, change_index)
        new_chord_index = bisect.bisect_left(chord_changes, change_index)
        correct_key = keys[new_key_index - 1]

        if chord_changes[new_chord_index] == change_index:
            if new_chord_index == len(chords):
                new_chord_index -= 1
            correct_chords = [chords[new_chord_index - 1], chords[new_chord_index]]
        else:
            correct_chords = [chords[new_chord_index - 1]]

        if new_key_index < len(key_changes) and key_changes[new_key_index] == change_index:
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
        logging.debug(
            "  Recent correct chords: %s",
            "; ".join(
                self.chord_labels[
                    [
                        chord.get_one_hot_index(relative=False, use_inversion=True, pad=False)
                        for chord in correct_chords
                    ]
                ]
            ),
        )

        total = 0
        total_correct = 0

        for change_prob, state in zip(key_change_probs, states):
            if correct_key.get_one_hot_index() != state.key:
                # Skip if current state's key is incorrect
                continue

            if (
                correct_chords[-1].get_one_hot_index(relative=False, use_inversion=True, pad=False)
                != state.chord
            ):
                # Skip if current state's chord is incorrect
                continue

            total += 1

            if new_key_index < len(key_changes) and key_changes[new_key_index] == change_index:
                if change_prob > 0.5:
                    total_correct += 1
                    continue

            else:
                if change_prob < 0.5:
                    total_correct += 1
                    continue

            logging.debug("    Current key: %s", self.key_labels[state.key])
            logging.debug(
                "    Recent chords: %s",
                "; ".join([self.chord_labels[s.chord] for s in [state.prev_state, state]]),
            )
            logging.debug("        p(change) = %s", change_prob)

        logging.debug("KTM accuracy")
        if total > 0:
            logging.debug(
                "    Correct key transitions / correct key/chord states: %s / %s = %s",
                total_correct,
                total,
                total_correct / total,
            )
        else:
            logging.debug("    No correct key/chord states")

    def debug_key_sequences(self, key_log_priors: List[List[float]], states: List[State]):
        """
        Log key sequence outputs as debug messages.

        Parameters
        ----------
        key_log_priors : List[List[float]]
            The prior for the next key for each State.
        states : List[State]
            The States.
        """
        if len(states) == 0:
            return

        key_changes = self.piece.get_key_change_input_indices()
        keys = self.piece.get_keys()
        chord_changes = self.piece.get_chord_change_indices()
        chords = self.piece.get_chords()

        # Previous change is the same for all states here
        change_index = states[0].prev_state.change_index
        new_key_index = bisect.bisect_left(key_changes, change_index)
        new_chord_index = bisect.bisect_left(chord_changes, change_index)

        if new_key_index == len(key_changes) or key_changes[new_key_index] != change_index:
            # Piece doesn't have a key change here
            return

        correct_prev_key = keys[new_key_index - 1]
        correct_next_key = keys[new_key_index]
        correct_key_change_one_hot = correct_prev_key.get_key_change_one_hot_index(correct_next_key)

        if chord_changes[new_chord_index] == change_index:
            if new_chord_index == len(chords):
                new_chord_index -= 1
            correct_chords = [chords[new_chord_index - 1], chords[new_chord_index]]
        else:
            correct_chords = [chords[new_chord_index - 1]]

        logging.debug(
            "Key change at index %s: %s -> %s",
            change_index,
            correct_prev_key,
            correct_next_key,
        )
        logging.debug(
            "  Recent correct chords: %s",
            "; ".join(
                self.chord_labels[
                    [
                        chord.get_one_hot_index(relative=False, use_inversion=True, pad=False)
                        for chord in correct_chords
                    ]
                ]
            ),
        )

        total = 0
        total_chord_correct = 0
        total_correct = 0
        total_chord_correct_correct = 0

        relative_key_labels = hu.get_key_label_list(
            self.KEY_OUTPUT_TYPE,
            relative=True,
            relative_to=correct_prev_key.relative_tonic,
        )

        for state, key_prior in zip(states, np.exp(key_log_priors)):
            if correct_prev_key.get_one_hot_index() != state.key:
                # Skip if state's key is incorrect
                continue

            is_chord_correct = False
            if (
                correct_chords[-1].get_one_hot_index(relative=False, use_inversion=True, pad=False)
                == state.chord
            ):
                # State's chord is incorrect
                is_chord_correct = True
                total_chord_correct += 1

            total += 1

            rankings = list(np.argsort(-key_prior))
            correct_prob = key_prior[correct_key_change_one_hot]
            correct_rank = rankings.index(correct_key_change_one_hot)

            if correct_rank == 0:
                if is_chord_correct:
                    total_chord_correct_correct += 1
                total_correct += 1
                continue

            logging.debug("    Current key: %s", self.key_labels[state.key])
            logging.debug(
                "    Recent chords: %s",
                "; ".join([self.chord_labels[s.chord] for s in [state.prev_state, state]]),
            )
            logging.debug(
                "      p(%s)=%s, rank=%s",
                correct_next_key,
                correct_prob,
                correct_rank,
            )

            logging.debug("      Top keys:")
            for rank, one_hot in enumerate(
                rankings[: min(self.max_keys_to_print, correct_rank + 1)]
            ):
                logging.debug(
                    "         %s%s: p(%s) = %s",
                    "*" if one_hot == correct_key_change_one_hot else " ",
                    rank,
                    relative_key_labels[one_hot],
                    key_prior[one_hot],
                )

        logging.debug("KSM accuracy")
        if total > 0:
            logging.debug(
                "    Correct key states accuracy: %s / %s = %s",
                total_correct,
                total,
                total_correct / total,
            )
        else:
            logging.debug("    No correct key states")

        if total_chord_correct > 0:
            logging.debug(
                "    Correct key/chord states accuracy: %s / %s = %s",
                total_chord_correct_correct,
                total_chord_correct,
                total_chord_correct_correct / total_chord_correct,
            )
        else:
            logging.debug("    No correct key/chord states")

    def debug_chord_sequence_priors(self, states: List[State], max_to_print: int = 20):
        """
        Log chord sequence priors as debug messages.

        Parameters
        ----------
        states : List[State]
            A List of states which contain a csm_log_prior.
        max_to_print : int, optional
            Print maximum this many incorrect chords from the prior for each state
            (or until the correct chord is reached).
        """
        if len(states) == 0:
            return

        change_index = states[0].prev_state.change_index
        if change_index == 0:
            # ICM is used here
            return

        chords = self.piece.get_chords()
        chord_changes = self.piece.get_chord_change_indices()
        keys = self.piece.get_keys()
        key_changes = self.piece.get_key_change_input_indices()

        chord_index = bisect.bisect_left(chord_changes, change_index)
        if chord_index == len(chord_changes) or chord_changes[chord_index] != change_index:
            # Change index is incorrect
            return

        key_index = bisect.bisect_left(key_changes, change_index)
        if key_index != len(key_changes) and key_changes[key_index] == change_index:
            # Key change here (csm is unused)
            return

        correct_key = keys[key_index - 1]
        correct_key_one_hot = correct_key.get_one_hot_index()

        correct_next_chord = chords[chord_index]
        correct_next_chord_one_hot = correct_next_chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )
        correct_next_chord_relative_one_hot = correct_next_chord.get_one_hot_index(
            relative=True, use_inversion=True, pad=False
        )

        correct_prev_chords = chords[max(0, chord_index - 5) : chord_index]
        correct_prev_chords_one_hots = [
            chord.get_one_hot_index(relative=False, use_inversion=True, pad=False)
            for chord in correct_prev_chords
        ]

        relative_chord_labels = hu.get_chord_label_list(
            self.CHORD_OUTPUT_TYPE,
            use_inversions=True,
            relative=True,
            relative_to=correct_key.relative_tonic,
            pad=False,
        )

        logging.debug(
            "Chord prior for index %s relative to key %s:",
            change_index,
            correct_key,
        )
        logging.debug(
            "  Recent correct chords: %s",
            "; ".join(self.chord_labels[correct_prev_chords_one_hots]),
        )
        logging.debug("  Correct next chord: %s", self.chord_labels[correct_next_chord_one_hot])

        total = 0
        total_correct = 0

        for state in states:
            if state.prev_state.chord != correct_prev_chords_one_hots[-1]:
                # Most recent chord is incorrect
                continue

            if state.key != correct_key_one_hot:
                # Current key is incorrect
                continue

            total += 1

            csm_prior = np.exp(state.csm_log_prior)

            rankings = list(np.argsort(-csm_prior))
            correct_prob = csm_prior[correct_next_chord_relative_one_hot]
            correct_rank = rankings.index(correct_next_chord_relative_one_hot)
            logging.debug(
                "    p(%s)=%s rank=%s",
                self.chord_labels[correct_next_chord_one_hot],
                correct_prob,
                correct_rank,
            )

            if correct_rank == 0:
                total_correct += 1
                continue

            state_prev_chords_one_hots, _ = state.get_chords()
            state_prev_chords_one_hots = state_prev_chords_one_hots[-6:-1]
            logging.debug(
                "      Previous chords: %s",
                "; ".join(self.chord_labels[state_prev_chords_one_hots]),
            )

            logging.debug("      Top chords:")
            for rank, one_hot in enumerate(rankings[: min(max_to_print, correct_rank + 1)]):
                logging.debug(
                    "           %s%s: p(%s) = %s",
                    "*" if one_hot == correct_next_chord_relative_one_hot else " ",
                    rank,
                    relative_chord_labels[one_hot],
                    csm_prior[one_hot],
                )

        logging.debug("CSM accuracy")
        if total > 0:
            logging.debug(
                "    Correct chord priors / correct key/prev chord states: %s / %s = %s",
                total_correct,
                total,
                total_correct / total,
            )
        else:
            logging.debug("    No correct key/prev chord states")

    def debug_chord_change_probs(self, change_probs: List[float]):
        """
        Log chord change probabilities to as debug messages.

        Parameters
        ----------
        change_probs : List[float]
            The chord change probability for each input.
        """
        index_invalid = np.roll(self.piece.get_duration_cache() == 0, 1)
        index_invalid[0] = True

        change_correct = 0
        change_total = 0
        no_change_correct = 0
        no_change_total = 0

        for i, (change_prob, invalid) in enumerate(zip(change_probs, index_invalid)):
            if invalid:
                continue

            if i in self.piece.get_chord_change_indices():
                change_total += 1
                if change_prob < 0.5:
                    logging.debug(
                        "Piece changes chord on index %s but change_prob=%s",
                        i,
                        change_prob,
                    )
                else:
                    change_correct += 1

            else:
                no_change_total += 1
                if change_prob > 0.5:
                    logging.debug(
                        "Piece doesn't change chord on index %s but change_prob=%s",
                        i,
                        change_prob,
                    )
                else:
                    no_change_correct += 1

        logging.debug("CTM accuracy")
        logging.debug(
            "    Change: %s / %s = %s", change_correct, change_total, change_correct / change_total
        )
        logging.debug(
            "    No change: %s / %s = %s",
            no_change_correct,
            no_change_total,
            no_change_correct / no_change_total,
        )

    def debug_chord_classifications(
        self,
        chord_ranges: List[Tuple[int, int]],
        chord_classifications: List[float],
    ):
        """
        Log chord classifications as debug messages.

        Parameters
        ----------
        piece : Piece
            The piece whose chord classifications to debug.
        chord_ranges : List[Tuple[int, int]]
            A list of the chord ranges that were classified.
        chord_classifications : List[List[float]]
            The log_probability of each chord for each given range.
        """
        change_indices = self.piece.get_chord_change_indices()

        num_correct_ranges = 0
        num_correct_chords = 0

        for chord_range, chord_probs in zip(chord_ranges, np.exp(chord_classifications)):
            range_start, range_end = chord_range

            correct_chords = self.piece.get_chords_within_range(range_start, range_end)
            correct_chords_one_hot = [
                chord.get_one_hot_index(relative=False, use_inversion=True, pad=False)
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

            if is_range_correct:
                num_correct_ranges += 1

                if is_classification_correct:
                    num_correct_chords += 1

            # Bad range or already correct
            if is_any_classification_correct or not is_range_correct:
                continue

            correct_string = (
                "=== " if is_classification_correct else "*** " if is_range_correct else ""
            )

            logging.debug(
                "%sChord classification results for range %s:", correct_string, chord_range
            )
            logging.debug(
                "    correct chords: %s",
                "; ".join(self.chord_labels[correct_chords_one_hot]),
            )
            for one_hot, prob, rank in zip(correct_chords_one_hot, correct_probs, correct_rank):
                logging.debug(
                    "        p(%s)=%s, rank=%s",
                    self.chord_labels[one_hot],
                    prob,
                    rank,
                )

            logging.debug("    Top chords:")
            for rank, one_hot in enumerate(
                rankings[: min(self.max_chords_to_print, max(correct_rank) + 1)]
            ):
                logging.debug(
                    "       %s%s: p(%s) = %s",
                    "*" if one_hot in correct_chords_one_hot else " ",
                    rank,
                    self.chord_labels[one_hot],
                    chord_probs[one_hot],
                )

        logging.debug("CCM accuracy")
        logging.debug(
            "    correct_chords / correct_ranges: %s / %s = %s",
            num_correct_chords,
            num_correct_ranges,
            num_correct_chords / num_correct_ranges,
        )

    def debug_initial_chord_prior(self, icm_prior: List[float]):
        """
        Log the initial chord prior as a debug message.

        Parameters
        ----------
        icm_prior : List[float]
            A list of probabilities for each relative chord symbol.
        """
        rankings = list(np.argsort(-np.array(icm_prior)))

        correct_chord = self.piece.get_chords()[0]
        correct_key = self.piece.get_keys()[0]

        correct_chord_one_hot_relative = correct_chord.get_one_hot_index(
            relative=True,
            use_inversion=True,
            pad=False,
        )
        correct_chord_one_hot = correct_chord.get_one_hot_index(
            relative=False,
            use_inversion=True,
            pad=False,
        )

        correct_prob = icm_prior[correct_chord_one_hot_relative]
        correct_rank = rankings.index(correct_chord_one_hot_relative)

        logging.debug("Initial chord prior relative to key %s:", correct_key)
        logging.debug("  Correct chord: %s", self.chord_labels[correct_chord_one_hot])
        logging.debug("    prob=%s rank=%s", correct_prob, correct_rank)

        relative_chord_labels = hu.get_chord_label_list(
            correct_chord.pitch_type,
            use_inversions=True,
            relative=True,
            relative_to=correct_key.relative_tonic,
            pad=False,
        )

        if correct_rank != 0:
            logging.debug("      Top chords:")
            for rank, one_hot in enumerate(
                rankings[: min(self.max_chords_to_print, correct_rank + 1)]
            ):
                logging.debug(
                    "         %s%s: p(%s) = %s",
                    "*" if one_hot == correct_chord_one_hot_relative else " ",
                    rank,
                    relative_chord_labels[one_hot],
                    icm_prior[one_hot],
                )


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
        ksm_exponent=ARGS.ksm_exponent,
    )
