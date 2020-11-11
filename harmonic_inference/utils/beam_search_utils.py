"""Utils (beams and state) for running beam search."""
import copy
import heapq
from fractions import Fraction
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np

import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.data_types import PitchType
from harmonic_inference.data.piece import Chord, Key


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
        prev_state: "State" = None,
        hash_length: int = None,
        csm_hidden_state: np.array = None,
        ktm_hidden_state: np.array = None,
        csm_log_prior: np.array = None,
        key_obj: Key = None,
        must_key_transition: bool = False,
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
        key_obj : Key
            The key object of this state, in case it can be copied from the previous state.
        must_key_transition : bool
            True if this State must key transition to be valid. This happens in the case that
            the chord's root or bass note are outside of the valid range of relative pitches
            from the key's tonic.
        """
        self._valid = True

        self.chord = chord
        self.key = key
        self.change_index = change_index

        self.log_prob = log_prob

        self.prev_state = prev_state
        if hash_length is not None:
            if self.prev_state is None:
                self.hash_tuple = tuple([None] * (hash_length - 1) + [(key, chord)])
            else:
                self.hash_tuple = tuple(list(prev_state.hash_tuple[1:]) + [(key, chord)])

        self.csm_hidden_state = copy.deepcopy(csm_hidden_state)
        self.ktm_hidden_state = copy.deepcopy(ktm_hidden_state)
        self.csm_log_prior = csm_log_prior

        # Key/chord objects
        self.chord_obj = None
        self.key_obj = key_obj

        self._must_key_transition = must_key_transition

    def is_valid(self, check_key: bool = False) -> bool:
        """
        Return if this State is valid currently or not.

        Parameters
        ----------
        check_key : bool
            True to check whether this state needs to key transition. False otherwise.

        Returns
        -------
        is_valid : bool
            True if this state is currently valid. False otherwise.
        """
        if check_key:
            return self._valid and not self._must_key_transition
        return self._valid

    def invalidate(self):
        """
        Mark this State as invalid.
        """
        self._valid = False

    def copy(self) -> "State":
        """
        Return a deep copy of this State.

        Returns
        -------
        new_state : State
            A deep copy of this state.
        """
        return State(
            chord=self.chord,
            key=self.key,
            change_index=self.change_index,
            log_prob=self.log_prob,
            prev_state=self.prev_state,
            hash_length=len(self.hash_tuple) if hasattr(self, "hash_tuple") else None,
            csm_hidden_state=self.csm_hidden_state,
            ktm_hidden_state=self.ktm_hidden_state,
            csm_log_prior=self.csm_log_prior,
            key_obj=self.key_obj,
            must_key_transition=self._must_key_transition,
        )

    def chord_transition(
        self,
        chord: int,
        change_index: int,
        log_prob: float,
        pitch_type: PitchType,
        LABELS: Dict,
    ) -> "State":
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
        pitch_type : PitchType
            The pitch type used to store the chord root.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.

        Returns
        -------
        new_state : State
            The state resulting from the given transition.
        """
        root, chord_type, inversion = LABELS["chord"][chord]
        bass = hu.get_bass_note(chord_type, root, inversion, pitch_type)

        tonic = self.get_key(pitch_type, LABELS).relative_tonic
        relative_root = root - tonic
        relative_bass = bass - tonic

        minimum = hc.MIN_RELATIVE_TPC - hc.RELATIVE_TPC_EXTRA
        maximum = hc.MAX_RELATIVE_TPC + hc.RELATIVE_TPC_EXTRA

        if (
            relative_root < minimum
            or relative_bass < minimum
            or relative_root >= maximum
            or relative_bass >= maximum
        ):
            return None

        must_key_transition = (
            relative_root < hc.MIN_RELATIVE_TPC
            or relative_root >= hc.MAX_RELATIVE_TPC
            or relative_bass < hc.MIN_RELATIVE_TPC
            or relative_bass >= hc.MAX_RELATIVE_TPC
        )

        return State(
            chord=chord,
            key=self.key,
            change_index=change_index,
            log_prob=self.log_prob + log_prob,
            prev_state=self,
            hash_length=len(self.hash_tuple) if hasattr(self, "hash_tuple") else None,
            csm_hidden_state=self.csm_hidden_state,
            ktm_hidden_state=self.ktm_hidden_state,
            csm_log_prior=self.csm_log_prior,
            key_obj=self.key_obj,
            must_key_transition=must_key_transition,
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

    def key_transition(
        self,
        key: int,
        log_prob: float,
        pitch_type: PitchType,
        LABELS: Dict,
    ) -> "State":
        """
        Transition to a new key on the most recent chord.

        Parameters
        ----------
        key : int
            The new key to transition to.
        log_prob : float
            The log-probability of the given key transition, in terms of the input index (KTM)
            and the new key (KSM).
        pitch_type : PitchType
            The pitch type used to store the chord root.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.

        Returns
        -------
        new_state : State
            Essentially, a replacement of this state (the new state's prev_state is the same),
            but with a new key.
        """
        root, chord_type, inversion = LABELS["chord"][self.chord]
        bass = hu.get_bass_note(chord_type, root, inversion, pitch_type)

        tonic, _ = LABELS["key"][key]
        relative_root = root - tonic
        relative_bass = bass - tonic

        if (
            relative_root < hc.MIN_RELATIVE_TPC
            or relative_bass < hc.MIN_RELATIVE_TPC
            or relative_root >= hc.MAX_RELATIVE_TPC
            or relative_bass >= hc.MAX_RELATIVE_TPC
        ):
            return None

        return State(
            chord=self.chord,
            key=key,
            change_index=self.change_index,
            log_prob=self.prev_state.log_prob + log_prob,
            prev_state=self.prev_state,
            hash_length=len(self.hash_tuple) if hasattr(self, "hash_tuple") else None,
            csm_hidden_state=self.csm_hidden_state,
            ktm_hidden_state=self.ktm_hidden_state,
            csm_log_prior=self.csm_log_prior,
            key_obj=None,
            must_key_transition=False,
        )

    def add_csm_prior(
        self,
        pitch_type: PitchType,
        duration_cache: np.array,
        onset_cache: List[Tuple[int, Fraction]],
        onset_level_cache: List[Fraction],
        LABELS: Dict,
    ):
        """
        Add the log_prior for this state's current relative chord to this state's log_prob.

        This has its own method essentially to notify the State that its prior, chord, and key
        are all now valid.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root.
        duration_cache : np.array
            The duration of each input in the current piece.
        onset_cache : List[Tuple[int, Fraction]]
            The onset of each input in the current piece.
        onset_level_cache : List[int]
            The onset level of each input in the current piece.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.
        """
        range_length = self.change_index - self.prev_state.change_index
        self.log_prob += (
            self.csm_log_prior[
                self.get_relative_chord_index(
                    pitch_type,
                    duration_cache,
                    onset_cache,
                    onset_level_cache,
                    LABELS,
                )
            ]
            * range_length
        )

    def get_csm_input(
        self,
        pitch_type: PitchType,
        duration_cache: np.array,
        onset_cache: List[Tuple[int, Fraction]],
        onset_level_cache: List[int],
        LABELS: Dict,
    ) -> np.array:
        """
        Get the input for the next step of this state's CSM.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root and key tonic.
        duration_cache : np.array
            The duration of each input in the current piece.
        onset_cache : List[Tuple[int, Fraction]]
            The onset of each input in the current piece.
        onset_level_cache : List[int]
            The onset level of each input in the current piece.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.

        Returns
        -------
        csm_input : np.array
            The input for the next step of this state's CSM.
        """
        key_change_vector = np.zeros(Key.get_key_change_vector_length(pitch_type, one_hot=False))
        is_key_change = 0
        if self.prev_state is not None and self.prev_state.key != self.key:
            # Key change
            is_key_change = 1
            prev_key = self.prev_state.get_key(pitch_type, LABELS)
            key_change_vector = prev_key.get_key_change_vector(self.get_key(pitch_type, LABELS))

        return np.expand_dims(
            np.concatenate(
                [
                    self.get_chord(
                        pitch_type,
                        duration_cache,
                        onset_cache,
                        onset_level_cache,
                        LABELS,
                    ).to_vec(pad=False),
                    key_change_vector,
                    [is_key_change],
                ]
            ),
            axis=0,
        )

    def get_ktm_input(
        self,
        pitch_type: PitchType,
        duration_cache: np.array,
        onset_cache: List[Tuple[int, Fraction]],
        onset_level_cache: List[int],
        LABELS: Dict,
    ) -> np.array:
        """
        Get the input for the next step of this state's KTM.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root and key tonic.
        duration_cache : np.array
            The duration of each input in the current piece.
        onset_cache : List[Tuple[int, Fraction]]
            The onset of each input in the current piece.
        onset_level_cache : List[int]
            The onset level of each input in the current piece.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.

        Returns
        -------
        ktm_input : np.array
            The input for the next step of this state's KTM.
        """
        return np.expand_dims(
            self.get_chord(
                pitch_type,
                duration_cache,
                onset_cache,
                onset_level_cache,
                LABELS,
            ).to_vec(pad=True),
            axis=0,
        )

    def get_ksm_input(
        self,
        pitch_type: PitchType,
        duration_cache: np.array,
        onset_cache: List[Tuple[int, Fraction]],
        onset_level_cache: List[int],
        LABELS: Dict,
        length: int = 0,
    ) -> np.array:
        """
        Get the input for this state's KSM. This should be run only when the KTM has decided
        that there should be a key change, and the generated input will be from the last key
        change to the current state.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root and key tonic.
        duration_cache : np.array
            The duration of each input in the current piece.
        onset_cache : List[Tuple[int, Fraction]]
            The onset of each input in the current piece.
        onset_level_cache : List[int]
            The onset level of each input in the current piece.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.
        length : int
            The number of input vectors that will be appended to the returned input nd-array.
            The default value (0) should be used for the initial (non-recursive) call.

        Returns
        -------
        ksm_input : np.array
            The input for the KSM form the last key change until now, with `length` additional
            input vectors filled with 0 appended to the end.
        """
        this_ksm_input = self.get_chord(
            pitch_type,
            duration_cache,
            onset_cache,
            onset_level_cache,
            LABELS,
        ).to_vec(pad=True)

        if (
            self.prev_state is None
            or self.prev_state.key != self.key
            or self.prev_state.chord is None
        ):
            # Base case - this is the first state
            ksm_input = np.zeros((1 + length, len(this_ksm_input)))
            ksm_input[0] = this_ksm_input
            return ksm_input

        ksm_input = self.prev_state.get_ksm_input(
            pitch_type,
            duration_cache,
            onset_cache,
            onset_level_cache,
            LABELS,
            length=length + 1,
        )
        ksm_input[len(ksm_input) - length - 1] = this_ksm_input
        return ksm_input

    def get_chord(
        self,
        pitch_type: PitchType,
        duration_cache: np.array,
        onset_cache: List[Tuple[int, Fraction]],
        onset_level_cache: List[int],
        LABELS: Dict,
    ) -> Chord:
        """
        Get the Chord object of this state.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root.
        duration_cache : np.array
            The duration of each input in the current piece.
        onset_cache : List[Tuple[int, Fraction]]
            The onset of each input in the current piece.
        onset_level_cache : List[int]
            The onset level of each input in the current piece.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.

        Returns
        -------
        chord : Chord
            The chord object of this state, relative to its key.
        """
        if self.chord_obj is None:
            key = self.get_key(pitch_type, LABELS)
            root, chord_type, inversion = LABELS["chord"][self.chord]

            prev_index = self.prev_state.change_index
            index = self.change_index

            self.chord_obj = Chord(
                root,
                hu.get_bass_note(chord_type, root, inversion, pitch_type),
                key.relative_tonic,
                key.relative_mode,
                chord_type,
                inversion,
                onset_cache[prev_index],
                onset_level_cache[prev_index],
                onset_cache[index],
                onset_level_cache[index],
                np.sum(duration_cache[prev_index : self.change_index]),
                pitch_type,
            )

        return self.chord_obj

    def get_key(self, pitch_type: PitchType, LABELS: Dict) -> Key:
        """
        Get the Key object of this state.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the key tonic.

        Returns
        -------
        key : Key
            The key object of this state.
        """
        if self.key_obj is None:
            tonic, mode = LABELS["key"][self.key]

            if self.prev_state is None:
                global_tonic = tonic
                global_mode = mode
            else:
                prev_key = self.prev_state.get_key(pitch_type, LABELS)
                global_tonic = prev_key.global_tonic
                global_mode = prev_key.global_mode

            self.key_obj = Key(tonic, tonic, global_tonic, mode, mode, global_mode, pitch_type)

        return self.key_obj

    def get_relative_chord_index(
        self,
        pitch_type: PitchType,
        duration_cache: np.array,
        onset_cache: List[Tuple[int, Fraction]],
        onset_level_cache: List[int],
        LABELS: Dict,
    ) -> int:
        """
        Get the one-hot index of the chord symbol, relative to the current key.

        Parameters
        ----------
        pitch_type : PitchType
            The pitch type used to store the chord root.
        duration_cache : np.array
            The duration of each input in the current piece.
        onset_cache : List[Tuple[int, Fraction]]
            The onset of each input in the current piece.
        onset_level_cache : List[int]
            The onset level of each input in the current piece.
        LABELS : Dict
            A Dictionary of key and chord labels for the current piece.

        Returns
        -------
        relative_chord : int
            The current chord, relative to the current key.
        """
        return self.get_chord(
            pitch_type,
            duration_cache,
            onset_cache,
            onset_level_cache,
            LABELS,
        ).get_one_hot_index(relative=True, use_inversion=True, pad=False)

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

    def fits_in_beam(self, state: State, check_hash: bool = True) -> bool:
        """
        Check if the given state will fit in the beam, but do not add it.

        This should be used only to check for early exits. If you will add the
        state to the beam immediately anyways, it is faster to just use
        beam.add(state) and check its return value.

        Parameters
        ----------
        state : State
            The state to check.
        check_hash : bool
            If True and this is a HashedBeam, check the State's hash slot. Otherwise, only
            check the global beam.

        Returns
        -------
        fits : bool
            True if the state will fit in the beam both by size and by log_prob.
        """
        return len(self) < self.beam_size or self.beam[0] < state

    def add(self, state: State, force: bool = False) -> bool:
        """
        Add the given state to the beam, if it fits, and return a boolean indicating
        if the state fit.

        Parameters
        ----------
        state : State
            The state to add to the beam.
        force : bool
            Force the state into the beam. Do not check the beam size first.

        Returns
        -------
        added : bool
            True if the given state was added to the beam. False otherwise.
        """
        if not force and len(self) >= self.beam_size:
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
        return max(self) if len(self) > 0 else None

    def empty(self):
        """
        Empty this beam.
        """
        self.beam = []

    def __iter__(self) -> Iterator[State]:
        return self.beam.__iter__()

    def __len__(self) -> int:
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

    def fits_in_beam(self, state: State, check_hash: bool = True) -> bool:
        """
        Check if the given state will fit in the beam, but do not add it.

        This should be used only to check for early exits. If you will add the
        state to the beam immediately anyways, it is faster to just use
        beam.add(state) and check its return value.

        Parameters
        ----------
        state : State
            The state to check.
        check_hash : bool
            If True, check the State's hash slot. Otherwise, only check the global beam.

        Returns
        -------
        fits : bool
            True if the state will fit in the beam both by size and by log_prob.
        """
        global_check = len(self) < self.beam_size or self.beam[0] < state

        if global_check and check_hash:
            state_hash = state.get_hash()
            return state_hash not in self.state_dict or self.state_dict[state_hash] < state

        return global_check

    def _fix_beam_min(self):
        """
        Remove all states with valid == False from the top of the min-heap until the min
        state is valid.
        """
        while not self.beam[0].is_valid(check_key=False):
            heapq.heappop(self.beam)

    def add(self, state: State, force: bool = False) -> bool:
        """
        Add the given state to the beam, if it fits, and return a boolean indicating
        if the state fit.

        Parameters
        ----------
        state : State
            The state to add to the beam.
        force : bool
            Force the state into the beam. The hash is still enforced, but not the full
            beam.

        Returns
        -------
        added : bool
            True if the given state was added to the beam. False otherwise.
        """
        state_hash = state.get_hash()

        if state_hash in self.state_dict:
            if self.state_dict[state_hash] < state:
                self.state_dict[state_hash].invalidate()
                self.state_dict[state_hash] = state
                heapq.heappush(self.beam, state)
                self._fix_beam_min()
                return True
            return False

        # Here, the state is in a new hash
        if not force and len(self) >= self.beam_size:
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

    def __iter__(self) -> Iterator[State]:
        return self.state_dict.values().__iter__()

    def __len__(self) -> int:
        return len(self.state_dict)
