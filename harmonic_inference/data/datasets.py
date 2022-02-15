"""Module containing datasets for the various models."""
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.key import get_key_change_vector_length
from harmonic_inference.data.piece import (
    Piece,
    get_score_piece_from_data_frames,
    get_score_piece_from_music_xml,
)
from harmonic_inference.data.vector_decoding import (
    reduce_chord_one_hots,
    reduce_chord_types,
    remove_chord_inversions,
    transpose_chord_vector,
    transpose_note_vector,
    update_chord_vector_info,
)
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES
from harmonic_inference.utils.harmonic_utils import (
    get_bass_note,
    get_chord_from_one_hot_index,
    get_chord_one_hot_index,
    get_key_from_one_hot_index,
    get_key_one_hot_index,
    get_vector_from_chord_type,
)


class HarmonicDataset(Dataset):
    """
    The base dataset that all model-specific dataset objects will inherit from.
    """

    def __init__(
        self,
        transform: Callable = None,
        pad_inputs: bool = True,
        pad_targets: bool = True,
        save_padded_input_lengths: bool = True,
        save_padded_target_lengths: bool = True,
        input_mask: List[int] = None,
    ):
        """
        Create a new base dataset.

        Parameters
        ----------
        transform : Callable
            A function to call on each returned data point.
        pad_inputs : bool
            Pad this datset's inputs (when calling self.pad()).
        pad_targets : bool
            Pad this dataset's targets (when calling self.pad()).
        save_padded_input_lengths : bool
            Save the returned lengths when padding inputs (when calling self.pad()).
        save_padded_target_lengths : bool
            Save the returned lengths when padding targets (when calling self.pad()).
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        self.h5_path = None
        self.padded = False
        self.in_ram = True
        self.transform = transform

        self.inputs = None
        self.input_lengths = None
        self.max_input_length = None
        self.pad_inputs = pad_inputs
        self.save_padded_input_lengths = save_padded_input_lengths

        self.targets = None
        self.target_lengths = None
        self.max_target_length = None
        self.pad_targets = pad_targets
        self.save_padded_target_lengths = save_padded_target_lengths

        self.input_mask = input_mask

        self.hidden_states = None

    def finalize_data(
        self,
        data: Dict[str, np.array],
        item,
        transposition: int = None,
    ) -> Dict:
        """
        Finalize the given data return dict by adding the hidden states corresponding
        to the given index (if hidden_states are loaded), and then transforming the
        data with the Dataset's transform Callable.

        Parameters
        ----------
        data : Dict[str, np.array]
            The data dictionary.
        item : int
            The index (or other indexer) to load the correct hidden state.
        transposition : int
            A transposition to be applied to the data, if any.

        Returns
        -------
        data : Dict
            The given data, finalized with hidden states and transformed.
        """
        if self.hidden_states is not None:
            data["hidden_states"] = (
                self.hidden_states[0][:, item],
                self.hidden_states[1][:, item],
            )

        self.reduce(data, transposition=transposition)

        if self.input_mask is not None:
            data["inputs"][:] *= self.input_mask

        if self.transform:
            data.update(
                {
                    key: self.transform(value)
                    for key, value in data.items()
                    if isinstance(value, np.ndarray)
                }
            )

        return data

    def reduce(self, data: Dict, transposition: int = None):
        """
        Reduce the given data in place.

        Parameters
        ----------
        data : Dict
            The data, created by the __getitem__(item) function, to be reduced with chord
            type and inversion reductions. This default implementation does nothing.
        transposition : int
            A transposition to be applied to the data, if any.
        """

    def set_hidden_states(self, hidden_states: np.array):
        """
        Load hidden states into this Dataset object.

        Parameters
        ----------
        hidden_states : np.array
            The hidden states to load into this Dataset.
        """
        self.hidden_states = hidden_states

    def __len__(self) -> int:
        """
        Return how many input data points this dataset contains.

        Returns
        -------
        length : int
            The number of data points in this dataset.
        """
        if not self.in_ram:
            with h5py.File(self.h5_path, "r") as h5_file:
                return len(h5_file["inputs"])
        return len(self.inputs)

    def __getitem__(self, item, transposition: int = None) -> Dict:
        """
        Get a specific item (or range of items) from this dataset.

        Parameters
        ----------
        item : Indexer
            Some type of indexer which can index into lists and numpy arrays.
            This specifies the input data to be returned.
        transposition : int
            A transposition to be applied to the given data point, if desired.

        Returns
        -------
        data : Dict
            The data points specified by the item Indexer. This will contain the fields:
            inputs, input_lengths, targets, target_legnths, and hidden_states, if these
            are attributes of the dataset object.
        """
        if not self.in_ram:
            assert self.h5_path is not None, "Data must be either in ram or in an h5_file."
            with h5py.File(self.h5_path, "r") as h5_file:
                data = {
                    key: h5_file[key][item]
                    for key in ["inputs", "targets", "input_lengths", "target_lengths"]
                    if key in h5_file
                }

        else:
            data = {"inputs": self.inputs[item]}

            # During inference, we have no targets
            if self.targets is not None:
                data["targets"] = self.targets[item]

            if self.input_lengths is not None:
                if self.max_input_length is None:
                    self.max_input_length = np.max(self.input_lengths)
                padded_input = np.zeros(([self.max_input_length] + list(data["inputs"][0].shape)))
                padded_input[: len(data["inputs"])] = data["inputs"]
                data["inputs"] = padded_input
                data["input_lengths"] = self.input_lengths[item]

            if self.target_lengths is not None:
                if self.max_target_length is None:
                    self.max_target_length = np.max(self.target_lengths)
                try:
                    padded_target = np.zeros(
                        ([self.max_target_length] + list(data["targets"][0].shape))
                    )
                except AttributeError:
                    # data["targets"][0] is an int -- target is just list of ints
                    padded_target = np.zeros(self.max_target_length)
                padded_target[: len(data["targets"])] = data["targets"]
                data["targets"] = padded_target
                data["target_lengths"] = self.target_lengths[item]

        return self.finalize_data(data, item, transposition=transposition)

    def pad(self):
        """
        Default padding function to pad a HarmonicDataset's input and target arrays to be of the
        same size, so that they can be combined into a numpy nd-array, one element per row
        (with 0's padded to the end).

        This function works if input and target are lists of np.ndarrays, and the ndarrays match
        in every dimension except possibly the first.

        This also adds fields input_lengths and target_lengths (both arrays, with 1 integer per
        input and target), representing the lengths of the non-padded entries for each piece.

        This also sets self.padded to True.
        """
        if self.padded:
            return

        if self.pad_targets:
            self.targets, target_lengths = pad_array(self.targets)
            if self.save_padded_target_lengths:
                self.target_lengths = target_lengths

        if self.pad_inputs:
            self.inputs, input_lengths = pad_array(self.inputs)
            if self.save_padded_input_lengths:
                self.input_lengths = input_lengths

        self.padded = True

    def to_h5(self, h5_path: Union[str, Path], file_ids: Iterable[int] = None):
        """
        Write this HarmonicDataset out to an h5 file, containing its inputs and targets.
        If the dataset is already being read from an h5py file, this simply copies that
        file (self.h5_path) over to the given location and updates self.h5_path.

        Parameters
        ----------
        h5_path : Union[str, Path]
            The filename of the h5 file to write to.
        file_ids : Iterable[int]
            The file_ids of the pieces in this dataset. Will be added to the h5 file as `file_ids`
            if given.
        """
        if isinstance(h5_path, str):
            h5_path = Path(h5_path)

        if not h5_path.parent.exists():
            h5_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.padded:
            self.pad()

        if self.h5_path:
            try:
                shutil.copy(self.h5_path, h5_path)
                self.h5_path = h5_path
            except OSError:
                logging.exception("Error copying existing h5 file %s to %s", self.h5_path, h5_path)
            return

        h5_file = h5py.File(h5_path, "w")

        keys = [
            "inputs",
            "targets",
            "input_lengths",
            "target_lengths",
            "piece_lengths",
            "key_change_replacements",
            "target_pitch_type",
        ]
        for key in keys:
            if hasattr(self, key) and getattr(self, key) is not None:
                h5_file.create_dataset(key, data=np.array(getattr(self, key)), compression="gzip")
        if file_ids is not None:
            h5_file.create_dataset("file_ids", data=np.array(file_ids), compression="gzip")
        h5_file.close()


class ChordTransitionDataset(HarmonicDataset):
    """
    A dataset to detect chord transitions.

    Each input is the sequence of input vectors of a piece.

    Each target is a list of 0 (no change), 1 (change), and -100 (do not measure).
    """

    train_batch_size = 8
    valid_batch_size = 64
    chunk_size = 64

    def __init__(
        self,
        pieces: List[Piece],
        transform: Callable = None,
        dummy_targets: bool = False,
        input_mask: List[int] = None,
    ):
        """
        Create a new Chord Transition Dataset from the given pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to get the data from.
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        dummy_targets : boolean
            True if the targets will be ignored. False to load targets from the given pieces.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        super().__init__(transform=transform, input_mask=input_mask)
        self.inputs = [
            np.vstack(
                [
                    note.to_vec(dur_from_prev=from_prev, dur_to_next=to_next)
                    for note, from_prev, to_next in zip(
                        piece.get_inputs(),
                        [None] + list(piece.get_duration_cache()),
                        list(piece.get_duration_cache()),
                    )
                ]
            )
            for piece in pieces
        ]

        self.targets = [np.zeros(len(piece_input), dtype=int) for piece_input in self.inputs]
        if not dummy_targets:
            for piece, target in zip(pieces, self.targets):
                target[piece.get_chord_change_indices()] = 1
                target[0] = -100
                target[np.roll(piece.get_duration_cache() == 0, 1)] = -100

        self.input_lengths = np.array([len(inputs) for inputs in self.inputs])
        self.target_lengths = np.array([len(target) for target in self.targets])

        self.pad_targets = True
        self.pad_inputs = True


class ChordClassificationDataset(HarmonicDataset):
    """
    A dataset to classify chords.

    Each input is a list of note vectors of the notes in each chord
    in a piece, plus some window length (2) on each end.

    The targets are the one_hot indexes of each of those chords.
    """

    train_batch_size = 256
    valid_batch_size = 512
    chunk_size = 1024

    def __init__(
        self,
        pieces: List[Piece],
        transform: Callable = None,
        ranges: List[List[Tuple[int, int]]] = None,
        change_indices: List[List[int]] = None,
        dummy_targets: bool = False,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        transposition_range: Union[List[int], Tuple[int, int]] = (0, 0),
        input_mask: List[int] = None,
    ):
        """
        Create a new chord classification dataset from the given pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to get the data from.
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        ranges : List[List[Tuple[int, int]]]
            Custom chord ranges to use, if not using the default ones from each piece.
            This should be a list of tuple [start, stop) ranges for each piece.
        change_indices : List[List[int]]
            Custom change indices to use, if not using the default ones from each piece.
            This should be a list of chord change indices for each piece.
        dummy_targets : bool
            True to store all zeros for targets. False to get targets from each piece.
        reduction : Dict[ChordType, ChordType]
            A reduction to apply to target chords when returning data with __getitem__.
            These are not applied when the data is loaded, but only when it is returned.
            So it is safe to change this after initialization.
        use_inversions : bool
            True to use inversions for target chords when returning data with __getitem__.
            False to reduce all chords to root position. This is not applied when the data
            is loaded, but only when it is returned. So it is safe to change this after
            initialization.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each note and chord of the
            dataset. Each __getitem__ call will return every possible transposition in this
            (min, max) range, inclusive on each side. The transpositions are measured in
            whatever PitchType is used in the dataset.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        super().__init__(transform=transform, pad_targets=False, input_mask=input_mask)
        self.target_pitch_type = []

        if ranges is None:
            ranges = np.full(len(pieces), None)
        if change_indices is None:
            change_indices = np.full(len(pieces), None)

        self.inputs = []
        for piece, piece_ranges, piece_change_indices in zip(pieces, ranges, change_indices):
            self.inputs.extend(
                piece.get_chord_note_inputs(
                    ranges=piece_ranges, change_indices=piece_change_indices
                )
            )

        self.input_lengths = np.array([len(inputs) for inputs in self.inputs])

        if dummy_targets:
            self.targets = np.zeros(len(self.inputs))
        else:
            self.targets = np.array(
                [
                    chord.get_one_hot_index(relative=False, use_inversion=True, pad=False)
                    for piece in pieces
                    for chord in piece.get_chords()
                ]
            )

            if len(pieces) > 0 and len(pieces[0].get_chords()) > 0:
                self.target_pitch_type = [pieces[0].get_chords()[0].pitch_type.value]

        self.dummy_targets = dummy_targets
        self.reduction = reduction
        self.use_inversions = use_inversions
        self.transposition_range = tuple(transposition_range)
        self.num_transpositions = self.transposition_range[1] - self.transposition_range[0] + 1

    def __len__(self):
        return super().__len__() * self.num_transpositions

    def __getitem__(self, item, transposition: int = None) -> Dict:
        orig_item = item // self.num_transpositions
        transposition = self.transposition_range[0] + (item % self.num_transpositions)

        return super().__getitem__(orig_item, transposition=transposition)

    def generate_intermediate_targets(self, target: int) -> Dict[str, Union[int, List]]:
        """
        For the given target index (already reduced), generate the intermediate targets of:
            - a "pitches" presence vector (length num_pitches)
            - a "bass" note pitch index (one-hot).
            - a "root" note pitch index (one-hot).

        Parameters
        ----------
        target : int
            The target chord index, already reduced according to use_inversions, reduction,
            and pitch_type.

        Returns
        -------
        intermediate_targets : Dict[str, Union[int, List]]
            A dictionary containing a "pitches" presence vector, "bass" one-hot, and "root"
            one-hot for the given target.
        """
        root, chord_type, inversion = get_chord_from_one_hot_index(
            target,
            PitchType(self.target_pitch_type[0]),
            use_inversions=self.use_inversions,
            relative=False,
            pad=False,
            reduction=self.reduction,
        )

        return {
            "root": root,
            "bass": get_bass_note(
                chord_type,
                root,
                inversion,
                PitchType(self.target_pitch_type[0]),
            ),
            "pitches": get_vector_from_chord_type(
                chord_type,
                PitchType(self.target_pitch_type[0]),
                root,
            ),
        }

    def reduce(self, data: Dict, transposition: int = None):
        """
        Reduce the targets using the chord type, inversion, and pitch type reductions.

        If not using dummy targets (as is done during inference), this function also:
        - Transpose each data point by the given transposition interval.
        - Generates and saves the intermediate targets into the dictionary.
        """
        if self.dummy_targets:
            return

        if transposition is None:
            transposition = 0

        data["targets"] = reduce_chord_one_hots(
            data["targets"] if isinstance(data["targets"], np.ndarray) else [data["targets"]],
            False,
            PitchType(self.target_pitch_type[0]),
            inversions_present=True,
            reduction_present=None,
            relative=False,
            reduction=self.reduction,
            use_inversions=self.use_inversions,
        )[0]

        try:
            if transposition != 0:
                root, chord_type, inversion = get_chord_from_one_hot_index(
                    data["targets"],
                    PitchType(self.target_pitch_type[0]),
                    use_inversions=self.use_inversions,
                    relative=False,
                    pad=False,
                    reduction=self.reduction,
                )

                data["targets"] = get_chord_one_hot_index(
                    chord_type,
                    root + transposition,
                    PitchType(self.target_pitch_type[0]),
                    inversion=inversion,
                    use_inversion=self.use_inversions,
                    relative=False,
                    pad=False,
                    reduction=self.reduction,
                )

                for note_index, note_vector in enumerate(data["inputs"][: data["input_lengths"]]):
                    if sum(note_vector) == 0:
                        # Some vectors are empty because of the chord window
                        continue

                    data["inputs"][note_index] = transpose_note_vector(
                        note_vector, transposition, pitch_type=PitchType(self.target_pitch_type[0])
                    )

            data["intermediate_targets"] = self.generate_intermediate_targets(data["targets"])

        except ValueError:
            # Something transposed out of the valid pitch range
            data["targets"] = -1
            data["input_lengths"] = -1
            data["inputs"] *= 0
            data["intermediate_targets"] = {
                "bass": -1,
                "root": -1,
                "pitches": -np.ones(
                    NUM_PITCHES[PitchType(self.target_pitch_type[0])], dtype=np.long
                ),
            }
            return


class ChordSequenceDataset(HarmonicDataset):
    """
    A dataset representing chord sequences relative to the key.

    The inputs are one list per piece.
    Each input list is a list of a chord vectors with some additional information. Specifically,
    they are the size of a chord vector plus a key change vector, plus 1. These are usually a chord
    vector, relative to the current key (in which case the rest of the vector is 0). In the
    special case of the first chord after a key change, the key change vector is also populated and
    the additional slot is set to 1.

    The targets are the one-hot index of each chord, relative to its key. Back-propagation
    should not be performed on targets that lie on a key change, and targets should
    be shifted backwards by 1 (such that the target of each chord is the following chord).
    """

    train_batch_size = 32
    valid_batch_size = 64
    chunk_size = 256

    def __init__(
        self,
        pieces: List[Piece],
        transform: Callable = None,
        input_reduction: Dict[ChordType, ChordType] = None,
        output_reduction: Dict[ChordType, ChordType] = None,
        use_inversions_input: bool = True,
        use_inversions_output: bool = True,
        input_mask: List[int] = None,
    ):
        """
        Create a chord sequence dataset from the given pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to get the data from.
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        input_reduction : Dict[ChordType, ChordType]
            A chord type reduction to apply to the input chords when returning data with
            __getitem__. This is not applied when the data is loaded, but only when it is
            returned. So it is safe to change this after initialization.
        output_reduction : Dict[ChordType, ChordType]
            A chord type reduction to apply to the target chords when returning data with
            __getitem__. This is not applied when the data is loaded, but only when it is
            returned. So it is safe to change this after initialization.
        use_inversions_input : bool
            True to use input inversions when returning data with __getitem__. False to reduce
            all input chords to root position. This is not applied when the data is loaded, but
            only when it is returned. So it is safe to change this after initialization.
        use_inversions_output : bool
            True to use target inversions when returning data with __getitem__. False to reduce
            all target chords to root position. This is not applied when the data is loaded, but
            only when it is returned. So it is safe to change this after initialization.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        super().__init__(transform=transform, input_mask=input_mask)
        self.inputs = []
        self.targets = []
        self.target_pitch_type = []

        for piece in pieces:
            piece_input = []
            chords = piece.get_chords()
            key_changes = piece.get_key_change_indices()
            keys = piece.get_keys()
            key_vector_length = 1 + get_key_change_vector_length(
                keys[0].tonic_type,
                one_hot=False,
            )

            if len(chords) > 0:
                self.target_pitch_type = [chords[0].pitch_type.value]

            for prev_key, key, start, end in zip(
                [None] + list(keys[:-1]), keys, key_changes, list(key_changes[1:]) + [len(chords)]
            ):
                chord_vectors = np.vstack([chord.to_vec(pad=False) for chord in chords[start:end]])
                key_vectors = np.zeros((len(chord_vectors), key_vector_length))
                if prev_key is not None:
                    key_vectors[0, :-1] = prev_key.get_key_change_vector(key)
                    key_vectors[0, -1] = 1

                piece_input.append(np.hstack([chord_vectors, key_vectors]))

            self.inputs.append(np.vstack(piece_input))
            self.targets.append(
                np.array(
                    [
                        chord.get_one_hot_index(relative=True, use_inversion=True, pad=False)
                        for chord in chords
                    ]
                )
            )

        self.target_lengths = np.array([len(target) for target in self.targets])
        self.input_lengths = np.array([len(inputs) for inputs in self.inputs])

        self.input_reduction = input_reduction
        self.output_reduction = output_reduction
        self.use_inversions_input = use_inversions_input
        self.use_inversions_output = use_inversions_output

    def reduce(self, data: Dict, transposition: int = None):
        reduce_chord_types(data["inputs"], self.input_reduction, pad=False)
        if not self.use_inversions_input:
            remove_chord_inversions(data["inputs"], pad=False)

        data["targets"] = reduce_chord_one_hots(
            data["targets"],
            False,
            PitchType(self.target_pitch_type[0]),
            inversions_present=True,
            reduction_present=None,
            relative=True,
            reduction=self.output_reduction,
            use_inversions=self.use_inversions_output,
        )


class KeyHarmonicDataset(HarmonicDataset):
    """
    An abstract super-class containing functionality common to the Key Transition and
    Key Sequence Datasets. Both of these must unwrap their inputs in the __getitem__
    method.
    """

    def __init__(
        self,
        transform: Callable = None,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        input_mask: List[int] = None,
    ):
        """
        A base class for for key sequence and key transition datasets.

        Parameters
        ----------
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        reduction : Dict[ChordType, ChordType]
            A reduction to apply to the input chord vectors when returning data with
            __getitem__. These are not applied when the data is loaded, but only when
            it is returned. So it is safe to change this after initialization.
        use_inversions : bool
            True to use inversions in the input chord vectors when returning data with
            __getitem__. False to reduce all chords to root position. This is not
            applied when the data is loaded, but only when it is returned. So it is
            safe to change this after initialization.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        super().__init__(
            transform=transform,
            pad_targets=False,
            save_padded_input_lengths=False,
            input_mask=input_mask,
        )
        self.reduction = reduction
        self.use_inversions = use_inversions

        self.piece_lengths = []
        self.key_change_replacements = []

    def __len__(self) -> int:
        if not self.in_ram:
            with h5py.File(self.h5_path, "r") as h5_file:
                return int(np.sum(h5_file["piece_lengths"]))
        return int(np.sum(self.piece_lengths))

    def __getitem__(self, item, transposition: int = None) -> Dict:
        def get_piece_index(item: int) -> Tuple[int, bool]:
            """
            Get the index of the piece from which the given item should be drawn.

            Parameters
            ----------
            item : int
                The index of the input/output we are looking for.
            transposition : int
                Not used. (Kept so the signature is the same as overriden function.)

            Returns
            -------
            piece_idx : int
                The index of the piece to which the item belongs.
            prev_in_piece : bool
                True if the previous item (item - 1) belongs to the same piece.
                False otherwise.

            Raises
            ------
            ValueError
                If item is greater than the size of this dataset.
            """
            for piece_idx, piece_length in enumerate(self.piece_lengths):
                if item < piece_length:
                    return piece_idx, item > 0
                item -= piece_length
            raise ValueError(f"Invalid item {item} requested for dataset of length {len(self)}")

        piece_idx, prev_in_piece = get_piece_index(item)

        if not self.in_ram:
            assert self.h5_path is not None, "Data must be either in ram or in an h5_file."
            with h5py.File(self.h5_path, "r") as h5_file:
                piece_input = h5_file["inputs"][piece_idx]
                input_length = h5_file["input_lengths"][item]
                start_index = h5_file["input_lengths"][item - 1] if prev_in_piece else 0
                key_change_replacement = h5_file["key_change_replacements"][item]
                target = h5_file["targets"][item]

        else:
            piece_input = self.inputs[piece_idx]
            input_length = self.input_lengths[item]
            start_index = self.input_lengths[item - 1] if prev_in_piece else 0
            if self.max_input_length is None:
                self.max_input_length = np.max(self.input_lengths)
            key_change_replacement = self.key_change_replacements[item]
            target = self.targets[item]

            if len(piece_input) < self.max_input_length:
                padded_input = np.zeros((self.max_input_length + 1, piece_input.shape[1]))
                padded_input[: len(piece_input)] = piece_input
                piece_input = padded_input

        # Create target list (if KTD)
        if isinstance(self, KeyTransitionDataset):
            targets = np.full(len(piece_input), -100)
            targets[start_index + 1 : input_length] = 0
            if target == 1:
                targets[input_length] = target
        else:
            targets = target

        # Add key_replacement as last input vector
        if target == 1 or isinstance(self, KeySequenceDataset):
            piece_input[input_length] = key_change_replacement
            input_length += 1

        data = {
            "targets": targets,
            "inputs": piece_input,
            "input_lengths": input_length,
        }

        return self.finalize_data(data, item)

    def reduce(self, data: Dict, transposition: int = None):
        reduce_chord_types(data["inputs"], self.reduction, pad=True)
        if not self.use_inversions:
            remove_chord_inversions(data["inputs"], pad=True)


class KeyTransitionDataset(KeyHarmonicDataset):
    """
    A dataset representing key change locations.

    There is 1 input sequence per key change per piece.
    In each sequence, there is one input vector per chord symbol, from the beginning of the piece
    leading up to each key change. Essentially, this is the CSM's input up to each key change,
    plus one more symbol with the chord relative to the previous key.

    The targets are a list of targets for each input, where -100 means to ignore the output,
    and 0 and 1 are "real" targets.
    """

    train_batch_size = 32
    valid_batch_size = 64
    chunk_size = 256

    def __init__(
        self,
        pieces: List[Piece],
        transform: Callable = None,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        input_mask: List[int] = None,
    ):
        """
        Create a key transition dataset from the given pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to get the data from.
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        reduction : Dict[ChordType, ChordType]
            A reduction to apply to the input chord vectors when returning data with
            __getitem__. These are not applied when the data is loaded, but only when
            it is returned. So it is safe to change this after initialization.
        use_inversions : bool
            True to use inversions in the input chord vectors when returning data with
            __getitem__. False to reduce all chords to root position. This is not
            applied when the data is loaded, but only when it is returned. So it is
            safe to change this after initialization.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        super().__init__(
            transform=transform,
            reduction=reduction,
            use_inversions=use_inversions,
            input_mask=input_mask,
        )
        self.inputs = []
        self.targets = []

        self.piece_lengths = []
        self.key_change_replacements = []
        self.input_lengths = []

        for piece in pieces:
            piece_input = []
            chords = piece.get_chords()
            key_changes = piece.get_key_change_indices()
            keys = piece.get_keys()
            key_vector_length = 1 + get_key_change_vector_length(
                keys[0].tonic_type,
                one_hot=False,
            )

            self.piece_lengths.append(len(keys))

            self.input_lengths.extend(key_changes[1:])
            self.input_lengths.append(len(chords))

            self.targets.extend([1] * (len(keys) - 1))
            self.targets.append(0)

            for prev_key, key, start, end in zip(
                [None] + list(keys[:-1]), keys, key_changes, list(key_changes[1:]) + [len(chords)]
            ):
                chord_vectors = np.vstack([chord.to_vec(pad=True) for chord in chords[start:end]])
                key_vectors = np.zeros((len(chord_vectors), key_vector_length))

                if prev_key is not None:
                    key_vectors[0, :-1] = prev_key.get_key_change_vector(key)
                    key_vectors[0, -1] = 1

                self.key_change_replacements.append(
                    np.concatenate(
                        (
                            chords[end].to_vec(relative_to=key, pad=True),
                            np.zeros(key_vector_length),
                        )
                    )
                    if end != len(chords)
                    else np.zeros(chord_vectors.shape[1] + key_vector_length)
                )

                piece_input.append(np.hstack([chord_vectors, key_vectors]))

            self.inputs.append(np.vstack(piece_input))

        if len(self.key_change_replacements) > 0:
            self.key_change_replacements = np.vstack(self.key_change_replacements)


class KeySequenceDataset(KeyHarmonicDataset):
    """
    A dataset representing key changes.

    There is 1 input sequence per key change per piece.
    In each sequence, there is one input vector per chord symbol, from the beginning of the piece
    leading up to each key change. Essentially, this is the CSM's input up to each key change,
    plus one more symbol with the chord relative to the previous key.

    There is one target per input list: a one-hot index representing the key change (transposition
    and new mode of the new key).
    """

    train_batch_size = 32
    valid_batch_size = 64
    chunk_size = 256

    def __init__(
        self,
        pieces: List[Piece],
        transform: Callable = None,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        input_mask: List[int] = None,
    ):
        """
        Create a key transition dataset from the given pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to get the data from.
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        reduction : Dict[ChordType, ChordType]
            A reduction to apply to the input chord vectors when returning data with
            __getitem__. These are not applied when the data is loaded, but only when
            it is returned. So it is safe to change this after initialization.
        use_inversions : bool
            True to use inversions in the input chord vectors when returning data with
            __getitem__. False to reduce all chords to root position. This is not
            applied when the data is loaded, but only when it is returned. So it is
            safe to change this after initialization.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        """
        super().__init__(
            transform=transform,
            reduction=reduction,
            use_inversions=use_inversions,
            input_mask=input_mask,
        )
        self.inputs = []
        self.targets = []

        self.piece_lengths = []
        self.key_change_replacements = []
        self.input_lengths = []

        for piece in pieces:
            piece_input = []
            chords = piece.get_chords()
            key_changes = piece.get_key_change_indices()
            keys = piece.get_keys()
            key_vector_length = 1 + get_key_change_vector_length(
                keys[0].tonic_type,
                one_hot=False,
            )

            if len(keys) <= 1:
                continue

            self.piece_lengths.append(len(keys) - 1)
            self.input_lengths.extend(key_changes[1:])

            for prev_key, key, start, end in zip(
                [None] + list(keys[:-2]), keys[:-1], key_changes[:-1], key_changes[1:]
            ):
                chord_vectors = np.vstack([chord.to_vec(pad=True) for chord in chords[start:end]])
                key_vectors = np.zeros((len(chord_vectors), key_vector_length))

                if prev_key is not None:
                    try:
                        self.targets.append(prev_key.get_key_change_one_hot_index(key))
                        key_vectors[0, :-1] = prev_key.get_key_change_vector(key)
                        key_vectors[0, -1] = 1
                    except ValueError:
                        # Key change pitch outside of valid relative range
                        logging.warning(
                            "Key change from %s to %s falls outside of valid range. Not generating "
                            "as a key change for the KSM",
                            prev_key,
                            key,
                        )
                        self.targets.append(-1)

                self.key_change_replacements.append(
                    np.concatenate(
                        (
                            chords[end].to_vec(relative_to=key, pad=True),
                            np.zeros(key_vector_length),
                        )
                    )
                )

                piece_input.append(np.hstack([chord_vectors, key_vectors]))

            self.targets.append(keys[-2].get_key_change_one_hot_index(keys[-1]))
            self.inputs.append(np.vstack(piece_input))

        if len(self.key_change_replacements) > 0:
            self.key_change_replacements = np.vstack(self.key_change_replacements)


class KeyPostProcessorDataset(HarmonicDataset):
    """
    A dataset taking in a sequence of absolute chord symbols, and outputting an aligned
    sequence of absolute keys.

    The inputs are one list per piece.
    Each input list is a list of absolute chord vectors.

    The targets are the one-hot index of the absolute key for each chord in the input.
    """

    train_batch_size = 32
    valid_batch_size = 64
    chunk_size = 256

    def __init__(
        self,
        pieces: List[Piece],
        transform: Callable = None,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        transposition_range: Union[List[int], Tuple[int, int]] = (0, 0),
        input_mask: List[int] = None,
        dummy_targets: bool = False,
        scheduled_sampling_h5_path: Union[str, Path] = None,
        scheduled_sampling_prob: float = 0.0,
    ):
        """
        Create a chord sequence dataset from the given pieces.

        Parameters
        ----------
        pieces : List[Piece]
            The pieces to get the data from.
        transform : Callable
            A function to transform numpy arrays returned by __getitem__ by default.
        reduction : Dict[ChordType, ChordType]
            A chord type reduction to apply to the input chords when returning data with
            __getitem__. This is not applied when the data is loaded, but only when it is
            returned. So it is safe to change this after initialization.
        use_inversions : bool
            True to use input inversions when returning data with __getitem__. False to reduce
            all input chords to root position. This is not applied when the data is loaded, but
            only when it is returned. So it is safe to change this after initialization.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each chord and target key of the
            dataset. Each __getitem__ call will return every possible transposition in this
            (min, max) range, inclusive on each side. The transpositions are measured in
            whatever PitchType is used in the dataset.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask.
        dummy_targets : bool
            Use dummy targets for this data creation.
        scheduled_sampling_h5_path : Union[str, Path]
            An h5 dataset file to load scheduled sampling data from.
        scheduled_sampling_prob : float
            The probability of changing a ground truth input chord into one from scheduled
            sampling.
        """
        super().__init__(transform=transform, input_mask=input_mask)

        self.reduction = reduction
        self.use_inversions = use_inversions
        self.transposition_range = transposition_range
        self.dummy_targets = dummy_targets

        self.scheduled_sampling_prob = scheduled_sampling_prob
        self.scheduled_sampling_h5_path = scheduled_sampling_h5_path

        self.inputs = []
        self.input_lengths = []
        self.targets = []
        self.target_lengths = []

        self.target_pitch_type = []
        self.num_transpositions = self.transposition_range[1] - self.transposition_range[0] + 1

        for piece in pieces:
            self.inputs.append(
                np.vstack([chord.to_vec(absolute=True) for chord in piece.get_chords()])
            )
            self.input_lengths.append(len(piece.get_chords()))
            self.target_lengths.append(len(piece.get_chords()))
            self.targets.append(
                [
                    get_key_one_hot_index(chord.key_mode, chord.key_tonic, chord.pitch_type)
                    for chord in piece.get_chords()
                ]
            )

        if len(pieces) > 0 and len(pieces[0].get_chords()) > 0:
            self.target_pitch_type.append(pieces[0].get_chords()[0].pitch_type)

    def load_scheduled_sampling_data(self) -> Union[List, None]:
        """
        Load scheduled sampling data in from the h5 file at the path in
        self.scheduled_sampling_h5_data. This should be called only after all other
        data is loaded. The data will be loaded into self.scheduled_sampling_data
        """
        if self.scheduled_sampling_h5_path is None:
            self.scheduled_sampling_data = None
            return

        with h5py.File(self.scheduled_sampling_h5_path, "r") as h5_file:
            sched_outputs = np.array(h5_file["outputs"], dtype=np.float16)
            pitch_type = PitchType(h5_file["pitch_type"][0])
            use_inversions = h5_file["use_inversions"][0]
            reduction = (
                {ChordType(red[0]): ChordType(red[1]) for red in h5_file["reduction"]}
                if "reduction" in h5_file
                else None
            )

        self.scheduled_sampling_data = np.copy(self.inputs)
        output_labels = np.array(
            get_chord_from_one_hot_index(
                slice(None),
                pitch_type,
                use_inversions=use_inversions,
                reduction=reduction,
            )
        )[np.argmax(sched_outputs, axis=1)]

        start_idx = 0
        for piece_idx, input_length in enumerate(self.input_lengths):
            for output_idx, (root, chord_type, inversion) in enumerate(
                output_labels[start_idx : start_idx + input_length]
            ):
                self.scheduled_sampling_data[piece_idx, output_idx] = update_chord_vector_info(
                    self.scheduled_sampling_data[piece_idx, output_idx],
                    root_pitch=root,
                    bass_pitch=get_bass_note(chord_type, root, inversion, pitch_type),
                    chord_type=chord_type,
                    inversion=inversion,
                )

            start_idx += input_length

    def __len__(self):
        return super().__len__() * self.num_transpositions

    def __getitem__(self, item, transposition: int = None) -> Dict:
        orig_item = item // self.num_transpositions
        transposition = self.transposition_range[0] + (item % self.num_transpositions)

        return super().__getitem__(orig_item, transposition=transposition)

    def reduce(self, data: Dict, transposition: int = None):
        if self.dummy_targets:
            return

        if transposition is None:
            transposition = 0

        try:
            if transposition != 0:
                for target_index, target in enumerate(data["targets"][: data["target_lengths"]]):
                    tonic, mode = get_key_from_one_hot_index(
                        int(target),
                        PitchType(self.target_pitch_type[0]),
                    )

                    data["targets"][target_index] = get_key_one_hot_index(
                        mode,
                        tonic + transposition,
                        PitchType(self.target_pitch_type[0]),
                    )

                for chord_index, chord_vector in enumerate(data["inputs"][: data["input_lengths"]]):
                    data["inputs"][chord_index] = transpose_chord_vector(
                        chord_vector, transposition
                    )

        except ValueError:
            # Something transposed out of the valid pitch range
            data["targets"] = [-1] * len(data["targets"])
            data["input_lengths"] = -1
            data["target_lengths"] = -1
            data["inputs"] *= 0
            return

    @staticmethod
    def get_scheduled_sampling_path(h5_dir_path: Union[Path, str], seed: int, split: str) -> Path:
        """
        Get the path to a scheduled sampling h5 dataset within the given directory.

        Parameters
        ----------
        h5_dir_path : Union[Path, str]
            The directory containing the scheduled sampling datasets.
        seed : int
            The seed of the dataset we want.
        split : str
            The split of the dataset we want.

        Returns
        -------
        h5_path : Path
            The path to the desired h5 dataset.
        """
        return Path(Path(h5_dir_path) / f"ccm_{split}_sched_samp_seed_{seed}.h5")


def h5_to_dataset(
    h5_path: Union[str, Path],
    dataset_class: HarmonicDataset,
    transform: Callable = None,
    dataset_kwargs: Dict[str, Any] = None,
) -> HarmonicDataset:
    """
    Load a harmonic dataset object from an h5 file into the given HarmonicDataset subclass.

    Parameters
    ----------
    h5_path : str or Path
        The h5 file to load the data from.
    dataset_class : HarmonicDataset
        The dataset class to load the data into and return.
    transform : Callable
        A function to pass each element of the dataset's returned data dicts to. For example,
        torch.from_numpy().
    dataset_kwargs : Dict[str, Any]
        Keyword arguments to pass to the dataset.init() call.

    Returns
    -------
    dataset : HarmonicDataset
        A HarmonicDataset of the given class, loaded with inputs and targets from the given
        h5 file.
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    dataset = dataset_class([], transform=transform, **dataset_kwargs)

    with h5py.File(h5_path, "r") as h5_file:
        assert "inputs" in h5_file, f"{h5_file} must have a dataset called inputs"
        assert "targets" in h5_file, f"{h5_file} must have a dataset called targets"
        dataset.h5_path = h5_path
        dataset.padded = False
        dataset.in_ram = False
        chunk_size = dataset.chunk_size

        # This data is small and can be assumed to fit in RAM in any case
        for key in ["piece_lengths", "key_change_replacements", "target_pitch_type"]:
            if key in h5_file:
                setattr(dataset, key, np.array(h5_file[key]))

        try:
            for data in ["input", "target"]:
                if f"{data}_lengths" in h5_file:
                    lengths = np.array(h5_file[f"{data}_lengths"])
                    data_list = []

                    for chunk_start in tqdm(
                        range(0, len(lengths), chunk_size),
                        desc=f"Loading {data} chunks from {h5_path}",
                    ):
                        chunk = h5_file[f"{data}s"][chunk_start : chunk_start + chunk_size]
                        chunk_lengths = lengths[chunk_start : chunk_start + chunk_size]
                        data_list.extend(
                            [
                                np.array(item[:length], dtype=np.float16)
                                for item, length in zip(chunk, chunk_lengths)
                            ]
                        )

                    setattr(dataset, f"{data}_lengths", lengths)
                    setattr(dataset, f"{data}s", data_list)

                else:
                    setattr(dataset, f"{data}s", np.array(h5_file[f"{data}s"], dtype=np.float16))

            dataset.in_ram = True

        except MemoryError:
            logging.exception("Dataset too large to fit into RAM. Reading from h5 file.")
            dataset.padded = True

    return dataset


def pad_array(array: List[np.array]) -> Tuple[np.array, np.array]:
    """
    Pad the given list, whose elements must only match in dimensions past the first, into a
    numpy nd-array of equal dimensions.

    Parameters
    ----------
    array : List[np.array]
        A list of numpy ndarrays. The shape of each ndarray must match in every dimension except
        the first.

    Returns
    -------
    padded_array : np.array
        The given list, packed into a numpy nd-array. Since the first dimension of each given
        nested numpy array need not be equal, each is padded with zeros to match the longest.
    array_lengths : np.array
        The size of the first dimension of each nested numpy nd-array before padding. Using this,
        the original array[i] can be gotten with padded_array[i, :array_lengths[i]].
    """
    array_lengths = np.array([len(item) for item in array])

    full_array_size = [len(array), max(array_lengths)]
    if len(array[0].shape) > 1:
        full_array_size += list(array[0].shape)[1:]
    full_array_size = tuple(full_array_size)

    padded_array = np.zeros(full_array_size, dtype=np.float16)
    for index, item in enumerate(array):
        padded_array[index, : len(item)] = item

    return padded_array, array_lengths


def get_split_file_ids_and_pieces(
    data_dfs: Dict[str, pd.DataFrame] = None,
    xml_and_csv_paths: Dict[str, List[Union[str, Path]]] = None,
    splits: Iterable[float] = (0.8, 0.1, 0.1),
    seed: int = None,
) -> Tuple[Iterable[Iterable[int]], Iterable[Piece]]:
    """
    Get the file_ids that should go in each split of a split dataset.

    Parameters
    ----------
    data_dfs : Dict[str, pd.DataFrame]
        If using dataframes, a mapping of 'files', 'measures', 'chords', and 'notes' dfs.
    xml_and_csv_paths : Dict[str, List[Union[str, Path]]]
        If using the MusicXML ('xmls') and label csvs ('csvs'), a list of paths of the
        matching xml and csv files.
    splits : Iterable[float]
        An Iterable of floats representing the proportion of pieces which will go into each split.
        This will be normalized to sum to 1.
    seed : int
        A numpy random seed, if given.

    Returns
    -------
    split_ids : Iterable[Iterable[int]]
        An iterable, the length of `splits` containing the file_ids for each data point in each
        split.
    pieces : Iterable[Iterable[Piece]]
        The loaded Pieces of each split.
    """
    assert sum(splits) != 0
    splits = np.array(splits) / sum(splits)

    if seed is not None:
        np.random.seed(seed)

    indexes = []
    pieces = []

    if data_dfs is not None:
        for i in tqdm(data_dfs["files"].index):
            file_name = (
                f"{data_dfs['files'].loc[i].corpus_name}/{data_dfs['files'].loc[i].file_name}"
            )
            logging.info("Parsing %s (id %s)", file_name, i)

            dfs = [data_dfs["chords"], data_dfs["measures"], data_dfs["notes"]]
            names = ["chords", "measures", "notes"]
            exists = [i in df.index.get_level_values(0) for df in dfs]

            if not all(exists):
                for exist, name in zip(exists, names):
                    if not exist:
                        logging.warning(
                            "%s_df does not contain %s data (id %s).", name, file_name, i
                        )
                continue

            try:
                piece = get_score_piece_from_data_frames(
                    data_dfs["notes"].loc[i], data_dfs["chords"].loc[i], data_dfs["measures"].loc[i]
                )
                if piece.get_chords() is None or len(piece.get_chords()) == 0:
                    raise ValueError("No valid chords in Piece.")
                pieces.append(piece)
                indexes.append(i)
            except Exception:
                logging.exception("Error parsing index %s", i)
                continue

    elif xml_and_csv_paths is not None:
        for i, (xml_path, csv_path) in tqdm(
            enumerate(zip(xml_and_csv_paths["xmls"], xml_and_csv_paths["csvs"])),
            desc="Parsing MusicXML files",
            total=len(xml_and_csv_paths["xmls"]),
        ):
            piece = get_score_piece_from_music_xml(xml_path, csv_path)
            pieces.append(piece)
            indexes.append(i)

    # Shuffle the pieces and the df_indexes the same way
    shuffled_indexes = np.arange(len(indexes))
    np.random.shuffle(shuffled_indexes)
    pieces = np.array(pieces)[shuffled_indexes]
    indexes = np.array(indexes)[shuffled_indexes]

    split_pieces = []
    split_indexes = []
    prop = 0
    for split_prop in splits:
        start = int(round(prop * len(pieces)))
        prop += split_prop
        end = int(round(prop * len(pieces)))
        length = end - start

        if length == 0:
            split_pieces.append([])
            split_indexes.append([])
        elif length == 1:
            split_pieces.append([pieces[start]])
            split_indexes.append([indexes[start]])
        else:
            split_pieces.append(pieces[start:end])
            split_indexes.append(indexes[start:end])

    return split_indexes, split_pieces


def get_dataset_splits(
    datasets: Iterable[HarmonicDataset],
    data_dfs: Dict[str, pd.DataFrame] = None,
    xml_and_csv_paths: Dict[str, List[Union[str, Path]]] = None,
    splits: Iterable[float] = (0.8, 0.1, 0.1),
    seed: int = None,
) -> Tuple[List[List[HarmonicDataset]], List[List[int]], List[List[Piece]]]:
    """
    Get datasets representing splits of the data in the given DataFrames.

    Parameters
    ----------
    datasets : Iterable[HarmonicDataset]
        An Iterable of HarmonicDataset class objects, each representing a different type of
        HarmonicDataset subclass to make a Dataset from. These are all passed so that they will
        have identical splits.
    data_dfs : Dict[str, pd.DataFrame]
        If using dataframes, a mapping of 'files', 'measures', 'chords', and 'notes' dfs.
    xml_and_csv_paths : Dict[str, List[Union[str, Path]]]
        If using the MusicXML ('xmls') and label csvs ('csvs'), a list of paths of the
        matching xml and csv files.
    splits : Iterable[float]
        An Iterable of floats representing the proportion of pieces which will go into each split.
        This will be normalized to sum to 1.
    seed : int
        A numpy random seed, if given.

    Returns
    -------
    dataset_splits : List[List[HarmonicDataset]]
        An iterable, the length of `dataset` representing the splits for each given dataset type.
        Each element is itself an iterable the length of `splits`.
    split_ids : List[List[int]]
        A list the length of `splits` containing the file_ids for each data point in each split.
    split_pieces : List[List[Piece]]
        A list of the pieces in each split.
    """
    split_ids, split_pieces = get_split_file_ids_and_pieces(
        data_dfs=data_dfs,
        xml_and_csv_paths=xml_and_csv_paths,
        splits=splits,
        seed=seed,
    )

    dataset_splits = np.full((len(datasets), len(splits)), None)
    for split_index, (split_prop, pieces) in enumerate(zip(splits, split_pieces)):
        if len(pieces) == 0:
            logging.warning(
                "Split %s with prop %s contains no pieces. Returning None for those.",
                split_index,
                split_prop,
            )
            continue

        for dataset_index, dataset_class in enumerate(datasets):
            dataset_splits[dataset_index][split_index] = dataset_class(pieces)

    return dataset_splits, split_ids, split_pieces


def transform_input_mask_to_binary(input_mask: List[int], input_length: int) -> List[int]:
    """
    Transform a given input mask into binary form. Specifically, if there are any
    values in the given mask > 1, a new list is returned of length equal to the given
    length, with 0s in the indexes given by the input mask and 1s everywhere else.

    Parameters
    ----------
    input_mask : List[int]
        A mask to use for inputs, either binary, with 0s indicating which indexes to mask
        (in which case it is returned unchanged), or non-binary, whose values are indexes
        to mask.
    input_length : int
        The length to make the resulting binary input mask.

    Returns
    -------
    input_mask : List[int]
        The input mask, if it is already binary and the correct length, or a new input
        mask of length input_length with 0s in the indexes contained in the given
        input_mask and 1s everywhere else.
    """
    if len(input_mask) == input_length and max(input_mask) <= 1:
        return input_mask

    binary_mask = np.ones(input_length, dtype=int)
    if len(input_mask) > 0:
        binary_mask[np.array(input_mask)] = 0

    return binary_mask


DATASETS = {
    "ctm": ChordTransitionDataset,
    "csm": ChordSequenceDataset,
    "ccm": ChordClassificationDataset,
    "ktm": KeyTransitionDataset,
    "ksm": KeySequenceDataset,
    "kppm": KeyPostProcessorDataset,
}
