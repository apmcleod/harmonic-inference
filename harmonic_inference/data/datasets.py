from typing import List, Iterable, Union, Tuple, Callable
from pathlib import Path
import logging
import shutil
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset

from harmonic_inference.data.piece import Piece, ScorePiece
from harmonic_inference.data.data_types import KeyMode
import harmonic_inference.utils.harmonic_utils as hu
import harmonic_inference.utils.harmonic_constants as hc


class HarmonicDataset(Dataset):
    def __init__(self, transform=None):
        self.h5_path = None
        self.padded = False
        self.in_ram = True
        self.transform = transform

    def __len__(self):
        if not self.in_ram:
            with h5py.File(self.h5_path, 'r') as h5_file:
                length = len(h5_file['inputs'])
            return length
        return len(self.inputs)

    def __getitem__(self, item):
        keys = ["inputs", "targets", "input_lengths", "target_lengths"]
        if not self.in_ram:
            assert self.h5_path is not None, "Data must be either in ram or in an h5_file."
            with h5py.File(self.h5_path, 'r') as h5_file:
                data = {key: h5_file[key][item] for key in keys if key in h5_file}
        else:
            data = {
                "inputs": self.inputs[item],
                "targets": self.targets[item],
            }
            if hasattr(self, "input_lengths"):
                if not hasattr(self, "max_input_length"):
                    self.max_input_length = max(self.input_lengths)
                padded_input = np.zeros(([self.max_input_length] + list(data["inputs"][0].shape)))
                padded_input[:len(data["inputs"])] = data["inputs"]
                data["inputs"] = padded_input
                data["input_lengths"] = self.input_lengths[item]

            if hasattr(self, "target_lengths"):
                if not hasattr(self, "max_target_length"):
                    self.max_target_length = max(self.target_lengths)
                padded_target = np.zeros(([self.max_target_length] + list(data["targets"][0].shape)))
                padded_target[:len(data["targets"])] = data["targets"]
                data["targets"] = padded_target
                data["target_lengths"] = self.target_lengths[item]

        if self.transform:
            data.update(
                {
                    key: self.transform(value)
                    for key, value in data.items()
                    if isinstance(value, np.ndarray)
                }
            )

        return data

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
        self.targets, self.target_lengths = pad_array(self.targets)
        self.inputs, self.input_lengths = pad_array(self.inputs)
        self.padded = True

    def to_h5(self, h5_path: Union[str, Path]):
        """
        Write this HarmonicDataset out to an h5 file, containing its inputs and targets.
        If the dataset is already being read from an h5py file, this simply copies that
        file (self.h5_path) over to the given location and updates self.h5_path.

        Parameters
        ----------
        h5_path : Union[str, Path]
            The filename of the h5 file to write to.
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
            except BaseException as e:
                logging.exception(
                    f"Error copying existing h5 file {self.h5_path} to {h5_path}:\n{e}"
                )
            return

        h5_file = h5py.File(h5_path, 'w')

        keys = ["inputs", "targets", "input_lengths", "target_lengths"]
        for key in keys:
            if hasattr(self, key):
                h5_file.create_dataset(key, data=getattr(self, key), compression="gzip")
        h5_file.close()


class ChordTransitionDataset(HarmonicDataset):
    """
    A dataset to detect chord transitions.

    Each input is the sequence of input vectors of a piece.

    Each target is a list of the indexes at which there is a chord change in that list of input
    vectors.
    """
    train_batch_size = 16
    valid_batch_size = 32
    chunk_size = 128

    def __init__(self, pieces: List[Piece], transform=None):
        super().__init__(transform=transform)
        self.targets = [piece.get_chord_change_indices() for piece in pieces]
        self.inputs = [
            np.vstack([note.to_vec() for note in piece.get_inputs()]) for piece in pieces
        ]


class ChordClassificationDataset(HarmonicDataset):
    """
    A dataset to classify chords.

    Each input is a list of note vectors of the notes in each chord
    in a piece, plus some window length (2) on each end.

    The targets are the one_hot indexes of each of those chords.
    """
    train_batch_size = 128
    valid_batch_size = 256
    chunk_size = 256

    def __init__(self, pieces: List[Piece], transform=None):
        super().__init__(transform=transform)
        self.targets = np.array([
            chord.get_one_hot_index(relative=False, use_inversion=True)
            for piece in pieces
            for chord in piece.get_chords()
        ])
        self.inputs = []
        for piece in pieces:
            self.inputs.extend(piece.get_chord_note_inputs(window=2))

    def pad(self):
        self.inputs, self.input_lengths = pad_array(self.inputs)


class ChordSequenceDataset(HarmonicDataset):
    """
    A dataset representing chord sequences relative to the key.

    The inputs are one list per piece.
    Each input list is a list of a chord vectors with some additional information. Specifically,
    they are the size of a chord vector plus a key change vector, plus 1. These are usually a chord
    vector, relative to the current key (in which case the rest of the vector is 0). In the
    special case of the first chord after a key change, the key change vector is also populated and
    the additional slot is set to 1.

    The targets are the one-hot index of the following chord, relative to its key. Back-propagation
    should not be performed on the target of the last chord in each key section.
    """
    train_batch_size = 32
    valid_batch_size = 64
    chunk_size = 256

    def __init__(self, pieces: List[Piece], transform=None):
        super().__init__(transform=transform)
        self.inputs = []
        self.targets = []

        for piece in pieces:
            piece_input = []
            chords = piece.get_chords()
            key_changes = piece.get_key_change_indices()
            keys = piece.get_keys()
            key_vector_length = hc.NUM_PITCHES[keys[0].tonic_type] * len(KeyMode) + 1

            for prev_key, key, start, end in zip(
                [None] + list(keys[:-1]),
                keys,
                key_changes,
                list(key_changes[1:]) + [len(chords)]
            ):
                chord_vectors = np.vstack([chord.to_vec() for chord in chords[start:end]])
                key_vectors = np.zeros((len(chord_vectors), key_vector_length))
                if prev_key is not None:
                    key_vectors[0, prev_key.get_key_change_one_hot_index(key)] = 1
                    key_vectors[0, -1] = 1

                piece_input.append(np.hstack([chord_vectors, key_vectors]))

            self.inputs.append(np.vstack(piece_input))
            self.targets.append(np.array([chord.get_one_hot_index() for chord in chords]))


class KeyTransitionDataset(HarmonicDataset):
    """
    A dataset representing key change locations.

    The inputs are lists of relative chord vectors for
    each sequence of chords without a key change, plus the following one chord, relative to the
    current chord sequence's key.

    The targets are 1 if the last chord is on a chord change, and 0 otherwise
    (if the last chord is the last chord of a piece).
    """
    train_batch_size = 32
    valid_batch_size = 64
    chunk_size = 256

    def __init__(self, pieces: List[Piece], transform=None):
        super().__init__(transform=transform)
        self.inputs = []
        self.targets = []

        for piece in pieces:
            chords = piece.get_chords()
            key_changes = piece.get_key_change_indices()

            for key, start, end in zip(
                piece.get_keys(),
                key_changes,
                list(key_changes[1:]) + [len(chords)]
            ):
                target = 0
                # If not the last chord, extend the input vector by 1 and set target to 1
                if end != len(chords):
                    end += 1
                    target = 1
                self.targets.append(np.array([0] * (end - start - 1) + [target]))

                self.inputs.append(
                    np.vstack([
                        chord.to_vec(relative_to=key)
                        for chord in chords[start:end]
                    ])
                )


class KeySequenceDataset(HarmonicDataset):
    """
    A dataset representing key changes.

    The inputs are lists of relative chord vectors for
    each sequence of chords without a key change, plus the following one chord, all relative to
    the current chord sequence's key. The last chord sequence of each key is not used because
    it does not end in a key change.

    There is one target per input list: a one-hot index representing the key change (transposition
    and new mode of the new key).
    """
    train_batch_size = 64
    valid_batch_size = 32
    chunk_size = 256

    def __init__(self, pieces: List[Piece], transform=None):
        super().__init__(transform=transform)
        self.inputs = []
        self.targets = []

        for piece in pieces:
            chords = piece.get_chords()
            key_changes = piece.get_key_change_indices()
            keys = piece.get_keys()

            for key, next_key, start, end in zip(
                keys[:-1],
                keys[1:],
                key_changes[:-1],
                list(key_changes[1:])
            ):
                self.inputs.append(
                    np.vstack([
                        chord.to_vec(relative_to=key)
                        for chord in chords[start:end + 1]
                    ])
                )
                self.targets.append(key.get_key_change_one_hot_index(next_key))

    def pad(self):
        self.inputs, self.input_lengths = pad_array(self.inputs)


def h5_to_dataset(
    h5_path: Union[str, Path],
    dataset_class: HarmonicDataset,
    transform: Callable = None
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

    Returns
    -------
    dataset : HarmonicDataset
        A HarmonicDataset of the given class, loaded with inputs and targets from the given
        h5 file.
    """
    dataset = dataset_class([], transform=transform)

    with h5py.File(h5_path, 'r') as h5_file:
        assert 'inputs' in h5_file
        assert 'targets' in h5_file
        dataset.h5_path = h5_path
        dataset.padded = False
        dataset.in_ram = False
        chunk_size = dataset.chunk_size

        try:
            if 'input_lengths' in h5_file:
                dataset.input_lengths = np.array(h5_file['input_lengths'])
                dataset.inputs = []
                for chunk_start in tqdm(
                    range(0, len(dataset.input_lengths), chunk_size),
                    desc=f"Loading input chunks from {h5_path}",
                ):
                    input_chunk = h5_file['inputs'][chunk_start:chunk_start + chunk_size]
                    chunk_lengths = dataset.input_lengths[chunk_start:chunk_start + chunk_size]
                    dataset.inputs.extend(
                        [
                            list(item[:length])
                            for item, length
                            in zip(input_chunk, chunk_lengths)
                        ]
                    )
            else:
                dataset.inputs = np.array(h5_file['inputs'])

            if 'target_lengths' in h5_file:
                dataset.target_lengths = np.array(h5_file['target_lengths'])
                dataset.targets = []
                for chunk_start in tqdm(
                    range(0, len(dataset.target_lengths), chunk_size),
                    desc=f"Loading target chunks from {h5_path}",
                ):
                    target_chunk = h5_file['targets'][chunk_start:chunk_start + chunk_size]
                    chunk_lengths = dataset.target_lengths[chunk_start:chunk_start + chunk_size]
                    dataset.targets.extend(
                        [
                            list(item[:length])
                            for item, length
                            in zip(target_chunk, chunk_lengths)
                        ]
                    )
            else:
                dataset.targets = np.array(h5_file['targets'])
            dataset.in_ram = True

        except Exception:
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

    padded_array = np.zeros(full_array_size)
    for index, item in enumerate(array):
        padded_array[index, :len(item)] = item

    return padded_array, array_lengths


def get_dataset_splits(
    files: pd.DataFrame,
    measures: pd.DataFrame,
    chords: pd.DataFrame,
    notes: pd.DataFrame,
    datasets: Iterable[HarmonicDataset],
    splits: Iterable[float] = [0.8, 0.1, 0.1],
    seed: int = None,
) -> Iterable[Iterable[HarmonicDataset]]:
    """
    Get datasets representing splits of the data in the given DataFrames.

    Parameters
    ----------
    files : pd.DataFrame
        A DataFrame with data about all of the files in the DataFrames.
    measures : pd.DataFrame
        A DataFrame with information about the measures of the pieces in the data.
    chords : pd.DataFrame
        A DataFrame with information about the chords of the pieces in the data.
    notes : pd.DataFrame
        A DataFrame with information about the notes of the pieces in the data.
    datasets : Iterable[HarmonicDataset]
        An Iterable of HarmonicDataset class objects, each representing a different type of
        HarmonicDataset subclass to make a Dataset from. These are all passed so that they will
        have identical splits.
    splits : Iterable[float]
        An Iterable of floats representing the proportion of pieces which will go into each split.
        This will be normalized to sum to 1.
    seed : int
        A numpy random seed, if given.

    Returns
    -------
    dataset_splits : Iterable[Iterable[HarmonicDataset]]
        An iterable, the length of `dataset` representing the splits for each given dataset type.
        Each element is itself an iterable the length of `splits`.
    """
    assert sum(splits) != 0
    splits = np.array(splits) / sum(splits)

    if seed is not None:
        np.random.seed(seed)

    pieces = []
    df_indexes = []

    for i in tqdm(files.index):
        file_name = f'{files.loc[i].corpus_name}/{files.loc[i].file_name}'
        logging.info(f"Parsing {file_name} (id {i})")

        dfs = [chords, measures, notes]
        names = ['chords', 'measures', 'notes']
        exists = [i in df.index.get_level_values(0) for df in dfs]

        if not all(exists):
            for exist, df, name in zip(exists, dfs, names):
                if not exist:
                    logging.warning(f'{name}_df does not contain {file_name} data (id {i}).')
            continue

        try:
            piece = ScorePiece(notes.loc[i], chords.loc[i], measures.loc[i])
            pieces.append(piece)
            df_indexes.append(i)
        except Exception as e:
            logging.exception(f"Error parsing index {i}: {e}")
            continue

    # Shuffle the pieces and the df_indexes the same way
    shuffled_indexes = np.arange(len(pieces))
    np.random.shuffle(shuffled_indexes)
    pieces = np.array(pieces)[shuffled_indexes]
    df_indexes = np.array(df_indexes)[shuffled_indexes]

    dataset_splits = np.full((len(datasets), len(splits)), None)
    prop = 0
    for split_index, split_prop in enumerate(splits):
        start = int(round(prop * len(pieces)))
        prop += split_prop
        end = int(round(prop * len(pieces)))

        if start == end:
            logging.warning(
                f"Split {split_index} with prop {split_prop} contains no pieces. Returning None "
                "for those."
            )
            continue

        for dataset_index, dataset_class in enumerate(datasets):
            dataset_splits[dataset_index][split_index] = dataset_class(pieces[start:end])

    return dataset_splits
