"""Utility functions for evaluating model outputs."""
import os
from typing import List

import numpy as np
import pandas as pd

import torch

import harmonic_utils as hu
import corpus_utils as cu
from harmonic_inference.data.harmonic_inference_data import MusicScoreDataset


def get_conf_mat(labels: List[int], outputs: torch.tensor,
                 normalize: bool = True) -> np.ndarray:
    """
    Get a confusion matrix from a model output.

    Parameters
    ----------
    labels : list
        A list of the one-hot target chord for each data point.

    outputs : torch.tensor
        A tensor of the outputs of the model for each data point.
        outputs[i, c] is the output of the model for ith data point of chord index c.

    normalize : bool
        True to normalize rows of the conf_mat. False to stay with counts.

    Returns
    -------
    conf_mat : np.ndarray
        A 2-d confusion matrix. conf_mat[i, j] is the model outputting chord j for ground
        truth chord i.
    """
    label_strings = hu.get_one_hot_labels()

    conf_mat = np.zeros((len(label_strings), len(label_strings)))

    for target, out in zip(labels, outputs):
        conf_mat[target, out.argmax()] += 1

    # Normalize rows
    if normalize:
        conf_mat /= np.sum(conf_mat, axis=1, keepdims=True)
        conf_mat = np.where(np.isnan(conf_mat), 0, conf_mat)

    return conf_mat



def get_correct_and_incorrect_indexes(labels: List[int], outputs: torch.tensor) -> (List[int],
                                                                                    List[int]):
    """
    Get the indices of correct and incorrect model outputs.

    Parameters
    ----------
    labels : list
        A list of the one-hot target chord for each data point.

    outputs : torch.tensor
        A tensor of the outputs of the model for each data point.
        outputs[i, c] is the output of the model for ith data point of chord index c.

    Returns
    -------
    correct : list
        A list of all of the indexes of data points for which the output was correct.

    incorrect : list
        A list of all of the indexes of data points for which the output was incorrect.
    """
    correct = []
    incorrect = []

    for index, label in enumerate(labels):
        if outputs[index].argmax() == label:
            correct.append(index)
        else:
            incorrect.append(index)

    return correct, incorrect



def print_result(index: int, labels: List[int], outputs: torch.tensor,
                 limit: int = None, prob: bool = True) -> None:
    """
    Print the model's output of a given data point in a nice format.

    Parameters
    ----------
    index : int
        The index of the data point to print.

    labels : list
        A list of the one-hot target chord for each data point.

    outputs : torch.tensor
        A tensor of the outputs of the model for each data point.
        outputs[i, c] is the output of the model for ith data point of chord index c.

    limit : int
        If given, only print this many most likely model outputs, in order.

    prob : bool
        True to convert outputs to probabilities using a softmax. False to return raw model output.
    """
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    label_strings = hu.get_one_hot_labels()
    outputs_this = softmax(outputs[index]) if prob else outputs[index]
    if limit is None:
        limit = len(label_strings)

    print(f'Correct: {labels[index]} ({label_strings[labels[index]]})')
    print(f'Guessed: {outputs[index].argmax()} ({label_strings[outputs_this.argmax()]})')
    print('\n'.join([f'{label_string}: {output}'
                     for output, label_string in
                     sorted(zip(outputs_this, label_strings), reverse=True)[:limit]]))



def get_input_df_rows(index: int, dataset: MusicScoreDataset) -> (pd.Series, pd.DataFrame):
    """
    Get the input chord and note_df rows for a particular data point from a dataset object.

    Parameters
    ----------
    index : int
        The index of the data point whose input to return.

    dataset : MusicScoreDataset
        The dataset from which to draw the input. It must include raw DataFrames.

    Returns
    -------
    chord : pd.Series
        The chord of the given data point.

    onset_notes : pd.DataFrame
        A DataFrame containing the notes which onset during the given chord.

    all_notes : pd.DataFrame
        A DataFrame containing all notes which overlap the given chord at all.
    """
    chord_id = dataset[index]['chord_index']
    chord = dataset.chords.loc[tuple(chord_id)]

    onset_notes = cu.get_notes_during_chord(chord, dataset.notes, onsets_only=True)
    all_notes = cu.get_notes_during_chord(chord, dataset.notes, onsets_only=False)

    return chord, onset_notes, all_notes



def get_correct_ranks(labels: List[int], outputs: torch.tensor) -> (List[int], List[List[int]]):
    """
    Get the rank of the correct chord label in the model's output for each data point.

    Parameters
    ----------
    labels : list
        A list of the one-hot target chord for each data point.

    outputs : torch.tensor
        A tensor of the outputs of the model for each data point.
        outputs[i, c] is the output of the model for ith data point of chord index c.

    Returns
    -------
    correct_ranks : list
        The rank of the correct chord in the model's output for each data point.

    indexes_by_rank : list
        The same info, but sorted into lists of the indexes of all data points with each rank.
    """
    correct_ranks = [[i for _, i in sorted(zip(out, range(len(out))), reverse=True)].index(label)
                     for out, label in zip(outputs, labels)]

    indexes_by_rank = [[] for _ in range(len(outputs[0]))]
    for i, rank in enumerate(correct_ranks):
        indexes_by_rank[rank].append(i)

    return correct_ranks, indexes_by_rank



def load_eval_df(filename: str) -> pd.DataFrame:
    """
    Load evaluation results from a csv into a dataframe.

    Parameters
    ----------
    filename : string
        The filename of the DataFrame to load.

    Returns
    -------
    df : pd.DataFrame
        The loaded evaluation DataFrame, like the one generated by get_eval_df.
    """
    return pd.read_csv(filename, index_col=0)



def write_eval_df(eval_df: pd.DataFrame, filename: str) -> None:
    """
    Write evaluation results from a pd.DataFrame into a csv file.

    Parameters
    ----------
    eval_df : pd.DataFrame
        An evaluation DataFrame created by get_eval_df.

    filename : string
        The filename to which to write the DataFrame.
    """
    dirname = os.path.dirname(filename)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    eval_df.to_csv(filename)



def get_eval_df(labels: List[int], outputs: torch.tensor,
                dataset: MusicScoreDataset) -> pd.DataFrame:
    """
    Get a DataFrame where each row is a data point and the columns are different statistics
    of that data point. This is useful for filtering and grouping by various values.

    Parameters
    ----------
    labels : list
        A list of the one-hot target chord for each data point.

    outputs : torch.tensor
        A tensor of the outputs of the model for each data point.
        outputs[i, c] is the output of the model for ith data point of chord index c.

    dataset : MusicScoreDataset
        The dataset corresponding to the labels and outputs. It must contain the raw
        dataframe information for certain statistics.

    Returns
    -------
    eval_df : pd.DataFrame
        A DataFrame containing various metrics for each data point for easy filtering and grouping.
    """
    contains_dfs = dataset.chords is not None
    label_strings = hu.get_one_hot_labels()
    label_roots = [label.split(':')[0] for label in label_strings]
    label_types = [label.split(':')[1] for label in label_strings]

    df_data = dict()
    df_data['rank'], _ = get_correct_ranks(labels, outputs)
    df_data['correct'] = [rank == 0 for rank in df_data['rank']]

    df_data['correct_chord'] = [None] * len(dataset)
    df_data['correct_root'] = [None] * len(dataset)
    df_data['correct_type'] = [None] * len(dataset)
    for i, label in enumerate(labels):
        df_data['correct_chord'][i] = label_strings[label]
        df_data['correct_root'][i] = label_roots[label]
        df_data['correct_type'][i] = label_types[label]

    df_data['guessed_chord'] = [None] * len(dataset)
    df_data['guessed_root'] = [None] * len(dataset)
    df_data['guessed_type'] = [None] * len(dataset)
    for i, out in enumerate(outputs):
        label = out.argmax()
        df_data['guessed_chord'][i] = label_strings[label]
        df_data['guessed_root'][i] = label_roots[label]
        df_data['guessed_type'][i] = label_types[label]

    df_data['num_notes'] = [None] * len(dataset)
    df_data['chord_onset_level'] = [None] * len(dataset)
    df_data['chord_offset_level'] = [None] * len(dataset)
    df_data['chord_duration'] = [None] * len(dataset)
    for i, sample in enumerate(dataset):
        df_data['num_notes'][i] = len(sample['notes'])
        df_data['chord_onset_level'][i] = sample['chord']['rhythm'][0]
        df_data['chord_offset_level'][i] = sample['chord']['rhythm'][1]
        df_data['chord_duration'][i] = sample['chord']['rhythm'][2]

    if contains_dfs:
        df_data['num_all_notes'] = []
        df_data['piece_id'] = []
        df_data['chord_id'] = []
        for i in range(len(dataset)):
            chord, _, all_notes = get_input_df_rows(i, dataset)
            df_data['num_all_notes'].append(len(all_notes))
            df_data['piece_id'].append(chord.name[0])
            df_data['chord_id'].append(chord.name[1])

    return pd.DataFrame(df_data)
