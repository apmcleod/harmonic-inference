"""Models that generate probability distributions over the pitches present in a given chord."""
import itertools
import logging
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.chord import (
    Chord,
    get_chord_pitches_target_vector_length,
    get_chord_pitches_vector_length,
)
from harmonic_inference.data.data_types import (
    MAJOR_MINOR_REDUCTION,
    TRIAD_REDUCTION,
    ChordType,
    PitchType,
)
from harmonic_inference.data.datasets import ChordPitchesDataset
from harmonic_inference.data.note import Note
from harmonic_inference.data.piece import Piece, ScorePiece, get_score_piece_from_dict, get_windows
from harmonic_inference.data.vector_decoding import get_relative_pitch_index
from harmonic_inference.utils.harmonic_constants import (
    CHORD_PITCHES,
    MAX_CHORD_PITCH_INTERVAL_TPC,
    NUM_PITCHES,
    NUM_RELATIVE_PITCHES,
    TPC_C,
)
from harmonic_inference.utils.harmonic_utils import absolute_to_relative, get_chord_inversion_count


class ChordPitchesModel(pl.LightningModule, ABC):
    """
    The base type for all Chord Pitches Models, which take as input sets of inputs and chords from
    Pieces, and output pitch presence probabilities for them.
    """

    def __init__(
        self,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType],
        use_inversions: bool,
        default_weight: float,
        learning_rate: float,
        input_mask: List[int],
        window: int,
    ):
        """
        Create a new base ChordPitchesModel with the given input and output formats.

        Parameters
        ----------
        input_pitch : PitchType
            What pitch type the model is expecting for chord roots and note pitches.
        output_pitch : PitchType
            The pitch type to use for outputs of this model.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the input chord types.
        use_inversions : bool
            Whether to use different inversions as different chords in the input.
        default_weight : float
            The weight to use in the loss function for chords whose targets are the default
            pitches for their chord type and root.
        learning_rate : float
            The learning rate.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        window : int
            The window size for data creation. As input to each chord, there will be this
            many additional input vectors on either side.
        """
        super().__init__()
        self.INPUT_PITCH = input_pitch
        self.OUTPUT_PITCH = output_pitch

        self.reduction = reduction
        self.use_inversions = use_inversions

        self.default_weight = default_weight

        self.lr = learning_rate

        self.input_mask = input_mask
        self.window = window

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """
        Get a kwargs dict that can be used to create a dataset for this model with
        the correct parameters.

        Returns
        -------
        dataset_kwargs : Dict[str, Any]
            A keyword args dict that can be used to create a dataset for this model with
            the correct parameters.
        """
        return {
            "reduction": self.reduction,
            "use_inversions": self.use_inversions,
            "input_mask": self.input_mask,
            "window": self.window,
        }

    def get_weights(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the weights to use for the loss calculation given a list of whether
        each chord contains the default pitches or not.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch Dictionary, containing the entry "is_default", which is:
            One boolean per input chord, with True if that chord contains only the default
            pitches and False otherwise.

        Returns
        -------
        weights : torch.Tensor
            A (batch_size x num_output_pitches) tensor where each row is all 1's
            for non-default inputs, and all self.default_weight for default inputs.
        """
        is_default = batch["is_default"]

        weights = torch.ones((len(is_default), self.output_dim), dtype=float)

        # Only use default_weight once the model has stagnated with default_weight 1
        if self.optimizers().param_groups[0]["lr"] != self.lr:
            weights[is_default, :] = self.default_weight

        return weights.to(self.device)

    def get_raw_output(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Load inputs from the batch data and return the raw, non-rounded output
        Tensor, directly from the model.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch Dictionary containing any needed inputs.

        Returns
        -------
        output : torch.Tensor
            A raw, unrounded output tensor, directly from the model.
        """
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]

        outputs = self(inputs[:, : torch.max(input_lengths)], input_lengths)

        return outputs

    def get_output(
        self, batch: Dict[str, torch.Tensor], return_notes: bool = False
    ) -> torch.Tensor:
        """
        Load inputs from the batch data and return a non-rounded output
        Tensor derived from the model.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch Dictionary containing any needed inputs.
        return_notes : bool
            Unused for a SimpleCPM.

        Returns
        -------
        output : torch.Tensor
            An unrounded output tensor derived from the model, modeling the pitches
            at intervals on the range [-13, 13] for TPC, and [0, 11] for MIDI
            above the root. Values closer to 1 indicate presence of the pitch.
        """
        return self.get_raw_output(batch)

    def get_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Load the correct targets from a given batch dictionary.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch Dictionary containing any needed inputs.

        Returns
        -------
        targets : torch.Tensor
            The appropriate targets to be used for this model's outputs.
        """
        return batch["targets"].float()[:, : torch.max(batch["input_lengths"])]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        targets = self.get_targets(batch)
        weights = self.get_weights(batch)
        outputs = self.get_raw_output(batch)

        loss = F.binary_cross_entropy(outputs, targets, weight=weights)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        all_targets = self.get_targets(batch)
        all_weights = self.get_weights(batch)
        all_outputs = self.get_raw_output(batch)

        for prefix, targets, outputs, weights in zip(
            ["", "default_", "non-default_"],
            [all_targets, all_targets[batch["is_default"]], all_targets[~batch["is_default"]]],
            [all_outputs, all_outputs[batch["is_default"]], all_outputs[~batch["is_default"]]],
            [all_weights, all_weights[batch["is_default"]], all_weights[~batch["is_default"]]],
        ):
            if len(targets) > 0:
                self.calculate_and_log_validation_metrics(prefix, targets, outputs, weights)

    def calculate_and_log_validation_metrics(
        self,
        prefix: str,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        prefix : str
            The prefix to use for each metric when logging. Should be something like "default_"
            or "non-default_".
        targets : torch.Tensor
            The model's targets.
        outputs : torch.Tensor
            The model's output.
        weights : torch.Tensor
            The weights to use in loss calculation.
        """
        rounded_outputs = outputs.round()

        pitch_correct = (rounded_outputs == targets).float()
        chord_correct = (torch.sum(pitch_correct, dim=1) == outputs.shape[1]).float().sum()
        pitch_correct = pitch_correct.sum()

        total_pitches = len(targets.flatten())
        total_chords = len(targets)

        pitch_acc = 100 * pitch_correct / total_pitches
        chord_acc = 100 * chord_correct / total_chords

        positive_target_mask = targets == 1
        positive_output_mask = rounded_outputs == 1

        tp = (positive_target_mask & positive_output_mask).sum().float()
        fp = (~positive_target_mask & positive_output_mask).sum().float()
        fn = (positive_target_mask & ~positive_output_mask).sum().float()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        loss = F.binary_cross_entropy(outputs, targets, weight=weights)

        self.log(f"{prefix}val_loss", loss)
        self.log(f"{prefix}val_pitch_acc", pitch_acc)
        self.log(f"{prefix}val_chord_acc", chord_acc)
        self.log(f"{prefix}val_precision", precision)
        self.log(f"{prefix}val_recall", recall)
        self.log(f"{prefix}val_f1", f1)

    def evaluate(self, dataset: ChordPitchesDataset) -> Dict[str, float]:
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        num_pitches = 0
        num_chords = 0
        tp = 0
        fp = 0
        fn = 0
        total_loss = 0
        total_pitch_acc = 0
        total_chord_acc = 0

        for batch in tqdm(dl, desc="Evaluating CPM"):
            targets = self.get_targets(batch)
            weights = self.get_weights(batch)
            outputs = self.get_raw_output(batch)

            rounded_outputs = outputs.round()

            pitch_correct = (rounded_outputs == targets).float()
            chord_correct = (torch.sum(pitch_correct, dim=1) == outputs.shape[1]).float().sum()
            pitch_correct = pitch_correct.sum()

            total_pitches = len(targets.flatten())
            total_chords = len(targets)

            total_pitch_acc += 100 * pitch_correct
            total_chord_acc += 100 * chord_correct

            positive_target_mask = targets == 1
            positive_output_mask = rounded_outputs == 1

            tp += (positive_target_mask & positive_output_mask).sum().float()
            fp += (~positive_target_mask & positive_output_mask).sum().float()
            fn += (positive_target_mask & ~positive_output_mask).sum().float()

            loss = F.binary_cross_entropy(outputs, targets, weight=weights)

            num_chords += total_chords
            num_pitches += total_pitches
            total_loss += loss * total_chords

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {
            "loss": (total_loss / num_chords).item(),
            "pitch_acc": (total_pitch_acc / num_pitches).item(),
            "chord_acc": (total_chord_acc / num_chords).item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    @abstractmethod
    def init_hidden(self, batch_size: int) -> Tuple[Variable, ...]:
        """
        Get initial hidden layers for this model.

        Parameters
        ----------
        batch_size : int
            The batch size to initialize hidden layers for.

        Returns
        -------
        hidden : Tuple[Variable, ...]
            A tuple of initialized hidden layers.
        """
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


class SimpleChordPitchesModel(ChordPitchesModel):
    """
    The most simple chord pitches model, with layers:
        1. Linear layer (embedding)
        2. Bi-LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer
    """

    def __init__(
        self,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        embed_dim: int = 128,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        default_weight: float = 1.0,
        learning_rate: float = 0.001,
        input_mask: List[int] = None,
        window: int = 2,
    ):
        """
        Create a new SimpleChordPitchesModel.

        Parameters
        ----------
        input_pitch : PitchType
            What pitch type the model is expecting for notes.
        output_pitch : PitchType
            The pitch type to use for outputs of this model. Used to derive the output length.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the input chord types.
        use_inversions : bool
            Whether to use different inversions as different chords in the input.
        embed_dim : int
            The size of the initial embedding layer.
        lstm_layers : int
            The number of Bi-LSTM layers to use.
        lstm_hidden_dim : int
            The size of each LSTM layer's hidden vector.
        hidden_dim : int
            The size of the output vector of the first linear layer.
        dropout : float
            The dropout proportion of the first linear layer's output.
        default_weight : float
            The weight to use in the loss function for chords whose targets are the default
            pitches for their chord type and root.
        learning_rate : float
            The learning rate.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        window : int
            The window size for data creation. As input to each chord, there will be this
            many additional input vectors on either side.
        """
        super().__init__(
            input_pitch,
            output_pitch,
            reduction,
            use_inversions,
            default_weight,
            learning_rate,
            input_mask,
            window,
        )
        self.save_hyperparameters()

        # Input and output derived from pitch_type and use_inversions
        self.input_dim = get_chord_pitches_vector_length(input_pitch)
        self.output_dim = get_chord_pitches_target_vector_length(input_pitch)

        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.input_dim, self.embed_dim)

        # LSTM hidden layer and depth
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 because bi-directional
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout1 = nn.Dropout(self.dropout)

    def init_hidden(self, batch_size: int) -> Tuple[Variable, Variable]:
        """
        Initialize the LSTM's hidden layer for a given batch size.

        Parameters
        ----------
        batch_size : int
            The batch size.
        """
        return (
            Variable(
                torch.zeros(
                    2 * self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device
                )
            ),
            Variable(
                torch.zeros(
                    2 * self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device
                )
            ),
        )

    def forward(self, inputs, lengths):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1).cpu()
        h_0, c_0 = self.init_hidden(batch_size)

        embed = F.relu(self.embed(inputs))

        packed_inputs = pack_padded_sequence(embed, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed_inputs, (h_0, c_0))
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)

        # Get lengths in proper format
        lstm_out_lengths_tensor = (
            lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand((-1, 1, lstm_out_forward.shape[2]))
        ).to(self.device)
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze(dim=1)
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)

        relu1 = F.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = F.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = self.fc2(drop1)

        return torch.sigmoid(output)


class NoteBasedChordPitchesModel(ChordPitchesModel):
    """
    The most simple chord pitches model, with layers:
        1. Linear layer (embedding)
        2. Bi-LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer

    Instead of a pitch-presence vector, this model outputs an estimate for each note of whether
    or not it is a chord tone. Its input is the same as the standard CPM input.
    """

    def __init__(
        self,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        embed_dim: int = 128,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        default_weight: float = 1.0,
        learning_rate: float = 0.001,
        input_mask: List[int] = None,
        window: int = 2,
    ):
        """
        Create a new SimpleChordPitchesModel.

        Parameters
        ----------
        input_pitch : PitchType
            What pitch type the model is expecting for notes.
        output_pitch : PitchType
            Unused. Only there to keep the signature the same as SimpleCPM.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the input chord types.
        use_inversions : bool
            Whether to use different inversions as different chords in the input.
        embed_dim : int
            The size of the initial embedding layer.
        lstm_layers : int
            The number of Bi-LSTM layers to use.
        lstm_hidden_dim : int
            The size of each LSTM layer's hidden vector.
        hidden_dim : int
            The size of the output vector of the first linear layer.
        dropout : float
            The dropout proportion of the first linear layer's output.
        default_weight : float
            The weight to use in the loss function for chords whose targets are the default
            pitches for their chord type and root.
        learning_rate : float
            The learning rate.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        window : int
            The window size for data creation. As input to each chord, there will be this
            many additional input vectors on either side.
        """
        super().__init__(
            input_pitch,
            input_pitch,
            reduction,
            use_inversions,
            default_weight,
            learning_rate,
            input_mask,
            window,
        )
        self.save_hyperparameters()

        # Input derived from pitch_type
        self.input_dim = get_chord_pitches_vector_length(input_pitch)

        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.input_dim, self.embed_dim)

        # LSTM hidden layer and depth
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 because bi-directional
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.dropout1 = nn.Dropout(self.dropout)

    def get_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Load the correct targets from a given batch dictionary.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch Dictionary containing any needed inputs.

        Returns
        -------
        targets : torch.Tensor
            The appropriate targets to be used for this model's outputs.
        """
        return batch["note_based_targets"].float()[:, : torch.max(batch["input_lengths"])]

    def get_weights(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the weights to use for the loss calculation given a list of whether
        each chord contains the default pitches or not.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch Dictionary, containing the entry "is_default", which is:
            One boolean per input chord, with True if that chord contains only the default
            pitches and False otherwise.

        Returns
        -------
        weights : torch.Tensor
            A (batch_size x num_output_pitches) tensor where each row is all 1's
            for non-default inputs, and all self.default_weight for default inputs.
        """
        is_default = batch["is_default"]

        weights = torch.ones(batch["note_based_targets"].shape, dtype=float)

        # Only use default_weight once the model has stagnated with default_weight 1
        if self.optimizers().param_groups[0]["lr"] != self.lr:
            weights[is_default, :] = self.default_weight

        # Set to 0 any weights where the target is -1
        weights[torch.where(batch["note_based_targets"] == -1)] = 0

        return weights.to(self.device)[:, : torch.max(batch["input_lengths"])]

    def calculate_and_log_validation_metrics(
        self,
        prefix: str,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        prefix : str
            The prefix to use for each metric when logging. Should be something like "default_"
            or "non-default_".
        targets : torch.Tensor
            The model's targets.
        outputs : torch.Tensor
            The model's output.
        weights : torch.Tensor
            The weights to use in loss calculation.
        """
        rounded_outputs = outputs.round()

        note_correct = (rounded_outputs == targets).float()
        chord_correct = (
            (torch.sum(note_correct, dim=1) == torch.sum(targets != -1, dim=1)).float().sum()
        )
        note_correct = note_correct.sum()

        total_notes = torch.sum(targets != -1)
        total_chords = len(targets)

        note_acc = 100 * note_correct / total_notes
        chord_acc = 100 * chord_correct / total_chords

        positive_target_mask = targets == 1
        positive_output_mask = rounded_outputs == 1

        tp = (positive_target_mask & positive_output_mask).sum().float()
        fp = (~positive_target_mask & positive_output_mask).sum().float()
        fn = (positive_target_mask & ~positive_output_mask).sum().float()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        loss = F.binary_cross_entropy(outputs, targets, weight=weights)

        self.log(f"{prefix}val_loss", loss)
        self.log(f"{prefix}val_note_acc", note_acc)
        self.log(f"{prefix}val_chord_acc", chord_acc)
        self.log(f"{prefix}val_precision", precision)
        self.log(f"{prefix}val_recall", recall)
        self.log(f"{prefix}val_f1", f1)

    def get_output(
        self, batch: Dict[str, torch.Tensor], return_notes: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load inputs from the batch data and return a non-rounded output
        Tensor derived from the model.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch Dictionary containing any needed inputs.
        return_notes : bool
            Also return the raw output notes binary vector for each input.

        Returns
        -------
        output : torch.Tensor
            An unrounded output tensor derived from the model, modeling the pitches
            at intervals on the range [-13, 13] for TPC, and [0, 11] for MIDI
            above the root. Values closer to 1 indicate presence of the pitch.
        notes_output : torch.Tensor
            A binary vector for each input, with 1 value per note, with each input
            being the probability that thee note is a chord tone.
        """
        raw_outputs = self.get_raw_output(batch)
        output = torch.zeros(
            (
                len(raw_outputs),
                (
                    NUM_PITCHES[PitchType.MIDI]
                    if self.OUTPUT_PITCH == PitchType.MIDI
                    else 2 * MAX_CHORD_PITCH_INTERVAL_TPC + 1
                ),
            ),
        )

        for i, (input_notes, length, targets, raw_output) in enumerate(
            zip(batch["inputs"], batch["input_lengths"], batch["note_based_targets"], raw_outputs)
        ):
            for note, target, out in zip(input_notes[:length], targets, raw_output):
                if target != -1:
                    # Ensure note is not padding
                    relative_pitch = get_relative_pitch_index(note, self.OUTPUT_PITCH)
                    if 0 <= relative_pitch < len(output[i]):
                        output[i, relative_pitch] = max(output[i, relative_pitch], out)

        if return_notes:
            return output, raw_outputs
        return output

    def init_hidden(self, batch_size: int) -> Tuple[Variable, Variable]:
        """
        Initialize the LSTM's hidden layer for a given batch size.

        Parameters
        ----------
        batch_size : int
            The batch size.
        """
        return (
            Variable(
                torch.zeros(
                    2 * self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device
                )
            ),
            Variable(
                torch.zeros(
                    2 * self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device
                )
            ),
        )

    def forward(self, inputs, lengths):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1).cpu()
        h_0, c_0 = self.init_hidden(batch_size)

        embed = F.relu(self.embed(inputs))

        packed_inputs = pack_padded_sequence(embed, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed_inputs, (h_0, c_0))
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        relu1 = F.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = F.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = torch.squeeze(self.fc2(drop1), dim=2)

        return torch.sigmoid(output)


def can_merge(
    pitches1: np.ndarray,
    pitches2: np.ndarray,
    default: np.ndarray,
    default_no_7th: np.ndarray,
    triad_type: ChordType,
    pitch_type: PitchType,
    no_aug_and_dim: bool,
) -> bool:
    """
    Check if two chord pitches arrays can be merged or not.

    Parameters
    ----------
    pitches1, pitches2 : np.ndarray
        The two chord pitches arrays, to check for mergability. The ordering of
        these two arrays does not matter.
    default : np.ndarray
        The default chord pitches vector for this chord.
    default_no_7th : np.ndarray
        The default chord pitches vector for this chord, without any 7th.
    triad_type : ChordType
        The triad type of this chord.
    pitch_type : PitchType
        The pitch type being used.
    no_aug_and_dim : bool
        True if the given chord vocabulary doesn't include aug or dim triads.
        False otherwise.

    Returns
    -------
    can_merge : bool
        True if the two arrays can be merged (by a logical or). False if there
        must be a chord split.
    """
    set1 = set(np.where(pitches1 == 1)[0])
    set2 = set(np.where(pitches2 == 1)[0])

    in_both = set1.intersection(set2)
    in_1_not_2 = set1 - in_both
    in_2_not_1 = set2 - in_both

    if len(in_1_not_2) == 0 or len(in_2_not_1) == 0:
        # One is a subset of the other. Merging is fine.
        return True

    M5_idx = CHORD_PITCHES[pitch_type][ChordType.MAJOR][2]
    if pitch_type == PitchType.TPC:
        M5_idx += len(default) // 2 - TPC_C
    d3_idx = CHORD_PITCHES[pitch_type][ChordType.DIMINISHED][1]
    if pitch_type == PitchType.TPC:
        d3_idx += len(default) // 2 - TPC_C
    dmM7_idxs = [
        CHORD_PITCHES[pitch_type][ChordType.DIM7][-1],
        CHORD_PITCHES[pitch_type][ChordType.MAJ_MIN7][-1],
        CHORD_PITCHES[pitch_type][ChordType.MAJ_MAJ7][-1],
    ]
    if pitch_type == PitchType.TPC:
        dmM7_idxs = [idx + len(default) // 2 - TPC_C for idx in dmM7_idxs]

    # Check every extra pitch for problems
    for (left_extra, left_all), (right_extra, right_all) in itertools.permutations(
        [(in_1_not_2, set1), (in_2_not_1, set2)]
    ):
        for extra_pitch in left_extra:
            if default[extra_pitch] == 1:
                # Extra pitch is default

                # Find possible (non-default) suspensions of this pitch
                possible_neighbors = get_neighbor_idxs(
                    extra_pitch,
                    set(np.where(default == 1)[0]).union(set(dmM7_idxs)),
                    pitch_type,
                    (
                        not no_aug_and_dim
                        and extra_pitch == M5_idx
                        and triad_type == ChordType.MAJOR
                    ),
                    extra_pitch == d3_idx and triad_type == ChordType.DIMINISHED,
                    no_aug_and_dim,
                )

                # Are the suspensions in right_extra?
                possible_neighbors = [
                    neighbor for neighbor in possible_neighbors if neighbor in right_extra
                ]

                # Does the neighbor suspend/replace extra_pitch (and not some other tone)?
                # If so, return False
                for neighbor_pitch in possible_neighbors:
                    # Pitches the neighbor_pitch might be replacing
                    possible_replacees = list(
                        set(
                            get_neighbor_idxs(
                                neighbor_pitch,
                                np.where(default == 0)[0],
                                pitch_type,
                                False,
                                False,
                                no_aug_and_dim,
                            )
                        )
                        - right_all
                    )

                    if len(possible_replacees) == 1 and possible_replacees[0] == extra_pitch:
                        # neighbor_pitch definitely replaces extra_pitch
                        return False

            else:
                # Extra pitch is non-default

                # Special handling to allow added 7ths on all triads
                if extra_pitch in dmM7_idxs and all(default == default_no_7th):

                    if any([pitch in dmM7_idxs for pitch in right_extra]):
                        # There's another 7th in right already, no merge
                        return False

                    # Otherwise, merging this pitch is fine.
                    continue

                # Find what (default) tones this one might replace
                possible_replacees = get_neighbor_idxs(
                    extra_pitch,
                    set(np.where(default == 0)[0]).union(left_all),
                    pitch_type,
                    False,
                    False,
                    no_aug_and_dim,
                )

                # If None, this pitch is fine (it is an added tone already)
                if len(possible_replacees) == 0:
                    continue

                # If not None, and all are already in right, return False
                if all([pitch in right_all for pitch in possible_replacees]):
                    return False

    # No problems found: can merge!
    return True


def merge_window_pitches(
    window_pitches: np.ndarray,
    default: np.ndarray,
    default_no_7th: np.ndarray,
    triad_type: ChordType,
    pitch_type: PitchType,
    no_aug_and_dim: bool = False,
) -> List[List]:
    """
    Merge the windowed chord pitches into a single chord_pitches array.

    Parameters
    ----------
    window_pitches : np.ndarray
        One binary chord-pitches array per window in this chord. These will be
        merged together.
    default : np.ndarray
        The default chord pitches array for this chord.
    default_no_7th : np.ndarray
        The default chord pitches array for this chord without 7ths.
    triad_type : ChordType
        The triad type of this chord.
    pitch_type : PitchType
        The pitch type used for this chord.
    no_aug_and_dim : bool
        True if the chord vocabulary uses no aug or dim triads. False otherwise.

    Returns
    -------
    chord_pitches : List[List[np.ndarray, int]]
        A List of the different chord pitches for this chord.
        Each element of the returned list is a duple containing:
            - The chord pitches.
            - The (exclusive) index to which this chord pitches array is valid.
        A chord which has no change in chord pitches during its duration
        will return a single-element list, whose index is the length of window_pitches.
    """

    M5_idx = CHORD_PITCHES[pitch_type][ChordType.MAJOR][2]
    if pitch_type == PitchType.TPC:
        M5_idx += window_pitches.shape[1] // 2 - TPC_C
    d3_idx = CHORD_PITCHES[pitch_type][ChordType.DIMINISHED][1]
    if pitch_type == PitchType.TPC:
        d3_idx += window_pitches.shape[1] // 2 - TPC_C
    dmM7_idxs = [
        CHORD_PITCHES[pitch_type][ChordType.DIM7][-1],
        CHORD_PITCHES[pitch_type][ChordType.MAJ_MIN7][-1],
        CHORD_PITCHES[pitch_type][ChordType.MAJ_MAJ7][-1],
    ]
    if pitch_type == PitchType.TPC:
        dmM7_idxs = [idx + window_pitches.shape[1] // 2 - TPC_C for idx in dmM7_idxs]

    # List of [chord_pitches, end_index] entries
    merged_chord_pitches = [[window_pitches[0], 1]]

    for pitches in window_pitches[1:]:
        if can_merge(
            pitches,
            merged_chord_pitches[-1][0],
            default,
            default_no_7th,
            triad_type,
            pitch_type,
            no_aug_and_dim=no_aug_and_dim,
        ):
            merged_chord_pitches[-1][0] = np.clip(
                np.add(merged_chord_pitches[-1][0], pitches), 0, 1
            )
            merged_chord_pitches[-1][1] += 1
        else:
            merged_chord_pitches.append([pitches, merged_chord_pitches[-1][1] + 1])

    return merged_chord_pitches


def decode_cpm_note_based_outputs(
    cpm_note_based_outputs: np.ndarray,
    all_notes: List[List[Note]],
    chords: List[Chord],
    defaults: np.ndarray,
    defaults_no_7ths: np.ndarray,
    triad_types: List[ChordType],
    cpm_chord_tone_threshold: float,
    cpm_non_chord_tone_add_threshold: float,
    cpm_non_chord_tone_replace_threshold: float,
    pitch_type: PitchType,
    no_aug_and_dim: bool = False,
) -> List[List[List]]:
    """
    Given the stacked outputs from the CPM (size num_chords x num_pitches), and the default
    output for each corresponding chord, return a List containing the binary chord pitches
    vector for each chord.

    Parameters
    ----------
    cpm_note_based_outputs : np.ndarray
        An ndarray of length num_chords, where each row is a list of the probability
        that the CPM has assigned to the corresponding note being a chord tone.
    all_notes : List[List[Note]]
        A List of the notes in each chord's window.
    chords : List[Chord]
        The chords in for the piece.
    defaults : List[np.ndarray]
        A binary array for each chord, encoding the default output, if there were no
        suspensions or alterations.
    defaults_no_7ths : List[np.ndarray]
        A binary array for each chord, encoding the default output, not including
        7ths.
    triad_types : List[ChordType]
        A list of the triad reduced type of each chord.
    cpm_chord_tone_threshold : float
        The threshold above which a default chord tone must reach in the CPM output
        in order to be considered present in a given chord.
    cpm_non_chord_tone_add_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to be an added tone in a given chord.
    cpm_non_chord_tone_replace_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to replace a chord tone in a given chord.
    pitch_type : PitchType
        The pitch type used in the outputs.
    no_aug_and_dim : bool
        True if the input vocabulary doesn't include augmented and diminished chords
        (and therefore we should allow them to be derived through changes).
        False otherwise.

    Returns
    -------
    chord_pitches : List[List[List[np.ndarray, int]]]
        A decoded CPM output after thresholding and other heuristic logic.
        For each chord:
            A List of "duples" (as length-2 lists) including:
                - A chord_pitches array (binary length num_rel_pitches).
                - An (exclusive) index to which window (within that chord)
                  the chord pitches array is valid.
    """

    def get_window_chord_pitches(
        cpm_note_based_output: np.ndarray,
        chord_root: int,
        notes: List[Note],
        default: np.ndarray,
        default_no_7th: np.ndarray,
        triad_type: ChordType,
        windows: List[Tuple[Fraction, Fraction]],
    ) -> List[List[float]]:
        """
        Get chord pitches lists for each chord window, given the cpm output.

        Parameters
        ----------
        cpm_note_based_output : np.ndarray
            The cpm's note-based output for this chord.
        chord_root : int
            The root pitch of the current chord.
        notes : List[Note]
            The notes within this chord window.
        default : np.ndarray
            The default chord pitches array for this chord.
        default_no_7th : np.ndarray
            The default chord pitches array with no 7ths, for this chord.
        triad_type : ChordType
            The triad type of this chord.
        windows : List[Tuple[Fraction, Fraction]]
            The windows for which we want to create chord pitches.

        Returns
        -------
        chord_pitches : List[List[float]]
            The output of the chord in each window.
        """
        window_chord_probs = []

        for start, end in windows:
            chord_probs = np.zeros_like(default, dtype=float)
            window_chord_probs.append(chord_probs)

            for note, note_output in zip(notes, cpm_note_based_output):
                if note is None or note.onset >= end or note.offset <= start:
                    continue

                try:
                    relative_pitch = absolute_to_relative(
                        note.pitch_class, chord_root, note.pitch_type, False, pad=True
                    )
                except ValueError:
                    # Note is outside of range
                    continue

                if note.pitch_type == PitchType.TPC:
                    relative_pitch -= int(
                        (
                            NUM_RELATIVE_PITCHES[PitchType.TPC][True]
                            - 2 * MAX_CHORD_PITCH_INTERVAL_TPC
                            - 1
                        )
                        / 2
                    )

                if 0 <= relative_pitch < len(chord_probs):
                    chord_probs[relative_pitch] = max(chord_probs[relative_pitch], note_output)

        num_windows = len(window_chord_probs)
        chord_pitches = decode_cpm_outputs(
            np.vstack(window_chord_probs),
            np.full((num_windows, len(default)), default),
            np.full((num_windows, len(default)), default_no_7th),
            np.full(num_windows, triad_type),
            cpm_chord_tone_threshold,
            cpm_non_chord_tone_add_threshold,
            cpm_non_chord_tone_replace_threshold,
            pitch_type,
            add_pitches=False,
            no_aug_and_dim=no_aug_and_dim,
        )

        return chord_pitches

    chord_pitches = [None] * len(cpm_note_based_outputs)
    for i, (cpm_note_based_output, notes, chord, default, default_no_7th, triad_type,) in enumerate(
        zip(
            cpm_note_based_outputs,
            all_notes,
            chords,
            defaults,
            defaults_no_7ths,
            triad_types,
        )
    ):
        window_pitches = get_window_chord_pitches(
            cpm_note_based_output,
            chord.root,
            notes,
            default,
            default_no_7th,
            triad_type,
            get_windows(chord.onset, chord.offset, notes),
        )

        chord_pitches[i] = merge_window_pitches(
            window_pitches,
            default,
            default_no_7th,
            triad_type,
            pitch_type,
            no_aug_and_dim=no_aug_and_dim,
        )
        num_windows = len(chord_pitches[i])
        # Add back extra non-present chord tones, etc.
        for window_idx, full_window_pitches in enumerate(
            decode_cpm_outputs(
                np.vstack([chord_pitches[i][j][0] for j in range(num_windows)]),
                np.tile(default, (num_windows, 1)),
                np.tile(default_no_7th, (num_windows, 1)),
                [triad_type] * num_windows,
                cpm_chord_tone_threshold,
                cpm_non_chord_tone_add_threshold,
                cpm_non_chord_tone_replace_threshold,
                pitch_type,
                no_aug_and_dim=no_aug_and_dim,
            )
        ):
            chord_pitches[i][window_idx][0] = full_window_pitches

    return chord_pitches


def get_neighbor_idxs(
    idx: int,
    do_not_return: List[int],
    pitch_type: PitchType,
    is_M5: bool,
    is_d3: bool,
    no_aug_and_dim: bool,
    minimum: int = 0,
    maximum: int = 2 * MAX_CHORD_PITCH_INTERVAL_TPC,
) -> np.ndarray:
    """
    Get the indexes for notes that could be a neighbor of the given note.

    Parameters
    ----------
    idx : int
        The index of the note whose neighbors we are looking for.
    do_not_return : List[int]
        A list of indexes not to include in the returned neighbor_idxs list.
        This can include, for example, other already present chord tones.
    pitch_type : PitchType
        The pitch type being used for indexing.
    is_M5 : bool
        The chord from which the note comes is a major chord and the idx represents
        its 5th. In that case, a flat version of the given idx is returned as a
        potential neighbor, since (root, M3, d5) is not another chord type.
    is_d3 : bool
        The chord from which the note comes is a diminished chord and the idx
        represents its 3rd. In that case, a flat version of the given idx is
        returned as a potential neighbor, since (root, dd3, m5) is not another
        chord type.
    no_aug_and_dim : bool
        Whether the input vocabulary includes diminished and augmented chords.
        If this valud is True (the input does not contain those chords),
        a flat and sharp version of the input pitch will be included in the
        returned neighbors. Otherwise, a sharp version will never be returned,
        and a flat version will only be returned if is_M5 or is_d3 is True.
    minimum : int
        The smallest index of a neighbor note to return.
    maximum : int
        The largest index of a neighbor note to return. Note that the default
        is the TPC default. If pitch_type is MIDI, 11 is used.

    Returns
    -------
    neighbor_idxs : np.ndarray
        The indexes of possible neighbor notes to the given note. For PitchType.MIDI,
        this is 3 semitones in either direction (allowing for an augmented 2nd).
        For PitchType.TPC, this is all altered versions of the given idx, and all
        altered versions of a 2nd up and down.
    """
    if pitch_type == PitchType.MIDI:
        neighbors = list(
            np.arange(max(minimum, idx - 3), min(NUM_PITCHES[PitchType.MIDI], idx + 4))
        )

    else:
        # Include a flat version of the given note if is_M5 or is_d3.
        neighbors = [idx - 7] if (is_M5 or is_d3) and (minimum <= idx - 7 <= maximum) else []
        if no_aug_and_dim:
            if minimum <= idx - 7 <= maximum:
                neighbors.append(idx - 7)
            if minimum <= idx + 7 <= maximum:
                neighbors.append(idx + 7)

        # Include altered versions of a 2nd up
        neighbor_up = idx + 2
        neighbors.extend(
            list(range(minimum + ((neighbor_up % 7) - (minimum % 7)) % 7, maximum + 1, 7))
        )

        # Include altered versions of a 2nd down
        neighbor_down = idx - 2
        neighbors.extend(
            list(range(minimum + ((neighbor_down % 7) - (minimum % 7)) % 7, maximum + 1, 7))
        )

    neighbors_set = set(neighbors) - set(do_not_return) - set([idx])

    return np.array(sorted(neighbors_set), dtype=int)


def decode_cpm_outputs(
    cpm_outputs: np.ndarray,
    defaults: np.ndarray,
    defaults_no_7ths: np.ndarray,
    triad_types: List[ChordType],
    cpm_chord_tone_threshold: float,
    cpm_non_chord_tone_add_threshold: float,
    cpm_non_chord_tone_replace_threshold: float,
    pitch_type: PitchType,
    add_pitches: bool = True,
    no_aug_and_dim: bool = False,
) -> np.ndarray:
    """
    Given the stacked outputs from the CPM (size num_chords x num_pitches), and the default
    output for each corresponding chord, return a List containing the binary chord pitches
    vector for each chord.

    Parameters
    ----------
    cpm_outputs : np.ndarray
        An ndarray of size num_chords * num_pitches, where each value [i, j] in (0-1)
        should roughly correspond to the probability the CPM assigns of pitch j being
        present in the ith chord.
    defaults : List[np.ndarray]
        A binary array for each chord, encoding the default output, if there were no
        suspensions or alterations.
    defaults_no_7ths : List[np.ndarray]
        A binary array for each chord, encoding the default output, not including
        7ths.
    triad_types : List[ChordType]
        A list of the triad reduced type of each chord.
    cpm_chord_tone_threshold : float
        The threshold above which a default chord tone must reach in the CPM output
        in order to be considered present in a given chord.
    cpm_non_chord_tone_add_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to be an added tone in a given chord.
    cpm_non_chord_tone_replace_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to replace a chord tone in a given chord.
    pitch_type : PitchType
        The pitch type used in the outputs.
    add_pitches : bool
        True to add additional default tones which are not in any pitch.
        False otherwise.
    no_aug_and_dim : bool
        True if the input vocabulary doesn't include augmented and diminished chords
        (and therefore we should allow them to be derived through changes).
        False otherwise.

    Returns
    -------
    chord_pitches : List[np.ndarray]
        A decoded CPM output. After thresholding and other heuristic logic, the pitches
        present in each chord, in a num_chords x num_rel_pitches array.
    """

    def get_best_pitches(
        cpm_output: np.ndarray,
        can_remove: np.ndarray,
        can_replace: np.ndarray,
    ) -> List[int]:
        """
        Get a list of the best pitches to include in the chord pitches for a given chord.

        Parameters
        ----------
        cpm_output : np.ndarray
            The output of the cpm, used to disambiguate selections of pitches and pick the
            best.
        can_remove : np.ndarray
            A list of the pitches that could be removed from the chord.
        can_replace : np.ndarray
            For each pitch in can_remove, a list of pitches that could replace that pitch.

        Returns
        -------
        to_add : List[int]
            A List of pitches to include in the chord. The returned list's length will be
            equal to the length of can_remove, and include one pitch per pitch in can_remove
            (though not necessarily in order). This might be the given can_remove pitch, or a
            pitch from the corresponding can_replace list, depending on the found optimal
            configuration.
        """
        # Always prefer replacing as many notes as possible
        best = (0, 1, list(can_remove))  # Num_replaced, prob, to_add
        index_ranges = [list(range(len(replace) + 1)) for replace in can_replace]

        for indexes in itertools.product(*index_ranges):
            prob = 1
            to_add = list(can_remove)
            added = set()
            num_added = 0

            for i, index in enumerate(indexes):
                if index < len(can_replace[i]):
                    replacement_idx = can_replace[i][index]

                    prob *= (1 - cpm_output[can_remove[i]]) * cpm_output[replacement_idx]

                    # Add idx to tracking objects
                    to_add[i] = replacement_idx
                    added.add(replacement_idx)
                    num_added += 1

            if len(added) == num_added:
                # Valid configuration: all added pitches were unique
                best = max(best, (num_added, prob, to_add))

        return best[2]

    chord_pitches = np.zeros_like(cpm_outputs, dtype=int)
    can_remove_tones = np.logical_and(
        defaults_no_7ths == 1, cpm_outputs <= cpm_chord_tone_threshold
    )
    can_remove_tones[:, 13] = False  # Disallow root removal
    cannot_remove_tones = np.logical_and(defaults == 1, ~can_remove_tones)
    can_add_tones = np.logical_and(defaults == 0, cpm_outputs >= cpm_non_chord_tone_add_threshold)
    can_replace_tones = np.logical_and(
        defaults == 0, cpm_outputs >= cpm_non_chord_tone_replace_threshold
    )

    M5_idx = CHORD_PITCHES[pitch_type][ChordType.MAJOR][2]
    if pitch_type == PitchType.TPC:
        M5_idx += chord_pitches.shape[1] // 2 - TPC_C
    d3_idx = CHORD_PITCHES[pitch_type][ChordType.DIMINISHED][1]
    if pitch_type == PitchType.TPC:
        d3_idx += chord_pitches.shape[1] // 2 - TPC_C

    for i, (can_remove, cannot_remove, can_add, can_replace, default, triad_type) in enumerate(
        zip(
            can_remove_tones,
            cannot_remove_tones,
            can_add_tones,
            can_replace_tones,
            defaults,
            triad_types,
        )
    ):
        # Keep non-removable chord tones
        chord_pitches[i, np.where(cannot_remove)[0]] = 1

        # Add non-chord tones
        chord_pitches[i, np.where(can_add)[0]] = 1

        # Replace chord tones with neighbors
        can_remove_idxs = np.where(can_remove)[0]
        can_replace_idxs = np.where(can_replace)[0]
        default_idxs = np.where(default)[0]

        if len(can_remove_idxs) == 0:
            continue

        neighbor_idxs = [
            get_neighbor_idxs(
                idx,
                default_idxs,
                pitch_type,
                not no_aug_and_dim and triad_type == ChordType.MAJOR and idx == M5_idx,
                triad_type == ChordType.DIMINISHED and idx == d3_idx,
                no_aug_and_dim,
            )
            for idx in can_remove_idxs
        ]

        can_replace_neighbors = [
            neighbors[np.isin(neighbors, can_replace_idxs)] for neighbors in neighbor_idxs
        ]
        # Mask tones for which no neighbors can actually replace them
        valid_mask = np.array([len(n) > 0 for n in can_replace_neighbors])

        # Add back tones that actually can't be removed
        if add_pitches:
            chord_pitches[i, can_remove_idxs[~valid_mask]] = 1

        # Apply mask to index lists
        can_remove_idxs = can_remove_idxs[valid_mask]
        can_replace_neighbors = [n for n, mask in zip(can_replace_neighbors, valid_mask) if mask]

        if len(can_replace_neighbors) == 1:
            # Only one chord tone can be removed -- Faster than using get_best_pitches
            can_replace_neighbors = can_replace_neighbors[0]

            if len(can_replace_neighbors) > 1:
                # Replace the note with the most likely neighbor
                to_add = can_replace_neighbors[np.argmax(cpm_outputs[i, can_replace_neighbors])]
            else:
                to_add = can_replace_neighbors[0]

            chord_pitches[i, to_add] = 1

        elif len(can_replace_neighbors) >= 2:
            # More than one chord tone might be altered
            # In the case that multiple neighbors are over threshold, we have to match them
            # In the case that only one neighbor is over threshold, we assign it to one tone
            for to_add in get_best_pitches(cpm_outputs[i], can_remove_idxs, can_replace_neighbors):
                if add_pitches or to_add not in can_remove_idxs:
                    chord_pitches[i, to_add] = 1

    return chord_pitches


def get_rule_based_cpm_outputs(
    all_notes: List[List[Note]],
    chords: List[Chord],
    defaults: np.ndarray,
    triad_types: List[ChordType],
    pitch_type: PitchType,
    no_7ths: bool = False,
    no_aug_and_dim: bool = False,
    suspensions: bool = False,
) -> List[List[List]]:
    """
    Get chord_pitches arrays from the given lists of notes and chords from the rule-based
    system. The rules are:
      - If no_7ths:
        - Any window containing one (and only one) 7th is considered to contain that 7th.
      - If no_aug_or_dim:
        - Any Maj window containing a #5 with no 5 will include that 5.
        - Any Min window containing a b5 with no 5 will include that 5.
      - Special handling for aug 6 chords:
        - Allow for bb3 if [(Dim triad) or (no_aug_or_dim and Min triad)] and no b3 in window.
        - Allow for b5 if Maj triad (and no 5 or #5 in window).
      - Otherwise, all tones are default.

    Additional rules if suspensions is True:
      - If no 5th:
        - If a 6th is present, add it.
      - If no 3rd:
        - If a 4th is present, add it.

    Parameters
    ----------
    all_notes : List[List[Note]]
        A List of the notes in each chord's window.
    chords : List[Chord]
        The chords in for the piece.
    defaults : List[np.ndarray]
        A binary array for each chord, encoding the default output, if there were no
        suspensions or alterations.
    triad_types : List[ChordType]
        A list of the triad reduced type of each chord.
    pitch_type : PitchType
        The pitch type used in the outputs.
    no_7ths : bool
        True if the input vocabulary doesn't include 7th chords. False otherwise.
    no_aug_and_dim : bool
        True if the input vocabulary doesn't include augmented and diminished chords
        (and therefore we should allow them to be derived through changes).
        False otherwise.
    suspensions : bool
        If True, add the additional suspension rules described above.

    Returns
    -------
    chord_pitches : List[List[List[np.ndarray, int]]]
        The rule-based output after its heuristic logic.
        For each chord:
            A List of "duples" (as length-2 lists) including:
                - A chord_pitches array (binary length num_rel_pitches).
                - An (exclusive) index to which window (within that chord)
                  the chord pitches array is valid.
    """
    if pitch_type == PitchType.MIDI:
        raise ValueError("The Rule-Based CPM is only well-defined with TPCs (not MIDI pitch).")

    if no_aug_and_dim and not no_7ths:
        logging.warning("no_aug_and_dim is True but no_7ths is False. Setting no_7ths to True.")
        no_7ths = True

    dmM7_idxs = [
        CHORD_PITCHES[pitch_type][ChordType.DIM7][-1],
        CHORD_PITCHES[pitch_type][ChordType.MAJ_MIN7][-1],
        CHORD_PITCHES[pitch_type][ChordType.MAJ_MAJ7][-1],
    ]
    dmM7_idxs = [idx + len(defaults[0]) // 2 - TPC_C for idx in dmM7_idxs]
    p5_idx = CHORD_PITCHES[pitch_type][ChordType.MAJOR][-1] + len(defaults[0]) // 2 - TPC_C
    m3_idx = CHORD_PITCHES[pitch_type][ChordType.MINOR][1] + len(defaults[0]) // 2 - TPC_C

    chord_pitches = [None] * len(chords)

    for i, (default, notes, chord, triad_type) in enumerate(
        zip(defaults, all_notes, chords, triad_types)
    ):
        windows = get_windows(chord.onset, chord.offset, notes)

        # Generate window pitches for each window
        window_pitches = np.zeros((len(windows), len(default)))
        for window_idx, (start, end) in enumerate(windows):

            # Get the set of present pitches in this window
            pitches = set()
            for note in notes:
                if note is None or note.onset >= end or note.offset <= start:
                    continue

                try:
                    relative_pitch = absolute_to_relative(
                        note.pitch_class, chord.root, pitch_type, False, pad=True
                    )
                except ValueError:
                    # Note is outside of range
                    continue

                relative_pitch -= int(
                    (
                        NUM_RELATIVE_PITCHES[PitchType.TPC][True]
                        - 2 * MAX_CHORD_PITCH_INTERVAL_TPC
                        - 1
                    )
                    / 2
                )

                if 0 <= relative_pitch < len(default):
                    pitches.add(relative_pitch)

            # Add all present default pitches
            for pitch in pitches:
                if default[pitch] == 1:
                    window_pitches[window_idx][pitch] = 1

            if no_7ths:
                # If there is only a single 7th, add it
                all_7ths = pitches.intersection(set(dmM7_idxs))
                if len(all_7ths) == 1:
                    window_pitches[window_idx][list(all_7ths)[0]] = 1

            if no_aug_and_dim:
                # If Major, check for #5 without 5
                if triad_type == ChordType.MAJOR:
                    if p5_idx not in pitches and (p5_idx + 7) in pitches:
                        window_pitches[window_idx][p5_idx + 7] = 1

                # If Minor, check for b5 without 5
                elif triad_type == ChordType.MINOR:
                    if p5_idx not in pitches and (p5_idx - 7) in pitches:
                        window_pitches[window_idx][p5_idx - 7] = 1

            # Check for aug 6ths
            if triad_type == ChordType.MAJOR:
                # b5 in major triad if no 5 or #5 yet
                if p5_idx not in pitches and (p5_idx + 7) not in pitches:
                    if (p5_idx - 7) in pitches:
                        window_pitches[window_idx][p5_idx - 7] = 1

            if triad_type == ChordType.DIMINISHED or (
                triad_type == ChordType.MINOR and no_aug_and_dim
            ):
                # bb3 in dim triad if no m3 yet
                if m3_idx not in pitches and (m3_idx - 7) in pitches:
                    window_pitches[window_idx][m3_idx - 7] = 1

            # Check for suspended 5th or 3rd
            if suspensions:
                # Suspended 5th:
                if (
                    p5_idx not in pitches
                    and (p5_idx + 7) not in pitches
                    and (p5_idx - 7) not in pitches
                ):
                    mM6_idxs = set([p5_idx + 2, p5_idx - 5])
                    all_6ths = pitches.intersection(mM6_idxs)
                    if len(all_6ths) == 1:
                        window_pitches[window_idx][list(all_6ths)[0]] = 1

                # Suspended 3rd:
                if (
                    m3_idx not in pitches
                    and (m3_idx + 7) not in pitches
                    and (m3_idx - 7) not in pitches
                ):
                    pda4_idxs = set([p5_idx - 2, p5_idx + 5, p5_idx - 9])
                    all_4ths = pitches.intersection(pda4_idxs)
                    if len(all_4ths) == 1:
                        window_pitches[window_idx][list(all_4ths)[0]] = 1

        chord_pitches[i] = merge_window_pitches(
            window_pitches,
            default,
            default,  # Ok because if there are 7ths, we won't get here
            triad_type,
            pitch_type,
            no_aug_and_dim=no_aug_and_dim,
        )
        num_windows = len(chord_pitches[i])
        # Add back extra non-present chord tones, etc.
        for window_idx, full_window_pitches in enumerate(
            decode_cpm_outputs(
                np.vstack([chord_pitches[i][j][0] for j in range(num_windows)]),
                np.tile(default, (num_windows, 1)),
                np.tile(default, (num_windows, 1)),
                [triad_type] * num_windows,
                0.5,
                0.5,
                0.5,
                pitch_type,
                no_aug_and_dim=no_aug_and_dim,
            )
        ):
            chord_pitches[i][window_idx][0] = full_window_pitches

    return chord_pitches


def get_chord_pitches_in_piece(
    cpm: ChordPitchesModel,
    piece: Piece,
    cpm_chord_tone_threshold: float,
    cpm_non_chord_tone_add_threshold: float,
    cpm_non_chord_tone_replace_threshold: float,
    merge_changes: bool = False,
    merge_reduction: Dict[ChordType, ChordType] = None,
    rule_based: bool = False,
    suspensions: bool = False,
) -> Piece:
    """
    Run a CPM on an input Piece, and return a new Piece containing the outputs, where
    chords have been reduced, processed, split, merged, and assigned their resulting
    chord_pitches arrays.

    Parameters
    ----------
    cpm : ChordPitchesModel
        The CPM to evaluate.
    piece : Piece
        The Piece to run the CPM on.
    cpm_chord_tone_threshold : float
        The threshold above which a default chord tone must reach in the CPM output
        in order to be considered present in a given chord.
    cpm_non_chord_tone_add_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to be an added tone in a given chord.
    cpm_non_chord_tone_replace_threshold : float
        The threshold above which a default non-chord tone must reach in the CPM output
        in order to replace a chord tone in a given chord.
    merge_changes : bool
        Merge chords which differ only by their chord tone changes into single chords
        as input. The targets will remain unchanged, so the CPM will ideally split
        such chords in its post-processing step.
    merge_reduction : Dict[ChordType, ChordType]
        Merge chords which no longer differ after this chord type reduction together
        as input. The targets will remain unchanged, so the CPM will ideally split
        such chords in its post-processing step.
    rule_based : bool
        Generate the rule-based output rather than running any CPM.
    suspensions : bool
        Look for 6 and 4 suspensions in the rule-based system.

    Returns
    -------
    processed_piece : Piece
        A copy of the input Piece, but with reductions applied, as well as the CPM outputs
        applied and assigned (and chords merged and split as applicable).
    """
    reduced_piece: ScorePiece = piece
    if merge_changes or merge_reduction is not None:
        reduced_piece = get_score_piece_from_dict(
            reduced_piece.measures_df, reduced_piece.to_dict(), name=reduced_piece.name
        )
        if merge_reduction is not None:
            for chord in reduced_piece.get_chords():
                chord.chord_type = merge_reduction[chord.chord_type]
                if chord.inversion >= get_chord_inversion_count(chord.chord_type):
                    chord.inversion = 0
        reduced_piece.merge_chords(merge_changes)

    dataset = ChordPitchesDataset([reduced_piece], **cpm.get_dataset_kwargs())
    dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

    if not rule_based:
        outputs = []
        note_outputs = []
        for batch in dl:
            if isinstance(cpm, NoteBasedChordPitchesModel):
                output, note_output = cpm.get_output(batch, return_notes=True)
                outputs.extend(output.numpy())
                note_outputs.extend(note_output.numpy())
            else:
                outputs.extend(cpm.get_output(batch).numpy())

        # Stack all outputs
        lengths = np.array([len(n) for n in note_outputs])
        if np.all(lengths == lengths[0]):
            note_outputs_np = np.vstack(note_outputs)
        else:
            note_outputs_np = np.zeros((len(note_outputs), np.max(lengths)))
            for i, note_output in enumerate(note_outputs):
                note_outputs_np[i][: len(note_output)] = note_output

    if rule_based:
        chord_pitches = get_rule_based_cpm_outputs(
            reduced_piece.get_chord_note_inputs(window=dataset.window[0], notes_only=True),
            reduced_piece.get_chords(),
            np.vstack(
                [
                    chord.get_chord_pitches_target_vector(default=True)
                    for chord in reduced_piece.get_chords()
                ]
            ),
            [TRIAD_REDUCTION[chord.chord_type] for chord in reduced_piece.get_chords()],
            cpm.INPUT_PITCH,
            no_7ths=merge_reduction is not None,
            no_aug_and_dim=merge_reduction == MAJOR_MINOR_REDUCTION,
            suspensions=suspensions,
        )

    elif isinstance(cpm, NoteBasedChordPitchesModel):
        chord_pitches = decode_cpm_note_based_outputs(
            note_outputs_np,
            reduced_piece.get_chord_note_inputs(window=dataset.window[0], notes_only=True),
            reduced_piece.get_chords(),
            np.vstack(
                [
                    chord.get_chord_pitches_target_vector(default=True)
                    for chord in reduced_piece.get_chords()
                ]
            ),
            np.vstack(
                [
                    chord.get_chord_pitches_target_vector(reduction=TRIAD_REDUCTION, default=True)
                    for chord in reduced_piece.get_chords()
                ]
            ),
            [TRIAD_REDUCTION[chord.chord_type] for chord in reduced_piece.get_chords()],
            cpm_chord_tone_threshold,
            cpm_non_chord_tone_add_threshold,
            cpm_non_chord_tone_replace_threshold,
            cpm.INPUT_PITCH,
            no_aug_and_dim=merge_reduction == MAJOR_MINOR_REDUCTION,
        )

    else:
        chord_pitches = decode_cpm_outputs(
            np.vstack(outputs),
            np.vstack(
                [chord.get_chord_pitches_target_vector() for chord in reduced_piece.get_chords()]
            ),
            np.vstack(
                [
                    chord.get_chord_pitches_target_vector(reduction=TRIAD_REDUCTION)
                    for chord in reduced_piece.get_chords()
                ]
            ),
            [TRIAD_REDUCTION[chord.chord_type] for chord in reduced_piece.get_chords()],
            cpm_chord_tone_threshold,
            cpm_non_chord_tone_add_threshold,
            cpm_non_chord_tone_replace_threshold,
            cpm.INPUT_PITCH,
            no_aug_and_dim=merge_reduction == MAJOR_MINOR_REDUCTION,
        )

    processed_piece = get_score_piece_from_dict(
        reduced_piece.measures_df, reduced_piece.to_dict(), name=reduced_piece.name
    )
    for pitches, chord in zip(chord_pitches, processed_piece.get_chords()):
        if isinstance(pitches, list):
            # Note-based (window-based) output. Need to do some special handling
            abs_pitches = []

            for window_pitches, window_index in pitches:
                # Convert binary pitches array into root-relative indices
                pitch_indices = np.where(window_pitches)[0]
                if cpm.INPUT_PITCH == PitchType.TPC:
                    pitch_indices -= MAX_CHORD_PITCH_INTERVAL_TPC

                # Convert root-relative indices into absolute pitches
                abs_pitches.append(
                    [set([chord.root + pitch for pitch in pitch_indices]), window_index]
                )

            # Remove list if only one window
            if len(abs_pitches) == 1:
                abs_pitches = abs_pitches[0][0]

        else:
            # Convert binary pitches array into root-relative indices
            pitch_indices = np.where(pitches)[0]
            if cpm.INPUT_PITCH == PitchType.TPC:
                pitch_indices -= MAX_CHORD_PITCH_INTERVAL_TPC

            # Convert root-relative indices into absolute pitches
            abs_pitches = set([chord.root + pitch for pitch in pitch_indices])

        chord.chord_pitches = abs_pitches

    processed_piece.split_chord_pitches_chords()

    return processed_piece
