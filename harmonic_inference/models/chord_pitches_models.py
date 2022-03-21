"""Models that generate probability distributions over the pitches present in a given chord."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.chord import (
    get_chord_pitches_target_vector_length,
    get_chord_pitches_vector_length,
)
from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.datasets import ChordPitchesDataset


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
        """
        super().__init__()
        self.INPUT_PITCH = input_pitch
        self.OUTPUT_PITCH = output_pitch

        self.reduction = reduction
        self.use_inversions = use_inversions

        self.default_weight = default_weight

        self.lr = learning_rate

        self.input_mask = input_mask

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

    def get_output(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Load inputs from the batch data and return a non-rounded output
        Tensor derived from the model.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch Dictionary containing any needed inputs.

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
        """
        super().__init__(
            input_pitch,
            output_pitch,
            reduction,
            use_inversions,
            default_weight,
            learning_rate,
            input_mask,
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
        """
        super().__init__(
            input_pitch,
            input_pitch,
            reduction,
            use_inversions,
            default_weight,
            learning_rate,
            input_mask,
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
