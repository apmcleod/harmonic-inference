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

    def get_weights(self, is_default: torch.Tensor[bool]) -> torch.Tensor[float]:
        """
        Get the weights to use for the loss calculation given a list of whether
        each chord contains the default pitches or not.

        Parameters
        ----------
        is_default : torch.Tensor[bool]
            One boolean per input chord, with True if that chord contains only the default
            pitches and False otherwise.

        Returns
        -------
        weights : torch.Tensor[float]
            A (batch_size x num_output_pitches) tensor where each row is all 1's
            for non-default inputs, and all self.default_weight for default inputs.
        """
        weights = torch.ones((len(is_default), self.output_dim), dtype=float)
        weights[~is_default, :] = self.default_weight
        return weights

    def get_output(self, batch):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]

        outputs = self(inputs, input_lengths)

        return outputs

    def training_step(self, batch, batch_idx):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]
        targets = batch["targets"].float()
        weights = self.get_weights(batch["is_default"])

        outputs = self(inputs, input_lengths)
        loss = F.binary_cross_entropy(outputs, targets, weight=weights)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]
        targets = batch["targets"].float()
        weights = self.get_weights(batch["is_default"])

        outputs = self(inputs, input_lengths)
        rounded_outputs = outputs.round()

        if len(targets) > 0:
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

            self.log("val_loss", loss)
            self.log("val_pitch_acc", pitch_acc)
            self.log("val_chord_acc", chord_acc)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1", f1)

    def evaluate(self, dataset: ChordPitchesDataset):
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
            inputs = batch["inputs"].float()
            input_lengths = batch["input_lengths"]
            targets = batch["targets"].float()
            weights = self.get_weights(batch["is_default"])

            outputs = self(inputs, input_lengths)
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
            The reduction used for the output chord types.
        use_inversions : bool
            Whether to use different inversions as different chords in the output. Used to
            derive the output length.
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
