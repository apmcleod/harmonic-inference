"""Models that generate probability distributions over the pitches present in a given chord."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

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
from harmonic_inference.data.data_types import ChordType, PieceType, PitchType
from harmonic_inference.data.datasets import ChordPitchesDataset


class ChordPitchesModel(pl.LightningModule, ABC):
    """
    The base type for all Chord Pitches Models, which take as input sets of inputs and chords from
    Pieces, and output pitch presence probabilities for them.
    """

    def __init__(
        self,
        input_type: PieceType,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType],
        use_inversions: bool,
        learning_rate: float,
        transposition_range: Union[List[int], Tuple[int, int]],
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

    def get_output(self, batch):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]

        outputs = self(inputs, input_lengths)

        # TODO: Not softmax here
        return F.softmax(outputs, dim=-1)

    def training_step(self, batch, batch_idx):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]
        targets = batch["targets"].long()

        outputs = self(inputs, input_lengths)
        # TODO: Not cross-ent (I think?)
        loss = F.cross_entropy(outputs, targets, ignore_index=-1)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]
        targets = batch["targets"].long()

        outputs = self(inputs, input_lengths)

        # TODO: Check mask and accuracy measurement
        # TODO: Add precision/recall?
        mask = targets != -1
        outputs = outputs[mask]
        targets = targets[mask]

        if len(targets) > 0:
            acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)
            loss = F.cross_entropy(outputs, targets, ignore_index=-1)

            self.log("val_loss", loss)
            self.log("val_acc", acc)

    def evaluate(self, dataset: ChordPitchesDataset):
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        total = 0
        total_loss = 0
        total_acc = 0

        for batch in tqdm(dl, desc="Evaluating CPM"):
            inputs = batch["inputs"].float()
            input_lengths = batch["input_lengths"]
            targets = batch["targets"].long()

            # TODO: Check len
            batch_count = len(targets)
            outputs = self(inputs, input_lengths)
            loss = F.cross_entropy(outputs, targets)  # TODO: Check loss
            # TODO: Check acc
            acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)
            # TODO: add prec/recall?

            total += batch_count
            total_loss += loss * batch_count
            total_acc += acc * batch_count

        return {
            "acc": (total_acc / total).item(),
            "loss": (total_loss / total).item(),
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


class SimpleChordClassifier(ChordPitchesModel):
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

        # TODO: Add sigmoid?
        return output
