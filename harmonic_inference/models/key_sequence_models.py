"""Models that generate probability distributions over the next key in a sequence."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_lightning as pl
from harmonic_inference.data.chord import get_chord_vector_length
from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.datasets import KeySequenceDataset
from harmonic_inference.data.key import get_key_change_vector_length


class KeySequenceModel(pl.LightningModule, ABC):
    """
    The base class for all Key Sequence Models, which model the sequence of keys of a Piece.
    """

    def __init__(
        self,
        input_chord_pitch_type: PitchType,
        input_key_pitch_type: PitchType,
        output_pitch_type: PitchType,
        learning_rate: float,
    ):
        """
        Create a new base KeySequenceModel with the given output and input data types.

        Parameters
        ----------
        input_chord_pitch_type : PitchType
            The pitch representation used in the input chord data.
        input_key_pitch_type : PitchType
            The pitch representation used in the input key data.
        output_pitch_type : PitchType
            The pitch representation used for the output data.
        learning_rate : float
            The learning rate.
        """
        super().__init__()
        self.INPUT_CHORD_PITCH_TYPE = input_chord_pitch_type
        self.INPUT_KEY_PITCH_TYPE = input_key_pitch_type
        self.OUTPUT_PITCH_TYPE = output_pitch_type
        self.lr = learning_rate

    def get_data_from_batch(self, batch):
        inputs = batch["inputs"].float()
        targets = batch["targets"].long()
        input_lengths = batch["input_lengths"]

        longest = max(input_lengths)
        inputs = inputs[:, :longest]

        return inputs, input_lengths, targets

    def get_output(self, batch):
        inputs, input_lengths, _ = self.get_data_from_batch(batch)

        hidden = (
            torch.transpose(batch["hidden_states"][0], 0, 1),
            torch.transpose(batch["hidden_states"][1], 0, 1),
        )

        return self(inputs, input_lengths, hidden=hidden)

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets = self.get_data_from_batch(batch)

        outputs, _ = self(inputs, input_lengths)

        loss = F.nll_loss(outputs, targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets = self.get_data_from_batch(batch)

        outputs, _ = self(inputs, input_lengths)

        loss = F.nll_loss(outputs, targets)
        acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def evaluate(self, dataset: KeySequenceDataset):
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        total = 0
        total_acc = 0
        total_loss = 0

        for batch in tqdm(dl, desc="Evaluating KSM"):
            inputs, input_lengths, targets = self.get_data_from_batch(batch)

            outputs, _ = self(inputs, input_lengths)

            batch_count = len(batch)
            loss = F.nll_loss(outputs, targets)
            acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)

            total += batch_count
            total_acc += acc * batch_count
            total_loss += loss * batch_count

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


class SimpleKeySequenceModel(KeySequenceModel):
    """
    The simplest key sequence model, with layers:
    1. Linear embedding layer
    2. Bi-LSTM
    3. Linear layer
    4. Dropout
    5. Linear layer
    """

    def __init__(
        self,
        input_chord_pitch_type: PitchType,
        input_key_pitch_type: PitchType,
        output_pitch_type: PitchType,
        embed_dim: int = 64,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        use_inversions: bool = True,
        reduction: Dict[ChordType, ChordType] = None,
    ):
        """
        Vreate a new Simple Key Sequence Model.

        Parameters
        ----------
        input_chord_pitch_type : PitchType
            The pitch representation used in the input chord data.
        input_key_pitch_type : PitchType
            The pitch representation used in the input key data.
        output_pitch_type : PitchType
            The pitch representation used for the output data.
        embed_dim : int
            The size of the linear embedding layer.
        lstm_layers : int
            The number of lstm layers to use.
        lstm_hidden_dim : int
            The size of the LSTM hidden vector.
        hidden_dim : int
            The size of the first linear layer output.
        dropout : float
            The dropout proportion to use.
        learning_rate : float
            The learning rate.
        use_inversions : bool
            True to use inversions in the input vectors. False otherwise.
        reduction : Dict[ChordType, ChordType]
            The reduction to use for chord types.
        """
        super().__init__(
            input_chord_pitch_type,
            input_key_pitch_type,
            output_pitch_type,
            learning_rate,
        )
        self.save_hyperparameters()

        self.use_inversions = use_inversions
        self.reduction = reduction

        # Input and output derived from input type and use_inversions
        self.input_dim = (
            get_chord_vector_length(
                input_chord_pitch_type,
                one_hot=False,
                relative=True,
                use_inversions=use_inversions,
                pad=True,
                reduction=reduction,
            )
            + get_key_change_vector_length(input_key_pitch_type, one_hot=False)
            + 1
        )
        self.output_dim = get_key_change_vector_length(output_pitch_type, one_hot=True)

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
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 * for bi-directional
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
            Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden_dim)),
            Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden_dim)),
        )

    def forward(self, inputs, lengths, hidden=None):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size) if hidden is None else hidden

        embedded = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed, (h_0, c_0))
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)

        # Get lengths in proper format
        lstm_out_lengths_tensor = (
            lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand((-1, 1, lstm_out_forward.shape[2]))
        )
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze(dim=1)
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = self.fc2(relu2)

        return F.log_softmax(output, dim=1), hidden_out
