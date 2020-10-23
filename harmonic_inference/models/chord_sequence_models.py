"""Models that generate probability distributions over the next chord in a sequence."""
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from harmonic_inference.data.data_types import PitchType
from harmonic_inference.data.piece import Chord, Key


class ChordSequenceModel(pl.LightningModule):
    """
    The base class for all Chord Sequence Models, which model the sequence of chords of a Piece.
    """

    def __init__(self, chord_type: PitchType, learning_rate: float):
        """
        Create a new base KeySequenceModel with the given output and input data types.

        Parameters
        ----------
        chord_type : PitchType
            The way a given model will output its chords.
        learning_rate : float
            The learning rate.
        """
        super().__init__()
        # pylint: disable=invalid-name
        self.CHORD_TYPE = chord_type
        self.lr = learning_rate

    def get_data_from_batch(self, batch):
        inputs = batch["inputs"].float()
        targets = batch["targets"].long()
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"].long()

        longest = max(input_lengths)
        inputs = inputs[:, :longest]

        targets = targets[:, :longest]
        targets[inputs[:, :, -1] == 1] = -100
        targets[:, 0] = -100
        for i, length in enumerate(target_lengths):
            targets[i, length:] = -100
        targets = torch.roll(targets, -1, dims=1)

        return inputs, input_lengths, targets

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets = self.get_data_from_batch(batch)

        outputs, _ = self(inputs, input_lengths)

        loss = F.nll_loss(outputs.permute(0, 2, 1), targets, ignore_index=-100)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets = self.get_data_from_batch(batch)

        outputs, _ = self(inputs, input_lengths)

        loss = F.nll_loss(outputs.permute(0, 2, 1), targets, ignore_index=-100)

        targets = targets.reshape(-1)
        mask = targets != 100
        outputs = outputs.argmax(-1).reshape(-1)[mask]
        targets = targets[mask]
        acc = 100 * (outputs == targets).sum().float() / len(targets)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)

    def init_hidden(self, batch_size: int):
        # Subclasses should implement this
        raise NotImplementedError

    def run_one_step(self, batch):
        inputs = batch["inputs"].float()
        hidden = (
            torch.transpose(batch["hidden_states"][0], 0, 1),
            torch.transpose(batch["hidden_states"][1], 0, 1),
        )

        return self(inputs, torch.ones(len(inputs)), hidden=hidden)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )


class SimpleChordSequenceModel(ChordSequenceModel):
    """
    The most simple chord sequence model, with layers:
        1. Linear embedding layer
        2. LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer
    """

    def __init__(
        self,
        chord_type: PitchType,
        use_inversions: bool = True,
        embed_dim: int = 64,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
    ):
        """
        Create a new simple chord sequence model.

        Parameters
        ----------
        chord_type : PitchType
            The type of pitch representation used for the chords, input and output.
        use_inversions : bool
            True to take inversions into account. False to ignore them.
        embed_dim : int
            The size of the input embedding.
        lstm_layers : int
            The number of bi-directional LSTM layers.
        lstm_hidden_dim : int
            The size of the LSTM's hidden dimension.
        hidden_dim : int
            The size of the hidden dimension between the 2 consecutive linear layers.
        dropout : float
            The dropout proportion.
        learning_rate : float
            The learning rate.
        """
        super().__init__(chord_type, learning_rate)
        self.save_hyperparameters()

        # Input and output derived from input type and use_inversions
        self.input_dim = (
            Chord.get_chord_vector_length(
                chord_type,
                one_hot=False,
                relative=True,
                use_inversions=use_inversions,
                pad=False,
            )
            + Key.get_key_change_vector_length(chord_type, one_hot=False)  # Key change vector
            + 1  # is_key_change
        )
        self.output_dim = Chord.get_chord_vector_length(
            chord_type,
            one_hot=True,
            relative=True,
            use_inversions=use_inversions,
            pad=False,
        )

        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.input_dim, self.embed_dim)

        # LSTM hidden layer and depth
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=False,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
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
            Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim)),
            Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim)),
        )

    def forward(self, inputs, lengths, hidden=None):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size) if hidden is None else hidden

        embedded = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed, (h_0, c_0))
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = self.fc2(relu2)

        return F.log_softmax(output, dim=2), hidden_out
