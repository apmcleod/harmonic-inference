"""Models that output the probability of a chord change occurring on a given input."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl
from harmonic_inference.data.data_types import PieceType, PitchType
from harmonic_inference.data.piece import Note


class ChordTransitionModel(pl.LightningModule):
    """
    The base class for all Chord Transition Models which model when a chord change will occur.
    """

    def __init__(self, input_type: PieceType, learning_rate: float):
        """
        Create a new base model.

        Parameters
        ----------
        input_type : PieceType
            What type of input the model is expecting.
        learning_rate : float
            The learning rate.
        """
        super().__init__()
        self.INPUT_TYPE = input_type
        self.lr = learning_rate

    def get_data_from_batch(self, batch):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]
        inputs = inputs[:, : max(input_lengths)]

        targets = batch["targets"].long()
        target_lengths = batch["target_lengths"].long()
        targets = targets[:, : max(target_lengths)]

        mask = ((inputs.sum(axis=2) > 0) & (targets != -100)).bool()

        return inputs, input_lengths, targets, mask

    def get_output(self, batch):
        inputs, input_lengths, _, _ = self.get_data_from_batch(batch)
        outputs = self(inputs, input_lengths)
        return outputs, input_lengths

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

        outputs = self(inputs, input_lengths)

        flat_mask = mask.reshape(-1)
        outputs = outputs.reshape(-1)[flat_mask]
        targets = targets.reshape(-1)[flat_mask]

        loss = F.binary_cross_entropy(outputs, targets.float())

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

        outputs = self(inputs, input_lengths)

        flat_mask = mask.reshape(-1)
        outputs = outputs.reshape(-1)[flat_mask]
        targets = targets.reshape(-1)[flat_mask]

        loss = F.binary_cross_entropy(outputs, targets.float())
        acc = 100 * (outputs.round() == targets).sum().float() / len(outputs)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


class SimpleChordTransitionModel(ChordTransitionModel):
    """
    The most simple chord transition model, with layers:
        1. Linear embedding layer
        2. Bi-LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer
    """

    def __init__(
        self,
        input_type: PieceType,
        embed_dim: int = 64,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
    ):
        """
        Create a new simple chord transition model.

        Parameters
        ----------
        input_type : PieceType
            The type of piece that the input data is coming from.
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
        super().__init__(input_type, learning_rate)
        self.save_hyperparameters()

        # Input and output derived from input type and use_inversions
        self.input_dim = Note.get_note_vector_length(
            PitchType.TPC if input_type == PieceType.SCORE else PitchType.MIDI
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
            bidirectional=True,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 because bi-directional
        self.fc2 = nn.Linear(self.hidden_dim, 1)
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

    def forward(self, inputs, lengths):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed, (h_0, c_0))
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = torch.squeeze(self.fc2(relu2), -1)

        return torch.sigmoid(output)
