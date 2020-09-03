"""Models that output the probability of a chord change occurring on a given input."""
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from harmonic_inference.data.piece import PieceType, PitchType
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES


class ChordTransitionModel(pl.LightningModule):
    """
    The base class for all Chord Transition Models which model when a chord change will occur.
    """
    def __init__(self, input_type: PieceType):
        """
        Create a new base model.

        Parameters
        ----------
        input_type : PieceType, optional
            What type of input the model is expecting.
        """
        super().__init__()
        self.input_type = input_type

    def get_data_from_batch(self, batch):
        inputs = batch['inputs'].float()
        input_lengths = batch['input_lengths']
        inputs = inputs[:, :max(input_lengths)]

        target_indexes = batch['targets'].long()
        target_lengths = batch['target_lengths'].long()
        targets = torch.zeros(inputs.shape[:2]).float()
        for i, (index, length) in enumerate(zip(target_indexes, target_lengths)):
            targets[i, index[:length]] = 1.0

        mask = (inputs.sum(axis=2) > 0).long()

        return inputs, input_lengths, targets, mask

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

        outputs = self.forward(inputs, input_lengths)

        loss = F.binary_cross_entropy(outputs * mask, targets)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

        outputs = self.forward(inputs, input_lengths)

        loss = F.binary_cross_entropy(outputs * mask, targets)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.001
        )


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
        """
        super().__init__(input_type)

        # Input and output derived from input type and use_inversions
        if input_type == PieceType.SCORE:
            self.input_dim = (
                NUM_PITCHES[PitchType.TPC] +  # Pitch class
                127 // NUM_PITCHES[PitchType.MIDI] +  # octave
                9  # 4 onset level, 4 offset level,  is_lowest
            )
        elif input_type == PieceType.MIDI:
            self.input_dim = (
                NUM_PITCHES[PitchType.MIDI] +  # Pitch class
                127 // NUM_PITCHES[PitchType.MIDI] +  # octave
                9  # 4 onset level, 4 offset level, is_lowest
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
            batch_first=True
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 * because bi-directional
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
            Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden_dim))
        )

    def forward(self, inputs, lengths):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed, (h_0, c_0))
        lstm_out, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = torch.squeeze(self.fc2(relu2), -1)

        return torch.sigmoid(output)
