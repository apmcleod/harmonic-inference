"""Models that output the probability of a key change occurring on a given input."""
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from harmonic_inference.data.data_types import PitchType
import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES


class KeyTransitionModel(pl.LightningModule):
    """
    The base class for all Key Transition Models which model when a key change will occur.
    """
    def __init__(self, input_type: PitchType = None):
        """
        Create a new base model.

        Parameters
        ----------
        input_type : PitchType, optional
            What type of input the model is expecting in get_change_prob(input_data).
        """
        super().__init__()
        # pylint: disable=invalid-name
        self.INPUT_TYPE = input_type

    def get_data_from_batch(self, batch):
        inputs = batch['inputs'].float()
        input_lengths = batch['input_lengths']
        max_length = max(input_lengths)
        inputs = inputs[:, :max_length]

        targets = batch['targets'].float()[:, :max_length]

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

        flat_mask = mask.reshape(-1)
        outputs = outputs.reshape(-1)[flat_mask]
        targets = targets.reshape(-1)[flat_mask]
        acc = 100 * (outputs.round().long() == targets).sum().float() / len(outputs)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        result.log('val_acc', acc)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.001
        )


class SimpleKeyTransitionModel(KeyTransitionModel):
    """
    The most simple key transition model, with layers:
        1. Linear embedding layer
        2. LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer
    """
    def __init__(
        self,
        input_type: PitchType,
        embed_dim: int = 128,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        """
        Create a new simple key transition model.

        Parameters
        ----------
        input_type : PitchType
            The type of input data for this model.
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

        # Input derived from input type
        self.input_dim = (
            NUM_PITCHES[input_type] +  # Root
            NUM_PITCHES[input_type] +  # Bass
            13  # 4 inversion, 4 onset level, 4 offset level, is_major
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
            batch_first=True
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
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
            Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim)),
            Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim))
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
