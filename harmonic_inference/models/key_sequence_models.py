"""Models that generate probability distributions over the next key in a sequence."""
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from harmonic_inference.data.data_types import PitchType, KeyMode
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES


class KeySequenceModel(pl.LightningModule):
    """
    The base class for all Key Sequence Models, which model the sequence of keys of a Piece.
    """
    def __init__(self, key_type: PitchType, input_type: PitchType, learning_rate: float):
        """
        Create a new base KeySequenceModel with the given output and input data types.

        Parameters
        ----------
        key_type : PitchType
            The way a given model will output its key tonics.
        input_type : PitchType, optional
            If a model will take input data, the format of that data.
        learning_rate : float
            The learning rate.
        """
        super().__init__()
        self.INPUT_TYPE = input_type
        self.KEY_TYPE = key_type
        self.lr = learning_rate

    def get_data_from_batch(self, batch):
        inputs = batch['inputs'].float()
        targets = batch['targets'].long()
        input_lengths = batch['input_lengths']

        longest = max(input_lengths)
        inputs = inputs[:, :longest]

        return inputs, input_lengths, targets

    def get_output(self, batch):
        # TODO
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets = self.get_data_from_batch(batch)

        outputs = self(inputs, input_lengths)

        loss = F.nll_loss(outputs, targets)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets = self.get_data_from_batch(batch)

        outputs = self(inputs, input_lengths)

        loss = F.nll_loss(outputs, targets)
        acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('val_loss', loss)
        result.log('val_acc', acc)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.001
        )


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
        input_type: PitchType,
        key_type: PitchType,
        embed_dim: int = 64,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
    ):
        """
        Vreate a new Simple Key Sequence Model.

        Parameters
        ----------
        input_type : PitchType
            The pitch representation used in the input data.
        key_type : PitchType
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
        """
        super().__init__(key_type, input_type, learning_rate)
        self.save_hyperparameters()

        # Input and output derived from input type and use_inversions
        self.input_dim = (
            NUM_PITCHES[input_type] +  # Root
            NUM_PITCHES[input_type] +  # Bass
            13  # 4 inversion, 4 onset level, 4 offset level, is_major
        )
        self.output_dim = len(KeyMode) * NUM_PITCHES[key_type]

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
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)

        # Get lengths in proper format
        lstm_out_lengths_tensor = lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand(
            (-1, 1, lstm_out_forward.shape[2])
        )
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze()
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = self.fc2(relu2)

        return F.log_softmax(output, dim=1)
