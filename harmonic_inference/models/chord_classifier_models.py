"""Models that generate probability distributions over chord classifications of a given input."""
from typing import Collection

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from pytorch_lightning import pl

from harmonic_inference.data.data_types import PieceType, PitchType
import harmonic_inference.utils.harmonic_constants as hc
import harmonic_inference.utils.harmonic_utils as hu


class ChordClassifierModel(pl.LightningModule):
    """
    The base type for all Chord Classifier Models, which take as input sets of frames from Pieces,
    and output chord probabilities for them.
    """
    def __init__(self, input_type: PieceType, output_type: PitchType):
        """
        Create a new base ChordClassifierModel with the given input and output formats.

        Parameters
        ----------
        input_type : PieceType
            The input type this model is expecting.
        output_type : PitchType
            The type of chord this model will output.
        """
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    def training_step(self, batch, batch_idx):
        notes = batch['inputs'].float()
        notes_lengths = batch['input_lengths']
        targets = batch['targets'].long()

        outputs = self.forward(notes, notes_lengths)
        loss = F.cross_entropy(outputs, targets)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        notes = batch['inputs'].float()
        notes_lengths = batch['input_lengths']
        targets = batch['targets'].long()

        outputs = self.forward(notes, notes_lengths)
        loss = F.cross_entropy(outputs, targets)

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


class SimpleChordClassifier(ChordClassifierModel):
    """
    The most simple chord classifier, with layers:
        1. Bi-LSTM
        2. Linear layer
        3. Dropout
        4. Linear layer
    """
    def __init__(
        self,
        input_type: PitchType,
        output_type: PitchType,
        use_inversions: bool,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        """
        Create a new SimpleChordClassifier.

        Parameters
        ----------
        input_type : PitchType
            The pitch type to use for inputs to this model. Used to derive the input length.
        output_type : PitchType
            The pitch type to use for outputs of this model. Used to derive the output length.
        use_inversions : bool
            Whether to use different inversions as different chords in the output. Used to
            derive the output length.
        lstm_layers : int
            The number of Bi-LSTM layers to use.
        lstm_hidden_dim : int
            The size of each LSTM layer's hidden vector.
        hidden_dim : int
            The size of the output vector of the first linear layer.
        dropout : float
            The dropout proportion of the first linear layer's output.
        """
        super().__init__(input_type, output_type)

        # Input and output derived from pitch_type and use_inversions
        self.input_dim = (
            hc.NUM_PITCHES[input_type] +  # Pitch class
            127 // hc.NUM_PITCHES[PitchType.MIDI] +  # octave
            12  # 4 onset level, 4 offset level, (onset, offset, dur) proportions, is_lowest
        )
        self.num_classes = len(hu.get_chord_label_list(output_type, use_inversions=use_inversions))

        # LSTM hidden layer and depth
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            self.input_dim,
            self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_dim, self.hidden_dim)  # 2 * because bi-directional
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
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

    def forward(self, notes, lengths):
        # pylint: disable=arguments-differ
        batch_size = notes.shape[0]
        if self.input_mask is not None:
            notes *= self.input_mask
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size)

        packed_notes = pack_padded_sequence(notes, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed_notes, (h_0, c_0))
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
        fc1 = self.fc1(relu1)
        relu2 = F.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = self.fc2(drop1)

        return output


class TranspositionInvariantCNNClassifier(nn.Module):
    """
    A transposition invariant CNN takes as input some (batch x num_input_channels x
    pitch_vector_length) matrix and classifies it in a transpositional invariant way.

    The last dimension should go along some representation of "pitches" such that a circular
    convolution along this dimension will represent transpositions of the input representation.
    The output channels of the convolutional layer are then fed into identical copies of the same
    feed-forward network.

    Parameters
    ----------
    num_chord_types : int
        The number of chord types for the network to output per root.

    num_hidden : int
        The number of hidden layers to use.

    hidden_size : int
        The number of nodes in the input layer and each hidden layer.

    batch_norm : boolean
        True to include batch normalization after the activation function of
        the input layer and each hidden layer.

    dropout : float
        The percentage of nodes in the input layer and each hidden layer to
        dropout. This is applied after activation (and before batch normalization
        if batch_norm is True, although it is not recommended to use both).

    input_mask : torch.tensor
        A binary tensor to multiply by the input (in forward()). Should be 1 in all
        locations except where to mask.
    """

    def __init__(self, num_chord_types, num_input_channels=1, pitch_vector_length=12,
                 num_conv_channels=10, num_hidden=1, hidden_size=100, batch_norm=False,
                 dropout=0.0, input_mask=None):
        super().__init__()

        self.input_mask = input_mask

        # Convolutional layer
        self.num_input_channels = num_input_channels
        self.pitch_vector_length = pitch_vector_length
        self.num_conv_channels = num_conv_channels

        self.conv = nn.Conv1d(self.num_input_channels, self.num_conv_channels,
                              self.pitch_vector_length, padding=self.pitch_vector_length,
                              padding_mode='circular')

        # Parallel linear layers
        self.num_chord_types = num_chord_types
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.input = nn.Linear(num_conv_channels, hidden_size)
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for i in range(num_hidden)])
        self.output = nn.Linear(hidden_size, num_chord_types)

        if batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size)
                                              for i in range(num_hidden + 1)])
        else:
            self.batch_norms = nn.ModuleList([None] * (num_hidden + 1))

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_hidden + 1)])

    def forward(self, data):
        # pylint: disable=arguments-differ
        if self.input_mask is not None:
            data *= self.input_mask

        # Conv layer
        conv = F.relu(self.conv(data.unsqueeze(1)))

        # Parallel linear layers
        parallel_in = conv.reshape(conv.shape[0] * 12, -1)

        # Input layer
        parallel = self.dropouts[0](F.relu(self.input(parallel_in)))
        if self.batch_norms[0] is not None:
            parallel = self.batch_norms[0](parallel)

        # Hidden layers
        for layer, dropout, batch_norm in zip(self.linear, self.dropouts[1:],
                                              self.batch_norms[1:]):
            parallel = dropout(F.relu(layer(parallel)))
            if batch_norm is not None:
                parallel = batch_norm(parallel)

        # Output layer
        parallel_out = F.relu(self.output(parallel))

        # Final output combination
        output = parallel_out.reshape(parallel_out.shape[0] / 12, -1)
        return output


class TransformerEncoder(nn.Module):
    """
    This model encodes a given input into a defined chord representation.

    Parameters
    ----------
    """

    def __init__(self):
        super().__init__()
        self.todo = True

    def forward(self, data):
        # pylint: disable=arguments-differ
        pass


class MusicScoreJointModel(nn.Module):
    """
    This model is a combination of an chord encoder (e.g., TransformerEncoder) and a
    chord classifier (e.g., TranspositionInvariantCNNClassifier). The output of the encoder is
    fed into the classifier.

    Parameters
    ----------
    encoder : nn.Module
        The chord encoder model.

    classifier : nn.Module
        The chord classifier model.
    """

    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, data: torch.tensor, stages: Collection[int]) -> torch.tensor:
        # pylint: disable=arguments-differ
        """
        Forward pass one or both modules.

        Parameters
        ----------
        data : torch.tensor
            A batch-first representation of the input data for the forward pass.

        stages : list
            A list of what stages to perform. If 0 is in the list, use the encoder.
            If 1 is in the list, use the classifier.
        """
        if 0 in stages:
            data = self.encoder.forward(data)
        if 1 in stages:
            data = self.classifier.forward(data)

        return data
