"""Models for harmonic inference datasets."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as f


class MusicScoreModel(nn.Module):
    """
    """
    def __init__(self, input_dim, num_classes, lstm_layers=1, lstm_dim=256, hidden_dim=256, dropout=0,
                 input_mask=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.input_mask = input_mask
        
        self.lstm = nn.LSTM(self.input_dim,
                            self.lstm_dim,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        
        self.fc1 = nn.Linear(2 * self.lstm_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout1 = nn.Dropout(self.dropout)
        
        
        
    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_dim)),
                Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_dim)))
        return h, c
    
    
    
    def forward(self, notes, lengths):
        batch_size = notes.shape[0]
        if self.input_mask is not None:
            notes *= self.input_mask
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size)
        
        packed_notes = pack_padded_sequence(notes, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (h_n, c_n) = self.lstm(packed_notes, (h_0, c_0))
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)
        
        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)
        
        # Get lengths in proper format
        lstm_out_lengths_tensor = lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand((-1, 1, lstm_out_forward.shape[2]))
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze()
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)
        
        relu1 = f.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = f.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = self.fc2(drop1)
        
        return output
    
    
    
    
class MusicScoreTransformerModel(nn.Module):
    """
    """
    def __init__(self, num_chord_types, num_nonrooted_labels=0, num_channels=10, num_hidden=1, hidden_size=100,
                 batch_norm=False, dropout=0.0, input_maks=None):
        """
        Parameters
        ----------
        num_chord_types : int
            The number of chord types for the network to output per root.
            
        num_nonrooted_labels : int
            The number of non-root chord types for the model to output, total.
            
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
        """
        super().__init__()
        
        self.input_mask = input_mask
        
        # Transformer
        
        
        # Upper layers
        self.num_qualities = num_qualities
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        self.conv = nn.Conv1d(1, num_channels, 12, padding=12, padding_mode='circular')
        
        self.input = nn.Linear(num_channels, hidden_size)
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_hidden)])
        self.output = nn.Linear(hidden_size, num_qualities)
        
        self.batch_norms = nn.ModuleList([None] * (num_hidden + 1))
        if batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(num_hidden + 1)])
            
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_hidden + 1)])
    
    
    
    def forward_1(self, notes, lengths):
        batch_size = notes.shape[0]
        if self.input_mask is not None:
            notes *= self.input_mask
        lengths = torch.clamp(lengths, min=1)
        h_0, c_0 = self.init_hidden(batch_size)
        
        packed_notes = pack_padded_sequence(notes, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (h_n, c_n) = self.lstm(packed_notes, (h_0, c_0))
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)
        
        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)
        
        # Get lengths in proper format
        lstm_out_lengths_tensor = lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand((-1, 1, lstm_out_forward.shape[2]))
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze()
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)
        
        relu1 = f.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = f.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = self.fc2(drop1)
        
        return output
    
    def forward_2(self, data):
        # Conv layer
        conv = F.relu(self.conv(data.unsqueeze(1)))
        
        # Parallel linear layers
        parallel_in = conv.reshape(conv.shape[0] * 12, -1)
        
        # Input layer
        parallel = self.dropouts[0](F.relu(self.input(parallel_in)))
        if self.batch_norms[0] is not None:
            parallel = self.batch_norms[0](parallel)
            
        # Hidden layers
        for layer, dropout, bn in zip(self.linear, self.dropouts[1:], self.batch_norms[1:]):
            parallel = dropout(F.relu(layer(parallel)))
            if bn is not None:
                parallel = bn(parallel)
        
        # Output layer
        parallel_out = F.relu(self.output(parallel))
        
        # Final output combination
        output = output.reshape(output.shape[0] / 12, -1)
        return output