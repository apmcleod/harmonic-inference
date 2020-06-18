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