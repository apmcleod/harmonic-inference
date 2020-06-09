"""Models for harmonic inference datasets."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as f


class MusicScoreModel(nn.Module):
    """
    """
    def __init__(self, input_dim, num_classes, lstm_layers=1, lstm_dim=256, hidden_dim=256, dropout=0):
        super().__init__()
        
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_dim,
                            self.lstm_dim,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        
        self.fc1 = nn.Linear(2 * lstm_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        
        
        
    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_dim)),
                Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_dim)))
        return h, c
    
    
    
    def forward(self, notes, lengths):
        batch_size = notes.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        
        packed_notes = pack_padded_sequence(notes, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_full, (h_n, c_n) = self.lstm(packed_notes, (h_0, c_0))
        lstm_out_unpacked, lengths = pad_packed_sequence(lstm_out_full, batch_first=True)
        
        lstm_out = lstm_out_unpacked[:, :, :]
        
        relu1 = f.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = f.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = self.fc2(drop1)
        
        return output