import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator


# RNN module
class RNN_base(nn.Module):
    def __init__(self, input_size = 128, hidden_size = 50, output_size = 128, n_layers=1):
        super(RNN_base, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.hidden_state = None
        self.reduction = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, self.hidden_state = self.rnn(self.reduction(x))
        out = self.fc(out)
        return out, self.hidden_state
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, S=64, output_size=1, n_layers=2):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.s = S
        self.out_size = output_size
        self.n_layers = n_layers
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        
        # Calculate the size of the features after the CNN
        # cnn_output_size = self._get_conv_output((input_size, S, S))
        cnn_output_size = 32*16*16 # check by uncommenting the print part in forward()
        
        # LSTM layer
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, n_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size*S*S)
        
    def forward(self, x):
        # Apply CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        # print('After first conv1', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print('After first conv2', x.shape)
        
        # Reshape output for LSTM layer, flatten
        x = x.view(x.size()[0], -1)  # Assuming batch_first=True
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # print('After LSTM', lstm_out.shape)
        
        # Fully connected layer
        out = self.fc(lstm_out)  # Use the last LSTM output
        
        return out.reshape(out.size()[0], self.s, self.s, self.out_size)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
