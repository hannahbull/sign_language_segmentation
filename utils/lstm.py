import torch
from torch import nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    """
    Simple LSTM model based on:
    https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """
    def __init__(self, emb_dim=10, hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, 1)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2,).cuda(),
            torch.randn(2, batch_size, self.hidden_dim // 2).cuda(),
        )

    def forward(self, embeds):
        self.hidden = self.init_hidden(embeds.shape[0])
        self.hidden = (self.hidden[0].type(torch.FloatTensor).cuda(), self.hidden[1].type(torch.FloatTensor).cuda()) 
        x, self.hidden = self.lstm(embeds, self.hidden)
        x = self.hidden2tag(x)
        x = x.view(embeds.shape[0], -1)
        return x