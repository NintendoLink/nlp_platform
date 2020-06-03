# python 3.6.4
# encoding: utf-8
import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement Position Embedding function"
    def __init__(self,d_model,dropout,max_len = 1000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0.,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.,d_model,2) * -(math.log(10000.) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        x =  x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)