# python 3.6.4
# encoding: utf-8
import torch
import torch.nn as nn

class RNNTextClassifier(nn.Module):

    """
    用于文本分类的RNN分类器
        1、情感分析
        2、文本分类
    """
    def __init__(self,config):

        super(RNNTextClassifier, self).__init__()
        self.config = config

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings= config.word_count,
                                      embedding_dim=config.embedding_dim)
        # lstm_out layer
        self.rnn = nn.LSTM(input_size=config.embedding_dim,
                           bidirectional=config.bidirectional,
                           num_layers=config.num_layers,
                           dropout=config.dropout,
                           hidden_size=config.hidden_size)
        # fc layers
        self.linear = nn.Linear(in_features=config.hidden_size,
                                out_features=config.class_num)

        # softmax layer
        self.softmax = nn.Softmax()

    def forward(self, sent_word):
        # data flow
        # embedding--->>>lstm--->>>fc--->>>softmax
        x = self.embedding(sent_word)

        lstm_out, (h_n, c_n) = self.rnn(x)

        final_feature_map = self.dropout(h_n)
        final_feature_map = torch.cat([final_feature_map[i] for i in range(final_feature_map.size(0))], dim=1)

        final_out = self.linear(final_feature_map)
        return self.softmax(final_out)

    def add_loss_fn(self,loss_fn):
        self.loss_fn = loss_fn

    def add_optim(self,optimizer):
        self.optimizer = optimizer

    def run_epoch(self):
        return None

    def evalute(self):
        return None

class CNNTextClassifier(nn.Module):

    def __init__(self,config):
        super(CNNTextClassifier, self).__init__()
        self.config  = config

    def forward(self, *input):
        return None

    def add_loss_fn(self,loss_fn):
        self.loss_fn = loss_fn

    def add_optim(self,optimizer):
        self.optimizer = optimizer

    def run_epoch(self):
        return None

    def evalute(self):
        return None
