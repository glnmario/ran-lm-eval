##############################################################
# This file is need for back-compatibility of loaded models. #
##############################################################

import torch.nn as nn
from torch.autograd import Variable
from torch import from_numpy as to_tensor

from ran import RAN

class RNNModel(nn.Module):
    """
    Container module with an embedding layer, a recurrent layer, and an
    output layer.
    """

    def __init__(self,
                 rnn_type,
                 vocab_size,
                 embed_dims,
                 n_units,
                 n_layers,
                 embeddings=None,
                 bidirectional=False,
                 dropout=0.2,
                 tie_weights=False):
        super(RNNModel, self).__init__()

        # optionally add dropout regularisation
        self.dropout = nn.Dropout(dropout)

        # the embedding matrix of size |V| x d
        self.embed = nn.Embedding(vocab_size, embed_dims)
        if embeddings is not None:
            self.embed.weight = nn.Parameter(to_tensor(embeddings))

        self.bidir = bidirectional

        # select the correct architecture

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_type = rnn_type
            self.rnn = getattr(nn, rnn_type)(embed_dims,
                                             n_units,
                                             n_layers,
                                             dropout=dropout,
                                             bidirectional=self.bidir)
        elif rnn_type == 'RAN':
            self.rnn = RAN(embed_dims, n_units, n_layers, dropout=dropout)
        else:
            try:
                model_info = rnn_type.split("_")
                self.rnn_type = model_info[0]
                nonlinearity = model_info[1].lower()
            except KeyError:
                raise ValueError("An invalid option for `--model` was supplied.\
                                 Options are ['LSTM', 'GRU', 'RNN_TANH', or\
                                 'RNN_RELU']")
            self.rnn = nn.RNN(embed_dims,
                              n_units,
                              n_layers,
                              nonlinearity=nonlinearity,
                              dropout=dropout,
                              bidirectional=self.bidir)

        # bidirectional needs 2x units
        n = int(self.bidir) + 1
        # output is linear as softmax is applied within the loss function
        self.output = nn.Linear(n * n_units, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf,
        # 2016) https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016) https://arxiv.org/abs/1611.01462
        if tie_weights:
            if n_units != embed_dims:
                raise ValueError('When using the tied flag, n_units must be\
                                 equal to embdims')
            self.output.weight = self.embed.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers


    def init_weights(self):
        nn.init.xavier_uniform(self.embed.weight)
        self.output.bias.data.fill_(0)
        nn.init.xavier_uniform(self.output.weight)

        # This was the original RNN initialisation:
        # initrange = 0.1
        # self.embed.weight.data.uniform_(-initrange, initrange)
        # self.output.bias.data.fill_(0)
        # self.output.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden):
        # Apply dropout to embedding layer as in "A Theoretically Grounded
        # Application of Dropout in Recurrent Neural Networks" (Gal &
        # Ghahramani, 2016) https://arxiv.org/pdf/1512.05287.pdf

        # embed = self.dropout(self.embed(input))
        embed = self.encoder(input)
        rnn_output, hidden = self.rnn(embed, hidden)
        # rnn_output = self.dropout(rnn_output)
        output = self.decoder(rnn_output.view(
                                        rnn_output.size(0) * rnn_output.size(1),
                                        rnn_output.size(2)))

        return output.view(rnn_output.size(0),
                           rnn_output.size(1),
                           output.size(1)
                          ), hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        try:
            n = int(self.bidir) + 1  # bidirectional needs 2x units
        except AttributeError:
            n = 1

        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(
                        n * self.n_layers, batch_size, self.n_units).zero_()),
                   (Variable(weight.new(
                        n * self.n_layers, batch_size, self.n_units).zero_())))
        else:
            try:
                return Variable(weight.new(
                        n * self.n_layers, batch_size, self.n_units).zero_())
            except AttributeError:
                return Variable(weight.new(
                        n * self.nlayers, batch_size, self.nhid).zero_())
