# coding: utf-8
import argparse
import time
import math
import csv
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import data
import model


model_path = '../models/model-ran-256-85.67ppl.pt'
corpus = data.Corpus('../data/penn/')
embed_dims = 50

sentence = 'i will go to the restaurant'
sentence += ' <eos>'

w2i = corpus.dictionary.word2idx

with open(model_path, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)
    model.eval()
    model.cpu()

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(batch_size=1)

input_word_indices = [w2i[w] for w in sentence.split()]
input_word_indices = Variable(torch.LongTensor(input_word_indices))

output, hidden = model(input_word_indices, hidden)
output_flat = output.view(-1, ntokens)

softmax = nn.Softmax()
output_flat = softmax(output_flat)

p = 1
for i, w in enumerate(sentence.split()):
    p *= output_flat[i][w2i[w]].data[0]

pp = p ** (-1/len(sentence))

print(sentence)
print('Probability:', p)
print('Perplexity:', pp)
