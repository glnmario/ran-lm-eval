# coding: utf-8
import argparse
import time
import math
import csv
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import data
import model

# TO EDIT
#-------------------------------------------------#
model_path = '../models/model-ran-256-85.67ppl.pt'
corpus = data.Corpus('../data/penn/')
embed_dims = 256
#-------------------------------------------------#

# TO EDIT
#-------------------------------------------------#
sentence = "i like pizza a lot"
#-------------------------------------------------#

sentence += ' <eos>'
sent_aslist = sentence.split()
sentence_len = len(sent_aslist)

# load model for CPU
with open(model_path, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)
    model.eval()
    model.cpu()

w2i = corpus.dictionary.word2idx
ntokens = len(corpus.dictionary)

# convert sentence to a list of word indices
input_word_indices = [w2i[w] for w in sent_aslist]
input_word_indices = Variable(torch.LongTensor(input_word_indices))

# delete old output if necessary
try:
    os.remove('./values.txt')  
except FileNotFoundError:
    pass


# initialise hidden for first timestep
hidden = model.init_hidden(batch_size=1)
# forward pass
output, hidden = model(input_word_indices, hidden)
# linear output
output_flat = output.view(-1, ntokens)
# apply softmax manually
softmax = nn.Softmax()
output_flat = softmax(output_flat)


############################################################
########################  ANALYSIS  ########################

# compute sentence probability and perplexity
p = 1
for i, w in enumerate(sent_aslist):
    p *= output_flat[i][w2i[w]].data[0]

pp = p ** (-1/sentence_len)

print('\n>> {} <<'.format(sentence))
print('Probability:', p)
print('Perplexity:', pp, '\n')


# now we recover values for candidates, input gates, and
# forget gates, which have been saved during forward pass
ctilde_list = []
i_list = []
f_list = []

with open('./values.npy', 'rb') as f:
    values = np.load(f)
    
    ctilde_list = values[0]
    i_list = values[1]
    f_list = values[2]

# compute 3D matrix fo weights
w = np.zeros((sentence_len, sentence_len, ctilde_list.shape[0]))
for t in range(sentence_len-1):
    for j in range(sentence_len-1):
        f_prod = 1
        for k in range(j+1, t):
            f_prod *= f_list[k]
        
        w[t][j] = i_list[j] * f_prod
         
print("weights ", w.shape)

# for each word, print the most active history word and the list of all activations
w_c_all = []
for t, word in enumerate(sent_aslist[:-1]):
   
    if t == 0:
        print(word, '[]\n', sep='\n')
        continue

    sums = np.zeros((t, ctilde_list.shape[0]))
    
    for i in range(t):
        
        sums[i] = (w[t][i] * ctilde_list[i])
        
    w_c_all.append(sums)
    
    


for l,k in enumerate(w_c_all):
   
    words = sentence.split()[1:-1]
        
        
        
    activations = np.sum(k, axis=1)

    print(activations.shape)
    print(words[l], '->', sent_aslist[np.argmax(activations)])
    print(activations, '\n')

print(ctilde_list.shape[0])

