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



# load model for CPU
with open(model_path, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)
    model.eval()
    model.cpu()

w2i = corpus.dictionary.word2idx
ntokens = len(corpus.dictionary)

filename = 'verb_form_attractor.txt'
with open('sentences/{}'.format(filename), 'r') as f_in, open('sentences/out_{}'.format(filename), 'w') as f_out:
    for line in f_in:
        sent = line.strip().split()

        verbs = []
        idx = None

        for i, word in enumerate(sent):
            if "/" in word:
                verbs = word[1:].split("/")
                stop_idx = i

        sent_len = stop_idx + 1

        verb_form_in_corpus = True

        sentences = []
        sentences_with_ids = []
        for verb in verbs:
            if verb not in w2i.keys():
                verb_form_in_corpus = False
                break

            # convert sentence to a list of word indices
            input_word_indices = [w2i[w] for i, w in enumerate(sent) if i < stop_idx] + [w2i[verb]]
            sentences_with_ids.append(Variable(torch.LongTensor(input_word_indices)))
            sentences.append([w for i, w in enumerate(sent) if i < stop_idx] + [verb])

        if not verb_form_in_corpus: continue

        # delete old output if necessary
        try:
            os.remove('./values.txt')
        except FileNotFoundError:
            pass

        for idx, s_with_ids in enumerate(sentences_with_ids):
            # initialise hidden for first timestep
            hidden = model.init_hidden(batch_size=1)
            # forward pass
            output, hidden = model(s_with_ids, hidden)
            # linear output
            output_flat = output.view(-1, ntokens)
            # apply softmax manually
            softmax = nn.Softmax()
            output_flat = softmax(output_flat)



            ############################################################
            ########################  ANALYSIS  ########################

            # compute sentence probability and perplexity
            p = 1
            for i, w in enumerate(s_with_ids):
                p *= output_flat[i][w].data[0]

            pp = p ** (-1 / sent_len)

            print('\n>> {} <<'.format(sentences[idx]), file=f_out)
            print('Probability:', p, file=f_out)
            print('Perplexity:', pp, '\n', file=f_out)


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
            w = np.zeros((sent_len, sent_len, ctilde_list.shape[0]))
            for t in range(sent_len):
                for j in range(sent_len):
                    f_prod = 1
                    for k in range(j+1, t):
                        f_prod *= f_list[k]

                    w[t][j] = i_list[j] * f_prod

            # for each word, print the most active history word and the list of all activations
            w_c_all = []
            for t in range(sent_len):
                if t == 0:
                    print(sentences[idx][t], '[]\n', sep='\n', file=f_out)
                    continue

                sums = np.zeros((sent_len, ctilde_list.shape[0]))
                for i in range(t):
                    sums[i] = w[t][i] * ctilde_list[i]

                w_c_all.append(sums)

            for l, k in enumerate(w_c_all):
                activations = np.sum(np.absolute(k), axis=1)

                print(sentences[idx][l+1], '->', sentences[idx][np.argmax(activations)], file=f_out)
                print(activations, '\n', file=f_out)
        print('\n\n', file=f_out)
