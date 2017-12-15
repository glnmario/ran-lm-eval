# coding: utf-8
import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

import data
import model

def pretty_print(sentence, activations, t, f_out=None):
    influencers_sorted = list(np.sort(activations[:t+1])[::-1])
    relative_order = [influencers_sorted.index(x) for x in list(activations[:t+1])]

    num_str = '  '
    for i, word in enumerate(sentence[:t]):
        num_str += str(relative_order[i]) + ' ' * (len(word)+3)

    print(sentence[t], '->', sentence[np.argmax(activations)], file=f_out)
    print(num_str, file=f_out)
    print(sentence[:t+1], file=f_out)
    print(activations[:t], '\n', file=f_out)


def plot_influence_sent(inf_matrix, sent):
    assert len(inf_matrix) == len(sent)
    print(sent)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(inf_matrix, cmap=plt.cm.Blues)
    cbar = fig.colorbar(cax)

    ax.set_xticks(np.arange(len(sent)))
    ax.set_yticks(np.arange(len(sent)))
    ax.set_xticklabels(sent, rotation=45)
    ax.set_yticklabels(sent, rotation=45)
    ax.axis('image')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse RAN language model.')
    parser.add_argument('--file', type=str,  help='path of the input file (a list of annotated sentences)')
    parser.add_argument('--mode', default='l1', help='l1, l2, max_w, or l1_c')
    args = parser.parse_args()

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

    # load dictionary from corpus
    w2i = corpus.dictionary.word2idx
    ntokens = len(corpus.dictionary)


    w_old = None
    c_old = None

    filename = args.file
    with open('sentences/{}'.format(filename), 'r') as f_in, open('sentences/'+'out_'+ most_influential_word_mode+'_{}'.format(filename), 'w') as f_out:
        for line in f_in:
            sent = line.strip().split()

            verbs = []  # candidates for next word on which we focus
            idx = None

            # collect all candidates
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

                # convert sentence to a list of word indices (up to index of the target verb)
                input_word_indices = [w2i[w] for i, w in enumerate(sent) if i < stop_idx] + [w2i[verb]]
                sentences_with_ids.append(Variable(torch.LongTensor(input_word_indices)))

                # but still keep the actual sentence (interrupted at target verb)
                sentences.append([w for i, w in enumerate(sent) if i < stop_idx] + [verb])

            # skip sentence if one inflection of the target verb is not in the dictionary
            if not verb_form_in_corpus:
                continue

            # now analyse all variants of each sentence (one variant for each candidate)
            for idx, s_with_ids in enumerate(sentences_with_ids):

                # delete old output if necessary
                try:
                    os.remove('./ctilde.npy')
                    os.remove('./i.npy')
                    os.remove('./f.npy')
                except FileNotFoundError:
                    pass


                hidden = model.init_hidden(batch_size=1)    # initialise hidden for first timestep
                output, hidden = model(s_with_ids, hidden)  # forward pass
                output_flat = output.view(-1, ntokens)      # linear output
                softmax = nn.Softmax()
                output_flat = softmax(output_flat)          # manually apply softmax


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
                ctilde_list = np.load('./ctilde.npy')
                i_list = np.load('./i.npy')
                f_list = np.load('./f.npy')

                # compute 3D matrix fo weights
                w = np.zeros((sent_len, sent_len, ctilde_list.shape[1]))
                for t in range(sent_len):
                    for j in range(t+1):
                        f_prod = np.ones((f_list.shape[1]))
                        for k in range(j+1, t+1):
                            f_prod *= f_list[k]

                        w[t][j] = i_list[j] * f_prod

                # select visualisation mode: l1_c, l1, l2, max_w
                if args.mode == 'l1_c':
                    w_c_all = []
                    for t in range(sent_len):
                        if t == 0:
                            print(sentences[idx][t], '[]\n', sep='\n', file=f_out)
                            continue

                        w_c = np.zeros((sent_len, ctilde_list.shape[1]))
                        for i in range(t):
                            w_c[i] = w[t][i] * ctilde_list[i]

                        w_c_all.append(w_c)

                    for t, w_c in enumerate(w_c_all):
                        activations = np.sum(np.absolute(w_c), axis=1)
                        pretty_print(sentences[idx], activations, t, f_out)
                        #print(sentences[idx][t+1], '->', sentences[idx][np.argmax(activations)], file=f_out)
                        #print(activations[:t+1], '\n', file=f_out)

                elif args.mode == 'max_w':
                    influence_matrix = np.zeros((sent_len, sent_len))
                    for t, w_vec in enumerate(w):
                        activations = np.max(np.absolute(w_vec), axis=1)
                        influence_matrix[t] = activations

                        pretty_print(sentences[idx], activations, t, f_out)

                    plot_influence_sent(influence_matrix.T, sentences[idx])

                elif args.mode == 'l2':
                    influence_matrix = np.zeros((sent_len, sent_len))
                    for t, w_vec in enumerate(w):
                        activations = np.linalg.norm(w_vec, axis=1)
                        influence_matrix[t] = activations

                        pretty_print(sentences[idx], activations, t, f_out)

                    plot_influence_sent(influence_matrix.T, sentences[idx])

                elif args.mode == 'l1':
                    influence_matrix = np.zeros((sent_len, sent_len))
                    for t, w_vec in enumerate(w):
                        activations = np.linalg.norm(w_vec, ord=1, axis=1)
                        influence_matrix[t] = activations

                        pretty_print(sentences[idx], activations, t, f_out)

                    plot_influence_sent(influence_matrix.T, sentences[idx])
            print('\n\n', file=f_out)
