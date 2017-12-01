#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:59:47 2017

@author: mario
"""

import numpy as np
import re

import nltk
from nltk.tag.perceptron import PerceptronTagger

import data


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


corpus = []

with open('../data/penn/train.txt', 'r') as f_in:
    for line in f_in:
        corpus.append(line.strip())

with open('../data/penn/valid.txt', 'r') as f_in:
    for line in f_in:
        corpus.append(line.strip())

with open('../data/penn/test.txt', 'r') as f_in:
    for line in f_in:
        corpus.append(line.strip())


tagger = PerceptronTagger()

out_filename = './sentences/subj-verb.txt'
f = open(out_filename, 'w')

with_verb_count = 0  # count how many sentences match our search

for i, sent in enumerate(corpus, start=1):
    # Logging
    if i % 10000 == 0:
        print('{} sentences analysed.'.format(i))

    # Prune sentences with unknown tokens
    if '<unk>' in sent:
        continue

    tagged_sent = tagger.tag(sent.split())
    with_verb = False

    for w, t in tagged_sent:
        # 3rd and non-3rd person singular, present tense
        if t in ('VBZ', 'VBP'):
            with_verb = True
            sent = sent.replace(w, '*{}*'.format(w), 1)

    if with_verb:
        print(sent, file=f)
        with_verb_count += 1


print('>> {} sentences with VBZs and VBPs were saved to {}'.format(with_verb_count, out_filename))
f.close()
