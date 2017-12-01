#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:59:47 2017

@author: mario
"""

import numpy as np
import re

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


regex_ed_1 = r'\bha(ve|s|d) ((\w|\d)+ ){1}\w+ed\b'
regex_ed_2 = r'\bha(ve|s|d) ((\w|\d)+ ){2}\w+ed\b'
regex_ed_3 = r'\bha(ve|s|d) ((\w|\d)+ ){3}\w+ed\b'
regex_ed_4 = r'\bha(ve|s|d) ((\w|\d)+ ){4}\w+ed\b'
regex_ed_5 = r'\bha(ve|s|d) ((\w|\d)+ ){5}\w+ed\b'

regex_ing_1 = r'\b(are|am|is) ((\w|\d)+ ){1}\w+(ed|ing)\b'
regex_ing_2 = r'\b(are|am|is) ((\w|\d)+ ){2}\w+(ed|ing)\b'
regex_ing_3 = r'\b(are|am|is) ((\w|\d)+ ){3}\w+(ed|ing)\b'
regex_ing_4 = r'\b(are|am|is) ((\w|\d)+ ){4}\w+(ed|ing)\b'
regex_ing_5 = r'\b(are|am|is) ((\w|\d)+ ){5}\w+(ed|ing)\b'

regex_irreg_1 = r'\bha(ve|s|d) ((\w|\d)+ ){1}(read|made|gone|done|come|got|bought|sought|been|seen|thought)\b'
regex_irreg_2 = r'\bha(ve|s|d) ((\w|\d)+ ){2}(read|made|gone|done|come|got|bought|sought|been|seen|thought)\b'
regex_irreg_3 = r'\bha(ve|s|d) ((\w|\d)+ ){3}(read|made|gone|done|come|got|bought|sought|been|seen|thought)\b'
regex_irreg_4 = r'\bha(ve|s|d) ((\w|\d)+ ){4}(read|made|gone|done|come|got|bought|sought|been|seen|thought)\b'
regex_irreg_5 = r'\bha(ve|s|d) ((\w|\d)+ ){5}(read|made|gone|done|come|got|bought|sought|been|seen|thought)\b'


regs = [regex_ed_1, regex_ed_2, regex_ed_3, regex_ed_4, regex_ed_5,
        regex_ing_1, regex_ing_2, regex_ing_3, regex_ing_4, regex_ing_5,
        regex_irreg_1, regex_irreg_2, regex_irreg_3, regex_irreg_4,
        regex_irreg_5]


for reg in regs:
    filename = namestr(reg, globals())[0]
    if len(filename) < 5:
        filename = namestr(reg, globals())[1]

    with open('./sentences/{}.txt'.format(filename), 'w') as f_out:
        for sent in corpus:
            if re.findall(reg, sent):
                f_out.write('{}\n'.format(sent))
