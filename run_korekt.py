#!/usr/bin/env python3
# Copyright Mycroft AI, Inc. 2017. All Rights Reserved.
import json
import sys
import numpy as np
from fann2 import libfann as fann
from os.path import isfile

if len(sys.argv) == 1 or not isfile(sys.argv[1] + '.net'):
    print('Usage:', sys.argv[0], 'MODEL_PREFIX [INPUT_STRING] [-d]')
    print('\t(Ex. with model.net and model.ids, the prefix is \'model\'')
    exit(1)

nn = fann.neural_net()
nn.create_from_file(sys.argv[1] + '.net')

with open(sys.argv[1] + '.ids') as f:
    ids = json.load(f)
id_to_word = ids['id_to_word']
char_to_id = ids['char_to_id']


def vectorize_in(word):
    vec = np.zeros(len(char_to_id))
    for char in word:
        if char in char_to_id:
            vec[char_to_id[char]] = 1.0
    return vec


def fix_spelling(string, debug):
    words = string.split()
    new_words = []
    for word in words:
        out = nn.run(vectorize_in(word))
        conf = max(out)
        if debug:
            print('\t' + word + ':', round(conf, 2))
        out_word = id_to_word[str(out.index(conf))]
        new_words.append(out_word if conf > 0.4 else word)
    return ' '.join(new_words)

debug = False
if '-d' in sys.argv:
    sys.argv.remove('-d')
    debug = True

if len(sys.argv) == 2:
    inp = ''
    while inp != 'q':
        if len(inp) != 0:
            print(fix_spelling(inp, debug))
        inp = input('> ')
else:
    query = ' '.join(sys.argv[2:])
    print(fix_spelling(query, debug))

