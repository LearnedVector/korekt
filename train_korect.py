#!/usr/bin/env python3
# Copyright Mycroft AI, Inc. 2017. All Rights Reserved.
import json
import sys
import numpy as np

from fann2 import libfann as fann
from os.path import isfile

if len(sys.argv) != 2 or not isfile(sys.argv[1]):
    print('Usage:', sys.argv[0], 'JSON_FILE')
    print('\tThe json file should contain a list of mispellings of words')
    print('\t  Ex. "mispelled": "tje", "correct": "the"')
    exit(1)

word_to_id, id_to_word, char_to_id = {}, {}, {}
with open(sys.argv[1]) as f:
    data = json.load(f)

# Load data into sets
spelling_sets = {}
wrong_sets = set()
for sample in data:
    spelling = sample['mispelled']
    word = sample['correct']

    # Register words
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)
        id_to_word[len(id_to_word)] = word

    # Register characters
    for char in spelling:
        if char not in char_to_id:
            char_to_id[char] = len(char_to_id)

    # Add spellings
    st = frozenset(spelling)
    if st in spelling_sets:
        spelling_sets[st].append(word)
    else:
        spelling_sets[st] = [word]

    # Inhibit subportions of spellings
    items = list(st)
    for i in range(0, len(st)):
        for j in range(i + 1, len(st) + 1):
            wrong_sets.add(frozenset(items[i:j]))

in_len = len(char_to_id)
out_len = len(id_to_word)


def vectorize_in(word):
    vec = np.zeros(in_len)
    for char in word:
        vec[char_to_id[char]] = 1.0
    return vec


def vectorize_out(word):
    vec = np.zeros(out_len)
    vec[word_to_id[word]] = 1.0
    return vec

print('Generating training data...')

# Generate spelling data
inputs, outputs = [], []
for letters, words in spelling_sets.items():
    inputs.append(vectorize_in(letters))
    outputs.append(np.maximum.reduce([vectorize_out(w) for w in words]))

# Generate inhibition data
outv = np.zeros(out_len)
for letters in wrong_sets:
    if letters not in spelling_sets:
        inputs.append(vectorize_in(letters))
        outputs.append(outv)


print('Creating network...')
nn = fann.neural_net()
nn.create_standard_array([in_len, out_len])
nn.set_train_stop_function(fann.STOPFUNC_BIT)
nn.set_training_algorithm(fann.TRAIN_INCREMENTAL)
nn.set_activation_steepness_output(0.001)
nn.set_learning_rate(1000)

print('Training...')
data = fann.training_data()
data.set_train_data(inputs, outputs)
nn.train_on_data(data, 10000, 0, 0)

print('Saving...')
prefix = sys.argv[1].replace('.json', '')
nn.save(prefix + '.net')
with open(prefix + '.ids', 'w') as f:
    json.dump(indent=4, fp=f, obj={
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'char_to_id': char_to_id
    })
