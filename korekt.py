#!/usr/bin/env python3
# Copyright Mycroft AI, Inc. 2017. All Rights Reserved.
import json
import sys
import numpy as np

from fann2 import libfann
from os.path import isfile


def print_usage():
    print('Usage:', sys.argv[0], 'TEXT_FILE')
    print('\tThe text file should contain a list of words in the dictionary')
    exit(1)

if len(sys.argv) != 2 or not isfile(sys.argv[1]):
    print_usage()

word_to_id, id_to_word, char_to_id = {}, {}, {}
with open(sys.argv[1]) as f:
    text = ' '.join(f.readlines()).lower()

words = set(text.split())
for word in words:
    word_to_id[word] = len(word_to_id)
    id_to_word[len(id_to_word)] = word

for char in set(text):
    char_to_id[char] = len(char_to_id)

in_len = len(char_to_id)
out_len = len(id_to_word)

print('Saving Ids...')
ids = {
    'word_to_id': word_to_id,
    'id_to_word': id_to_word,
    'char_to_id': char_to_id
}

prefix = sys.argv[1].replace('.txt', '')
with open(prefix + '.ids', 'w') as f:
    json.dump(ids, f, indent=4)


def vectorize_in(word):
    vec = np.zeros((in_len,))
    for char in word:
        vec[char_to_id[char]] = 1.0
    return vec


def vectorize_out(word):
    vec = np.zeros((out_len,))
    vec[word_to_id[word]] = 1.0
    return vec

print('Generating sets...')
sets = {}
for word in words:
    st = frozenset(word)
    if st in sets:
        sets[st].append(word)
    else:
        sets[st] = [word]

print('Creating network...')
nn = libfann.neural_net()
nn.create_standard_array([in_len, out_len])
nn.set_train_stop_function(libfann.STOPFUNC_BIT)
nn.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
nn.set_learning_rate(0.5)
data = libfann.training_data()

set_items = list(sets.items())
step_samples = min(2000, len(set_items))
num_steps = round(len(set_items) / step_samples)

for i in range(step_samples, len(set_items) + 1, step_samples):
    print('Training on mini-batch', str(round(i / step_samples)), '/', str(num_steps) + '...')

    inputs, outputs = [], []
    for inp_set, words in set_items[i - step_samples:i]:
        inputs.append(vectorize_in(inp_set))
        outputs.append(np.maximum.reduce([vectorize_out(w) for w in words]))
    data.set_train_data(inputs, outputs)
    nn.train_on_data(data, 1000, 1, 0)

print('Saving Network...')
nn.save(prefix + '.net')

inp = ''
while inp != 'q':
    if len(inp) != 0:
        result = nn.run(vectorize_in(inp))
        conf = max(result)
        word = id_to_word[result.index(conf)]
        print(word)
        print('Confidence:', round(conf, 2))
    inp = input('> ')
