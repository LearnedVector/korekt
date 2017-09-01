#!/usr/bin/env python3
# Copyright Mycroft AI, Inc. 2017. All Rights Reserved.
import json
import sys
import numpy as np

from fann2 import libfann
from os.path import isfile


def resolve_conflicts(inputs, outputs):
    """
    Checks for duplicate inputs and if there are any,
    remove one and set the output to the max of the two outputs
    Args:
        inputs (list<list<float>>): Array of input vectors
        outputs (list<list<float>>): Array of output vectors
    Returns:
        tuple<inputs, outputs>: The modified inputs and outputs
    """
    new_in, new_out = [], []
    for i in range(len(inputs)):
        found_duplicate = False
        for j in range(i + 1, len(inputs)):
            if np.array_equal(inputs[i], inputs[j]):
                found_duplicate = True
                outputs[j] =  np.maximum.reduce([outputs[i], outputs[j]])
        if not found_duplicate:
            new_in.append(inputs[i])
            new_out.append(outputs[i])
    return new_in, new_out


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

print(out_len)
def vectorize_in(word):
    vec = np.zeros((in_len,))
    for char in word:
        vec[char_to_id[char]] = 1.0
    return vec


def vectorize_out(word):
    vec = np.zeros((out_len,))
    vec[word_to_id[word]] = 1.0
    return vec

inputs = [vectorize_in(word) for word in words]
outputs = [vectorize_out(word) for word in words]

inputs, outputs = resolve_conflicts(inputs, outputs)

data = libfann.training_data()
data.set_train_data(inputs, outputs)

prefix = sys.argv[1].replace('.txt', '')

nn = libfann.neural_net()
nn.create_standard_array([in_len, out_len])
nn.set_train_stop_function(libfann.STOPFUNC_BIT)
nn.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
nn.set_learning_rate(1000)
nn.train_on_data(data, 100000, 1, 0)
nn.save(prefix + '.net')

ids = {
    'word_to_id': word_to_id,
    'id_to_word': id_to_word,
    'char_to_id': char_to_id
}

with open(prefix + '.ids', 'w') as f:
    json.dump(ids, f, indent=4)

inp = ''
while inp != 'q':
    if len(inp) != 0:
        result = nn.run(vectorize_in(inp))
        conf = max(result)
        word = id_to_word[result.index(conf)]
        print(word)
        print('Confidence:', round(conf, 2))
    inp = input('> ')
