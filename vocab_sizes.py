"""
Script to compute vocab size reduction due to phonetic transcription.
"""
import argparse
import os
import time
from tqdm import tqdm
import spacy
import json
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'parser', 'ner'])


def now(): return time.strftime("%Y-%m-%d_%H-%M-%S")


def main():
    parser = argparse.ArgumentParser(
        prog="vocab_sizes",
        description="Compute vocabulary sizes of a set of PAN20 datasets",
        add_help=True)
    parser.add_argument('--input',
                        '-i',
                        required=True,
                        help='Path to a directory of PAN20 datasets')
    parser.add_argument('--output',
                        '-o',
                        help='Name to tag the output file')
    args = parser.parse_args()

    directory = [d for d in os.scandir(args.input)]
    print(f'Found {len(directory)} unmasking results.')
    if args.output is None:
        output_filename = f'vocab_sizes_{now()}.json'
    else:
        output_filename = f'vocab_sizes_{now()}_{args.output}.json'

    type_dto = dict()

    # Go through PAN20 data files and add tokens to set
    for dir_entry in directory:
        paths = glob(os.path.join(dir_entry.path, '*.jsonl'))
        path_to_jsonl = [x for x in paths if not x.endswith('-truth.jsonl')][0]
        print(f'Counting types for {dir_entry.name}')
        types = set()
        with open(path_to_jsonl, 'r') as pan20_data_file:
            for pair_line in tqdm(pan20_data_file, desc='Pairs'):
                text = ' '.join(json.loads(pair_line)['pair'])
                doc = nlp(text)
                types.update([token.text.lower()
                              for token in doc
                              if (not token.is_punct and not token.like_num)])

        print(f'Types: {len(types)}\n')
        type_dto[dir_entry.name] = len(types)

    os.makedirs('data', exist_ok=True)
    with open(os.path.join('data', output_filename), 'w') as f:
        json.dump(type_dto, f)

    # Plot

    plt.bar(*zip(*sorted(type_dto.items(), key=lambda x: x[1], reverse=True)))
    plt.show()


if __name__ == '__main__':
    main()
