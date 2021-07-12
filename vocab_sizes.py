"""
Script to compute vocab size reduction due to phonetic transcription.
"""
import argparse
import os
import sys
import time
from tqdm import tqdm
import spacy
import json
from glob import glob
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

nlp_no_apostrophe_split = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
nlp_no_apostrophe_split.tokenizer.rules = {key: value for key, value in nlp_no_apostrophe_split.tokenizer.rules.items()
                                           if "'" not in key and "’" not in key and "‘" not in key}
suffixes = [suffix for suffix in nlp_no_apostrophe_split.Defaults.suffixes if suffix not in ["'s", "'S", '’s', '’S']]
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp_no_apostrophe_split.tokenizer.suffix_search = suffix_regex.search


def now(): return time.strftime('%Y-%m-%d_%H-%M-%S')


labels = {
    'verbatim': '$Verbatim$',
    'ipa': '$IPA$',
    'dolgo': '$Dolgo$',
    'asjp': '$ASJP$',
    'cv': '$CV$',
    'soundex': '$Soundex$',
    'refsoundex': '$RefSoundex$',
    'metaphone': '$Metaphone$',
    'punct': '$P$',
    'punct_lemma': '$PL$',
    'punct_lemma_stop': '$PLS$',
}


def plot(type_dto, name):
    # Math labels
    labeled_dto = {labels[key]: val for key, val in type_dto.items()}
    verbatim_types = type_dto['verbatim']

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(figsize=(5, 3.5))
    sorted_dto_list = sorted(labeled_dto.items(), key=lambda x: x[1], reverse=True)
    plt.bar(*zip(*sorted_dto_list))
    plt.xticks(rotation='vertical')
    plt.ylabel('Vocabulary size (types)')

    # Add scaling factor to first bar
    for i in range(0, len(sorted_dto_list)):
        vssf = round(sorted_dto_list[i][1] / verbatim_types, 4)  # vocab size scaling factor
        print(f'{sorted_dto_list[i][0]} & {vssf} ({sorted_dto_list[i][1]})')
        if vssf < 1:
            color = 'r'
            vssf = f' {abs(vssf)}'
        elif vssf > 1:
            color = 'g'
            vssf = f' {vssf}'
        else:
            color = 'k'
            vssf = f' {vssf}'
        if sorted_dto_list[i][0] == '$Verbatim$':
            continue
        plt.text(i, sorted_dto_list[i][1], vssf, ha='center', rotation='vertical', color=color)

    plt.savefig(f'{name[:-5]}.svg', format='svg', bbox_inches='tight')


def vocab_size(directory, output_filename):
    type_dto = dict()
    for dir_entry in directory:
        # if dir_entry.name not in ['verbatim', 'punct']:
        #     continue
        paths = glob(os.path.join(dir_entry.path, '*.jsonl'))
        path_to_jsonl = [x for x in paths if not x.endswith('-truth.jsonl')][0]
        print(f'Counting types for {dir_entry.name}')
        types = set()
        with open(path_to_jsonl, 'r') as pan20_data_file:
            for pair_line in tqdm(pan20_data_file, desc='Pairs'):
                text = ' '.join(json.loads(pair_line)['pair'])
                if dir_entry.name == 'verbatim':
                    doc = nlp_no_apostrophe_split(text)
                    types.update([token.text.lower()
                                  for token in doc
                                  if any(c.isalpha() for c in token.text)])
                else:
                    types.update([token.lower()
                                  for token in text.split(' ')
                                  if any(c.isalpha() for c in token)])

        print(f'Types: {len(types)}\n')
        type_dto[dir_entry.name] = len(types)
        # if dir_entry.name == 'verbatim':
        #     verbatim_types = types.copy()
        # if dir_entry.name == 'punct':
        #     punct_types = types.copy()

        with open(os.path.join('data', output_filename), 'w') as f:
            json.dump(type_dto, f)
    # print(verbatim_types.difference(punct_types))


def count_characters(directory, output_filename, count_doc_freq=False):
    char_dto = dict()
    c = Counter()
    for dir_entry in directory:
        paths = glob(os.path.join(dir_entry.path, '*.jsonl'))
        path_to_jsonl = [x for x in paths if not x.endswith('-truth.jsonl')][0]
        print(f'Counting characters for {dir_entry.name}')
        with open(path_to_jsonl, 'r') as pan20_data_file:
            for pair_line in tqdm(pan20_data_file, desc='Pairs'):
                text = ''.join(json.loads(pair_line)['pair'])
                if count_doc_freq:
                    c.update(set(text))
                else:
                    c.update(text)
        char_dto[dir_entry.name] = OrderedDict(c.most_common())
        c.clear()

        with open(os.path.join('data', output_filename), 'w') as f:
            json.dump(char_dto, f)


# gb_alphabet = ["e", "t", "a", "o", "n", "i", "h", "s", "r", "d", "l", "u", "c", "m", "w", "f", "g", "y", "p", ",", ".", "b", "\"",
#      "k", "v", "I", "-", "'", "T", "H", "A", "S", "W", "M", "?", "x", "B", "!", "C", ";", "j", "q", "N", "D", "z", "P",
#      "Y", "G", "O", "L", "E", "F", "R", "J", "K", ":", "V", "U", "0", "1", "(", ")", "Q", "2", "Z", "3", "5", "8", "9",
#      "4", "6", "7", "X", "&", "$", "]", "[", "/", "{", "}", "@", "%", "\n", "\ufeff", "=", "*", "+", "^", "\\"]
#
#
# with open('data/char_frequenciesvocab_sizes_2021-07-07_18-44-25_ff_proper.json', 'r') as f:
#     ff_char_freqs = json.load(f)
# ff_alphabet = list(ff_char_freqs['verbatim'].keys())
# print(ff_alphabet[:10])

# Coding:
# Document frequencies for characters
# How many types have characters not in gb_alphabet?
# How many pairs have types that have chars not in gb_alphabet?
# Then: What happens if we remove entire texts? Is this feasible? Depends on amount that is discarded.
# Also: What happens if we remove certain types, or even just certain characters?

# Decisions:
# Clean " in I"m -> Print all words with ", find out which are actually ok ("word") and which aren't (I"m)
# Clean other words / characters? No clue.

# Write:
# Either: This is the way it is, we kinda know why but won't change much.
# Or: Change things first, then re-evaluate.
# Anyways: Description of GB should be set.


def main():
    parser = argparse.ArgumentParser(
        prog="vocab_sizes",
        description="Compute vocabulary sizes of a set of PAN20 datasets",
        add_help=True)
    parser.add_argument('--input',
                        '-i',
                        help='Path to a directory of PAN20 datasets')
    parser.add_argument('--plot',
                        '-p',
                        help='Path to a vocab sizes .json file')
    parser.add_argument('--output',
                        '-o',
                        help='Name to tag the output file')
    parser.add_argument('--characters',
                        '-c',
                        help='Count character frequencies',
                        action='store_true')
    parser.add_argument('--docfreq',
                        '-d',
                        help='Count character document frequencies',
                        action='store_true')
    args = parser.parse_args()

    if args.plot is not None:
        with open(args.plot, 'r') as f:
            type_dto = json.load(f)
            plot(type_dto, args.plot)
        sys.exit(0)

    directory = [d for d in os.scandir(args.input)]
    print(f'Found {len(directory)} unmasking results.')
    if args.characters:
        output_filename = 'char_frequencies'
    elif args.docfreq:
        output_filename = 'char_doc_freq'
    else:
        output_filename = 'vocab_sizes'

    if args.output is None:
        output_filename = f'{output_filename}_{now()}.json'
    else:
        output_filename = f'{output_filename}_{now()}_{args.output}.json'

    os.makedirs('data', exist_ok=True)

    if args.characters:
        count_characters(directory, output_filename)
        sys.exit(0)
    if args.docfreq:
        count_characters(directory, output_filename, True)
        sys.exit(0)
    vocab_size(directory, output_filename)
    # Go through PAN20 data files and add tokens to set


if __name__ == '__main__':
    main()
