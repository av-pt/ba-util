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

import matplotlib.pyplot as plt

nlp_no_apostrophe_split = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
nlp_no_apostrophe_split.tokenizer.rules = {key: value for key, value in nlp_no_apostrophe_split.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
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
        if sorted_dto_list[i][0] == '$Verbatim$':
            continue
        plt.text(i, sorted_dto_list[i][1], vssf, ha='center', rotation='vertical', color=color)


    plt.savefig(f'{name[:-5]}.svg', format='svg', bbox_inches='tight')


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
    args = parser.parse_args()

    if args.plot is not None:
        with open(args.plot, 'r') as f:
            type_dto = json.load(f)
            plot(type_dto, args.plot)
            sys.exit(0)

    directory = [d for d in os.scandir(args.input)]
    print(f'Found {len(directory)} unmasking results.')
    if args.output is None:
        output_filename = f'vocab_sizes_{now()}.json'
    else:
        output_filename = f'vocab_sizes_{now()}_{args.output}.json'

    type_dto = dict()

    os.makedirs('data', exist_ok=True)

    # Go through PAN20 data files and add tokens to set
    for dir_entry in directory:
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

        with open(os.path.join('data', output_filename), 'w') as f:
            json.dump(type_dto, f)


if __name__ == '__main__':
    main()
