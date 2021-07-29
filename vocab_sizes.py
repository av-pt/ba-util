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
    'ipa_4grams': '$IPA$ $4$-$grams$',
    'dolgo_4grams': '$Dolgo$ $4$-$grams$',
    'asjp_4grams': '$ASJP$ $4$-$grams$',
    'cv_4grams': '$CV$ $4$-$grams$',
    'punct_4grams': '$P$ $4$-$grams$',
    'verbatimoriginal': '$Verbatim$ $(orig.)$',
}

def split_if_gb(type_dto, name):
    # If 'ipa_4grams' is in the results, it is the GB dataset and they
    # plotted separately
    if 'ipa_4grams' in type_dto.keys():
        pt_type_dto = {k: v for k, v in type_dto.items() if not k.endswith('_4grams')}
        ngram_type_dto = {k: v for k, v in type_dto.items() if k.endswith('_4grams') or k == 'verbatim'}
        plot(pt_type_dto, f'{name[:-5]}_pt.svg')
        plot(ngram_type_dto, f'{name[:-5]}_ngram.svg')
    else:
        plot(type_dto, f'{name[:-5]}_pt.svg')



def plot(type_dto, name):
    """
    Plot vocab sizes for the transcriptions of the datasets.
    """
    # Math labels
    labeled_dto = {labels[key]: val for key, val in type_dto.items()}
    verbatim_types = type_dto['verbatim']

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    fig = plt.figure(figsize=(5, 3.5))
    sorted_dto_list = sorted(labeled_dto.items(), key=lambda x: (x[1], x[0]), reverse=True)
    plt.bar(*zip(*sorted_dto_list), color='#185B8C')  # Dark blue
    plt.xticks(rotation='vertical')
    plt.ylabel('Vocabulary size (types)')

    # Add scaling factor to first bar
    for i in range(0, len(sorted_dto_list)):
        vssf = round(sorted_dto_list[i][1] / verbatim_types, 4)  # vocab size scaling factor
        # print(f'{sorted_dto_list[i][0]} & {vssf} ({sorted_dto_list[i][1]})')
        print(f'{sorted_dto_list[i][0]} & {vssf} ({sorted_dto_list[i][1]})')
        if vssf < 1:
            color = '#B30000'  # Dark red
            vssf = f'  {abs(vssf)}'
        elif vssf > 1:
            color = '#006600'  # Dark green
            vssf = f'  {vssf}'
        else:
            color = 'k'
            vssf = f'  {vssf}'
        if sorted_dto_list[i][0] == '$Verbatim$':
            continue
        # plt.text(i, plt.gca().get_ylim()[1] + 100, vssf, ha='center', rotation='vertical', color=color)
        # plt.text(i, sorted_dto_list[i][1], vssf, ha='center', rotation='vertical', color=color)

    plt.savefig(name, format='svg', bbox_inches='tight')
    plt.savefig(f'{name[:-4]}.pdf', format='pdf', bbox_inches='tight')


def vocab_size(directory, output_filename):
    """
    Determines the vocab sizes for the transcriptions of the datasets.
    """
    type_dto = dict()

    for dir_entry in directory:
        # Comment in, if only certain transcriptions should be examined
        # if dir_entry.name in ["asjp", "cv", "dolgo", "punct_lemma", "ipa", "metaphone", "refsoundex", "punct"]:
        #     continue
        paths = glob(os.path.join(dir_entry.path, '*.jsonl'))
        path_to_jsonl = [x for x in paths if not x.endswith('-truth.jsonl')][0]
        print(f'Counting types for {dir_entry.name}')
        types = set()
        with open(path_to_jsonl, 'r') as pan20_data_file:
            for pair_line in tqdm(pan20_data_file, desc='Pairs'):
                text = ' '.join(json.loads(pair_line)['pair'])
                if dir_entry.name in ['verbatim', 'verbatimoriginal']:
                    doc = nlp_no_apostrophe_split(text)
                    newtypes = list(set([token.text.lower()
                                for token in doc
                                if any(c.isalpha() for c in token.text)]))
                    types.update(newtypes)
                else:
                    newtypes = list(set([token.lower()
                                for token in text.split(' ')
                                if any(c.isalpha() for c in token)]))
                    types.update(newtypes)
        print(f'Types: {len(types)}\n')
        type_dto[dir_entry.name] = len(types)

        with open(os.path.join('data', output_filename), 'w') as f:
            json.dump(type_dto, f)


def vocab_list(directory, output_tag):
    """
    Generate vocabulary sets / bag of words for each sample in verbatim,
    verbatimoriginal and IPA transcriptions.
    Verbatim texts are simply split on spaces (split(' ')).
    The texts of a sample pair are NOT split up before creating the
    vocab list. 2 texts -> 1 vocab list.
    """
    timestamp = now()



    # verbatim_dto = dict()
    # verbatimoriginal_dto = dict()
    # ipa_dto = dict()
    # punct_dto = dict()

    # verbatim_vocab_list = []
    # verbatimoriginal_vocab_list = []
    # ipa_vocab_list = []
    # punct_vocab_list = []
    out_folder = os.path.join('data', f'vocab_lists_{output_tag}_{timestamp}')
    os.makedirs(out_folder, exist_ok=True)

    for dir_entry in directory:
        vocab_list = []
        if dir_entry.name in ['verbatim', 'verbatimoriginal', 'ipa', 'punct']:
            continue
        paths = glob(os.path.join(dir_entry.path, '*.jsonl'))
        path_to_jsonl = [x for x in paths if not x.endswith('-truth.jsonl')][0]
        print(f'Creating vocab lists for {dir_entry.name}')
        types = set()
        with open(path_to_jsonl, 'r') as pan20_data_file:
            for pair_line in tqdm(pan20_data_file, desc='Pairs'):
                text = ' '.join(json.loads(pair_line)['pair'])
                # newtypes = None
                if dir_entry.name in ['verbatimoriginal', 'verbatim']:
                    doc = nlp_no_apostrophe_split(text)
                    newtypes = list(set([token.text.lower()
                                for token in doc
                                if any(c.isalpha() for c in token.text)]))
                    types.update(newtypes)
                else:
                    newtypes = list(set([token.lower()
                                         for token in text.split(' ')
                                         if any(c.isalpha() for c in token)]))
                    types.update(newtypes)

                # if dir_entry.name == 'verbatim':
                #     verbatim_vocab_list.append(newtypes)
                # if dir_entry.name == 'verbatimoriginal':
                #     verbatimoriginal_vocab_list.append(newtypes)
                # if dir_entry.name == 'ipa':
                #     ipa_vocab_list.append(newtypes)
                # if dir_entry.name == 'punct':
                #     punct_vocab_list.append(newtypes)
                vocab_list.append(newtypes)

        # if dir_entry.name == 'verbatim':
        with open(os.path.join(out_folder, f'{dir_entry.name}.json'), 'w') as f:
            dto = dict()
            dto['data'] = vocab_list
            json.dump(dto, f)
        dto = None
        # if dir_entry.name == 'verbatim':
        #     with open(os.path.join(out_folder, 'verbatim.json'), 'w') as f:
        #         verbatim_dto['data'] = verbatim_vocab_list
        #         json.dump(verbatim_dto, f)
        #     verbatim_dto = None
        # if dir_entry.name == 'verbatimoriginal':
        #     with open(os.path.join(out_folder, 'verbatimoriginal.json'), 'w') as f:
        #         verbatimoriginal_dto['data'] = verbatimoriginal_vocab_list
        #         json.dump(verbatimoriginal_dto, f)
        #     verbatimoriginal_dto = None
        # if dir_entry.name == 'ipa':
        #     with open(os.path.join(out_folder, 'ipa.json'), 'w') as f:
        #         ipa_dto['data'] = ipa_vocab_list
        #         json.dump(ipa_dto, f)
        #     ipa_dto = None
        # if dir_entry.name == 'punct':
        #     with open(os.path.join(out_folder, 'punct.json'), 'w') as f:
        #         punct_dto['data'] = punct_vocab_list
        #         json.dump(punct_dto, f)
        #     punct_dto = None


def count_characters(directory, output_filename):
    """
    Count char frequencies, char document frequencies and ASCII char
    percentage.
    """
    char_dto = dict()
    char_dto['Char Frequencies'] = dict()
    char_dto['Char Document Frequencies'] = dict()
    char_dto['Pairs of only ASCII chars'] = dict()
    char_dto['Avg. percentage of ASCII chars over all chars'] = dict()
    c_char = Counter()
    c_docfreq = Counter()
    dataset_character_length = 0
    valid_texts = 0
    valid_char_count = 0
    for dir_entry in directory:
        paths = glob(os.path.join(dir_entry.path, '*.jsonl'))
        path_to_jsonl = [x for x in paths if not x.endswith('-truth.jsonl')][0]
        print(f'Counting characters for {dir_entry.name}')
        with open(path_to_jsonl, 'r') as pan20_data_file:
            for pair_line in tqdm(pan20_data_file, desc='Pairs'):
                text = ''.join(json.loads(pair_line)['pair'])
                c_char.update(text)
                c_docfreq.update(set(text))
                dataset_character_length += len(text)

                # Check if in ASCII range
                v = [char for char in text if ord(char) < 128]
                valid_char_count += len(v)
                if len(text) == len(v):
                    valid_texts += 1

        char_dto['Char Frequencies'][dir_entry.name] = OrderedDict(c_char.most_common())
        char_dto['Char Document Frequencies'][dir_entry.name] = OrderedDict(c_docfreq.most_common())
        char_dto['Pairs of only ASCII chars'][dir_entry.name] = valid_texts
        char_dto['Avg. percentage of ASCII chars over all chars'][
            dir_entry.name] = valid_char_count / dataset_character_length
        c_char.clear()
        c_docfreq.clear()
        valid_texts = 0
        valid_char_count = 0
        dataset_character_length = 0

        with open(os.path.join('data', output_filename), 'w') as f:
            json.dump(char_dto, f, indent=2)


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
                        help='Count character frequencies, character document'
                             'frequencies and ASCII to non-ASCII ratio',
                        action='store_true')
    parser.add_argument('--vocablists',
                        '-v',
                        help='Create vocab lists per sample per transcription',
                        action='store_true')
    args = parser.parse_args()

    # If vocab sizes are given, generate plot: vocab size vs. measure
    if args.plot is not None:
        with open(args.plot, 'r') as f:
            type_dto = json.load(f)
            split_if_gb(type_dto, args.plot)
        sys.exit(0)

    # Set correct ouput tag
    if args.characters:
        output_filename = 'char_frequencies'
    elif args.vocablists:
        output_filename = 'vocab_lists'
    else:
        output_filename = 'vocab_sizes'

    # Add timestamp and user entered tag
    if args.output is None:
        output_filename = f'{output_filename}_{now()}.json'
    else:
        output_filename = f'{output_filename}_{now()}_{args.output}.json'

    os.makedirs('data', exist_ok=True)

    directory = [d for d in os.scandir(args.input)]
    print(f'Found {len(directory)} datasets: {[dir_entry.name for dir_entry in directory]}')

    # Execute the specified function
    if args.characters:
        count_characters(directory, output_filename)
        sys.exit(0)
    elif args.vocablists:
        vocab_list(directory, args.output)
        sys.exit(0)
    vocab_size(directory, output_filename)


if __name__ == '__main__':
    main()
