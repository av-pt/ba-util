"""
Playground for various dataset visualizations.
"""

import json
import os
import string
import sys
from collections import Counter
import random
import time
from string import punctuation

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import aspell


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


# with open('data/char_frequenciesvocab_sizes_2021-07-07_18-44-25_ff_proper.json', 'r') as f:
#     ff_char_freqs = json.load(f)
# ff_alphabet = list(ff_char_freqs['verbatim'].keys())
# print(ff_alphabet[:10])


def plot_cum_vocab_sizes():
    """
    Plot a list of accumulated vocab sizes.
    """
    # with open('data/verbatim_cum_vocab_size_gb_2021-07-15_21-33-25.json', 'r') as f:
    with open('data/verbatim_cum_vocab_size_ff_2021-07-15_21-56-28.json', 'r') as f:
        verbatim_cum_vocab_size = json.load(f)['data'][:262]
    # with open('data/ipa_cum_vocab_size_gb_2021-07-15_21-33-25.json', 'r') as f:
    with open('data/ipa_cum_vocab_size_ff_2021-07-15_21-56-28.json', 'r') as f:
        ipa_cum_vocab_size = json.load(f)['data'][:262]

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(figsize=(5, 3.5))
    plt.plot(verbatim_cum_vocab_size, label='$Verbatim$')
    plt.plot(ipa_cum_vocab_size, label='$IPA$')
    plt.ylabel('Cumulative Vocab Size')
    plt.legend()
    plt.savefig(f'data/cum_vocab_size_test_ff.svg', format='svg', bbox_inches='tight')

def plot_vocab_lists():
    """
    Given a list of sets of tokens, accumulate the total vocabulary and
    plot Heaps' Law.
    """

    # FF indices
    # There are 27834 same-author pairs and 24767 different-author pairs
    # With long and short texts removed: 27178 same-author, 23771 diff-author, 50949 total
    indices = [x for x in range(50949)]

    # order = 'inorder'
    # order = 'inreverseorder'
    # order = 'shuffled'
    # order = 'inorder_onlysame'
    # order = 'inorder_onlydiff'
    order = None

    if order == 'inorder':
        vertical_line_value = 27178
        text_left = 'same author'
        text_right = 'diff. author'
    elif order == 'inreverseorder':
        indices.reverse()
        vertical_line_value = 23771
        text_left = 'diff. author'
        text_right = 'same author'
    elif order == 'shuffled':
        random.shuffle(indices)
    elif order == 'inorder_onlysame':
        indices = indices[:27178]
    elif order == 'inorder_onlydiff':
        indices = indices[-23771:]

    # GB indices in order (first: same, last: different)
    # indices_same = []
    # indices_diff = []
    # with open('../unmasking/NAACL-19/corpus/pan20/transcribed/verbatim/gb-truth.jsonl', 'r') as f:
    #     same = [json.loads(line)['same'] for line in f]
    # for i in range(len(same)):
    #     if same[i]:
    #         indices_same.append(i)
    #     else:
    #         indices_diff.append(i)
    # indices = indices_diff + indices_same
    # print(indices)
    indices = range(262)
    order = 'indefaultorder'

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(figsize=(5, 3.5))
    # plt.ylim(-40000, 600000)

    # directory = [d for d in os.scandir('data/vocab_lists_ff_2021-07-27_15-32-00')]
    directory = [d for d in os.scandir('data/vocab_lists_gb_ngrams_2021-07-29_22-08-01')]
    for dir_entry in directory:
        # if dir_entry.name in ['verbatim_spacetokenized.json', 'verbatimoriginal_spacetokenized.json']:
        #     continue
        # if dir_entry.name not in ['ipa.json', 'verbatim.json', 'dolgo.json', 'asjp.json']:
        #     continue
        types = set()
        cum_vocab_sizes = [0]
        print(f'Loading {dir_entry.name} data from {dir_entry.path}...')
        with open(dir_entry.path, 'r') as f:
            vocab_lists = json.load(f)['data']  # [:1000]
            for i in tqdm(indices):
                types.update(vocab_lists[i])
                cum_vocab_sizes.append(len(types))
            vocab_list_verbatim = None
            plt.plot(cum_vocab_sizes, label=labels[dir_entry.name[:-5]])

    if order in ['inorder', 'inreverseorder']:
        plt.axvline(x=vertical_line_value, color='black', linestyle='--', label=None, linewidth=1)
        plt.text(vertical_line_value + (plt.gca().get_xlim()[1] - vertical_line_value) / 2, 0, text_right, horizontalalignment='center')
        plt.text(vertical_line_value / 2, 0, text_left, horizontalalignment='center')

    plt.xlabel('Text pairs processed')
    plt.ylabel('Vocabulary size')
    plt.legend()
    plt.savefig(f'data/cum_vocab_size_gb_{order}_ngrams.pdf', format='pdf', bbox_inches='tight')


def individual_vocab_size():
    """
    Given a dataset, plot the individual vocabulary sizes of the texts.
    """
    # indices = [x for x in range(len(vocab_list_verbatim))]  # Indices of data points
    indices = [x for x in range(52583)]
    # indices = [x for x in range(262)]
    #indices.reverse()
    #indices = [x for x in range(1000)]
    #random.shuffle(indices)

    verbatim_types = set()
    ipa_types = set()

    # Holding individual vocab sizes
    vocab_sizes_verbatim = []
    vocab_sizes_ipa = []

    print('Loading Verbatim data...')
    # with open('data/vocab_list_verbatim_gb_2021-07-16_17-58-49.json', 'r') as f:
    with open('data/vocab_list_verbatim_ff_2021-07-16_18-15-43.json', 'r') as f:
        vocab_list_verbatim = json.load(f)['data']  # [:1000]
        for i in tqdm(indices):
            vocab_sizes_verbatim.append(len(vocab_list_verbatim[i]))
        vocab_list_verbatim = None
    # print('Loading IPA data...')
    # # with open('data/vocab_list_ipa_gb_2021-07-16_17-58-49.json', 'r') as f:
    # with open('data/vocab_list_ipa_ff_2021-07-16_18-07-11.json', 'r') as f:
    #     vocab_list_ipa = json.load(f)['data']  # [:1000]
    #     for i in tqdm(indices):
    #         vocab_sizes_ipa.append(len(vocab_list_ipa[i]))
    #     vocab_list_ipa = None

    moving_avg = [sum(x) / len(x) for x in tqdm(zip(*[vocab_sizes_verbatim[i:] for i in range(2000)]))]

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(figsize=(5, 3.5))
    plt.scatter(range(len(vocab_sizes_verbatim)), vocab_sizes_verbatim, label='$Verbatim$', s=1)
    # plt.scatter(range(len(vocab_sizes_verbatim)), vocab_sizes_ipa, label='$IPA$', s=1)
    plt.plot(moving_avg, label='Moving avg.', color='black', linewidth=1)
    plt.ylabel('Cumulative Vocab Size')
    plt.legend()
    plt.savefig(f'data/individual_vocab_size_ff_inorder.svg', format='svg', bbox_inches='tight')


def individual_text_length():
    """
    Given a dataset, plot the individual text lengths of the texts.
    """
    # indices = [x for x in range(len(vocab_list_verbatim))]  # Indices of data points
    indices = [x for x in range(52583)]
    # indices = [x for x in range(262)]
    #indices.reverse()
    #indices = [x for x in range(1000)]
    #random.shuffle(indices)

    # Holding individual text pair lengths in characters
    text_length_chars = []

    print('Loading Verbatim data...')
    # with open('data/vocab_list_verbatim_gb_2021-07-16_17-58-49.json', 'r') as f:
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/verbatim/verbatim_pan20-authorship-verification-training-small.jsonl', 'r') as f:
        for line in tqdm(f):
            pair = json.loads(line)['pair']
            length = len(pair[0]) + len(pair[1])
            text_length_chars.append(length)

    text_length_chars = [x for x in text_length_chars if x <= 100000]

    moving_avg = [sum(x)/len(x) for x in tqdm(zip(*[text_length_chars[i:] for i in range(2000)]))]



    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(figsize=(5, 3.5))
    plt.scatter(range(len(text_length_chars)), text_length_chars, label='$Verbatim$', s=1)
    plt.plot(moving_avg, label='Moving avg.', color='black', linewidth=1)
    # plt.scatter(range(len(text_length_chars)), vocab_sizes_ipa, label='$IPA$', s=1)
    plt.xlabel('Pair number')
    plt.ylabel('Pair length in characters')
    plt.legend()
    plt.savefig(f'data/individual_text_length_ff_inorder.svg', format='svg', bbox_inches='tight')


def count_authors():
    """
    Counts authors of FF and GB datasets wrt same-author and different-author classes.
    """
    # FF
    ff_truth = []
    same_authors_ff = set()
    diff_authors_ff = set()
    same_author_texts_ff = 0
    diff_author_texts_ff = 0
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/verbatim/pan20-authorship-verification-training-small-truth.jsonl', 'r') as f:
        for line in f:
            ff_truth.append(json.loads(line))
    for t in ff_truth:
        if t['same']:
            same_authors_ff.update(t['authors'])
            same_author_texts_ff += 2  # 2 as there are 2 texts in a pair
        else:
            diff_authors_ff.update(t['authors'])
            diff_author_texts_ff += 2
    same_text_per_author_ff = same_author_texts_ff / len(same_authors_ff)
    diff_text_per_author_ff = diff_author_texts_ff / len(diff_authors_ff)

    # GB
    gb_truth = []
    gb_meta = []
    same_authors_gb = set()
    diff_authors_gb = set()
    same_author_texts_gb = 0
    diff_author_texts_gb = 0
    with open('../unmasking/NAACL-19/corpus/pan20/transcribed/verbatim/gb-truth.jsonl', 'r') as f:
        for line in f:
            gb_truth.append(json.loads(line))
    with open('../unmasking/NAACL-19/corpus/pan20/transcribed/verbatim/verbatim_gb.jsonl', 'r') as f:
        for line in f:
            gb_meta.append(json.loads(line))
    for t in gb_truth:
        meta = [pair['meta'] for pair in gb_meta if pair['id'] == t['id']][0]
        if t['same']:
            same_authors_gb.update(meta['known_author'])
            same_authors_gb.update(meta['unknown_author'])
            same_author_texts_gb += 2
        else:
            diff_authors_gb.update(meta['known_author'])
            diff_authors_gb.update(meta['unknown_author'])
            diff_author_texts_gb += 2
    same_text_per_author_gb = same_author_texts_gb / len(same_authors_gb)
    diff_text_per_author_gb = diff_author_texts_gb / len(diff_authors_gb)


    # Persist data
    dto = dict()
    dto['ff'] = dict()
    dto['ff']['same_authors'] = len(same_authors_ff)
    dto['ff']['diff_authors'] = len(diff_authors_ff)
    dto['ff']['same_diff_intersection'] = len(same_authors_ff.intersection(diff_authors_ff))
    dto['ff']['same_text_per_author'] = same_text_per_author_ff
    dto['ff']['diff_text_per_author'] = diff_text_per_author_ff
    dto['gb'] = dict()
    dto['gb']['same_authors'] = len(same_authors_gb)
    dto['gb']['diff_authors'] = len(diff_authors_gb)
    dto['gb']['same_diff_intersection'] = len(same_authors_gb.intersection(diff_authors_gb))
    dto['gb']['same_text_per_author'] = same_text_per_author_gb
    dto['gb']['diff_text_per_author'] = diff_text_per_author_gb
    with open('data/authors.json', 'w') as f:
        json.dump(dto, f)

def char_frequencies():
    """
    Given a dataset, determine the percentage of ASCII characters in the
    same-author and different-author parts respectively.
    """
    same_counter = Counter()
    diff_counter = Counter()
    c = 0
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/verbatim/verbatim_pan20-authorship-verification-training-small.jsonl', 'r') as f:
        for line in tqdm(f):
            if c < 27834:
                same_counter.update(''.join(json.loads(line)['pair']))
            else:
                diff_counter.update(''.join(json.loads(line)['pair']))
            c += 1

    # Persist data
    dto = dict()
    dto['same_ascii_percentage'] = sum([freq for char, freq in same_counter.items() if char in string.ascii_letters]) / sum(same_counter.values())
    dto['same_nonascii_percentage'] = sum([freq for char, freq in same_counter.items() if char not in string.ascii_letters]) / sum(same_counter.values())
    dto['diff_ascii_percentage'] = sum([freq for char, freq in diff_counter.items() if char in string.ascii_letters]) / sum(diff_counter.values())
    dto['diff_nonascii_percentage'] = sum([freq for char, freq in diff_counter.items() if char not in string.ascii_letters]) / sum(diff_counter.values())
    print(dto)
    with open('data/ascii_char_frequencies_ff.json', 'w') as f:
        json.dump(dto, f)


def spell_check():
    """
    Numerous sanity check involving ASPELL for same-author and
    different-author part.
    """
    s = aspell.Speller('lang', 'en')
    same_vocab_ff = set()
    diff_vocab_ff = set()
    c = 0
    with open('data/vocab_list_verbatim_ff_2021-07-16_18-15-43.json', 'r') as f:
        vocab_list_verbatim = json.load(f)['data']#[:1000]
    for l in tqdm(vocab_list_verbatim):
        if c < 27834:
            same_vocab_ff.update(l)
        else:
            diff_vocab_ff.update(l)
        c += 1
    same_proper_words = [t for t in tqdm(same_vocab_ff) if s.check(t)]
    diff_proper_words = [t for t in tqdm(diff_vocab_ff) if s.check(t)]
    same_oov_words = [t for t in tqdm(same_vocab_ff) if not s.check(t)]
    diff_oov_words = [t for t in tqdm(diff_vocab_ff) if not s.check(t)]
    total_proper_words = [t for t in tqdm(same_vocab_ff.union(diff_vocab_ff)) if s.check(t)]
    total_oov_words = [t for t in tqdm(same_vocab_ff.union(diff_vocab_ff)) if not s.check(t)]
    total_words = list(set(total_oov_words).union(set(total_proper_words)))

    dto_json = dict()
    # dto_json['Size of in vocab words in same-author part / same_proper_words'] = len(same_proper_words)
    # dto_json['Size of in vocab words in diff-author part / diff_proper_words'] = len(diff_proper_words)
    # dto_json['Size of out of vocab words in same-author part / same_oov_words'] = len(same_oov_words)
    # dto_json['Size of out of vocab words in diff-author part / diff_oov_words'] = len(diff_oov_words)
    # dto_json['Size of in vocab words overall / total_proper_words'] = len(total_proper_words)
    # dto_json['Size of out of vocab words overall / total_oov_words'] = len(total_oov_words)
    # dto_json['Total vocab'] = len(same_vocab_ff.union(diff_vocab_ff))
    # dto_json['Size of vocab in same-author part'] = len(same_vocab_ff)
    # dto_json['Size of vocab in diff-author part'] = len(diff_vocab_ff)
    # dto_json['Percentage of in vocab words to all words in same-author part'] = len(same_proper_words) / len(same_vocab_ff)
    # dto_json['Percentage of in vocab words to all words in diff-author part'] = len(diff_proper_words) / len(diff_vocab_ff)
    # dto_json['Longest proper type in same-author part'] = len(max(same_proper_words, key=len))
    # dto_json['Longest proper type in diff-author part'] = len(max(diff_proper_words, key=len))
    # longest_proper = len(max(total_proper_words, key=len))
    # dto_json['Longest proper type overall'] = longest_proper
    # dto_json['Longest OOV type in same-author part'] = len(max(same_oov_words, key=len))
    # dto_json['Longest OOV type in diff-author part'] = len(max(diff_oov_words, key=len))
    # dto_json['Longest OOV type overall'] = len(max(total_oov_words, key=len))
    # long_oov = [w for w in tqdm(total_oov_words) if len(w) > longest_proper]
    # short_oov = [w for w in tqdm(total_oov_words) if len(w) <= longest_proper]
    # dto_json['Number of OOV types longer than longest proper word (total)'] = len(long_oov)
    # dto_json['Number of OOV types shorter or equal than longest proper word (total)'] = len(short_oov)
    # dto_json['Total types with >= 1 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 0])
    # dto_json['Total types with >= 2 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 1])
    # dto_json['Total types with >= 3 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 2])
    # dto_json['Total types with >= 4 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 3])
    # dto_json['Total types with >= 5 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 4])
    # dto_json['Total types with >= 6 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 5])
    # dto_json['Total types with >= 7 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 6])
    # dto_json['Total types with >= 8 punctuation symbols'] = len([1 for w in tqdm(total_words) if len([c for c in w if c in punctuation]) > 7])
    alphabet = set()
    [alphabet.update(w) for w in total_words]
    dto_json['Alphabet'] = (sorted(alphabet))




    # bad_texts = 0
    # for l in tqdm(vocab_list_verbatim):
    #     for w in l:
    #         if any(b in w for b in ["一"]):
    #             bad_texts += 1
    #             continue
    # dto_json['bad_pairs'] = bad_texts
    # dto_json['pairs'] = len(vocab_list_verbatim)

    # for i in range(10):
    #     print(f'\n>{i} punctuation symbols:')
    #     print(random.sample([w for w in total_words if len([c for c in w if c in punctuation]) == i], 10))
    #
    # print(f'Same OOV sample: {random.sample(same_oov_words, 100)}')
    # print(f'Diff OOV sample: {random.sample(diff_oov_words, 100)}')
    # print(f'Long OOV sample: {random.sample(long_oov, 100)}')
    # print(f'Short OOV sample: {random.sample(short_oov, 100)}')

    # vocab_list_proper = []
    # vocab_list_improper = []
    # for single_vocab_list in tqdm(vocab_list_verbatim):
    #     vocab_list_proper.append([word for word in single_vocab_list if s.check(word)])
    #     vocab_list_improper.append([word for word in single_vocab_list if not s.check(word)])
    #
    #
    # dto_json['Sum of sizes of pair vocab, in vocab words / vocab_list_proper'] = sum([len(x) for x in vocab_list_proper])
    # dto_json['Sum of sizes of pair vocab, out of vocab words / vocab_list_improper'] = sum([len(x) for x in vocab_list_improper])
    # dto_json['Sum of sizes of pair vocab, all words / vocab_list_verbatim'] = sum([len(x) for x in vocab_list_verbatim])
    #
    # dto_proper = dict()
    # dto_improper = dict()
    # dto_proper['data'] = vocab_list_proper
    # dto_improper['data'] = vocab_list_improper

    timestamp = now()
    # with open(f'data/vocab_list_verbatim_ff_proper_{timestamp}', 'w') as f:
    #     json.dump(dto_proper, f)
    # with open(f'data/vocab_list_verbatim_ff_improper_{timestamp}', 'w') as f:
    #     json.dump(dto_improper, f)

    with open('data/spellcheck_ff.json', 'w') as f:
        json.dump(dto_json, f, indent=4, ensure_ascii=False)


def ff_textlengths():
    """
    Plot text lengths of the Fanfiction dataset.
    """
    lengths = []
    with open('../teahan03-phonetic/data/verbatim_pan20-authorship-verification-training-small.jsonl') as f:
        for line in tqdm(f):
            pair = json.loads(line)
            lengths.extend([len(pair['pair'][0]), len(pair['pair'][1])])

    print(f'Zero-length texts: {sum(1 for l in lengths if l == 0)}')
    temp = sum(1 for l in lengths if l < 20500 or l >= 22500)
    print(f'Texts <20500 and >=22500: {temp}, percentage: {temp/len(lengths)}')

    plt.figure(figsize=(5, 3.5))
    plt.hist(lengths, bins=100, range=(20000, 23000))
    plt.xlabel('Text length (chars)')
    plt.ylabel('Frequency')
    plt.savefig(f'data/ff_textlengths.svg', format='svg', bbox_inches='tight')
    # Values for correctly cleaned verbatim text, Fan-fiction dataset
    # Zero-length texts: 2
    # Texts <20500 and >=22500: 1698, percentage: 0.016140377559361988


if __name__ == '__main__':
    plot_vocab_lists()
    # individual_vocab_size()
    # individual_text_length()
    # count_authors()
    # char_frequencies()
    # spell_check()
    # ff_textlengths()