"""
Tool for visualizing the results of multiple Authorship Verification
evaluations.
Input:
Path to folder of .json files. Path must end in '/'
{
    results: {
        measure1: number,
        measure2: number,
        ...
    }
    folds: int (optional)
}
"""
import argparse
import time
import os
import json
from glob import glob
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


manual_order = [
    'verbatim',
    'ipa',
    'dolgo',
    'asjp',
    'cv',
    'soundex',
    'refsoundex',
    'metaphone',
    'punct',
    'punct_lemma',
    'punct_lemma_stop'
]

system_to_label = {
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

measure_to_label = {
    'accuracy': 'Accuracy',
    'c_at_1': 'c@1',
    'c@1': 'c@1',
    'frac_classified': 'Fraction classified',
    'f1': 'F1',
    'F1': 'F1',
    'precision': 'Precision',
    'recall': 'Recall',
    'recall_total': 'Recall total',
    'f_05_u': 'F0.5u',
    'auc': 'AUC',
    'overall': 'Overall'
}


def now(): return time.strftime("%Y-%m-%d_%H-%M-%S")


def load_unmasking_results(input_folder):
    results = dict()
    # folds = []
    directory = [d for d in os.scandir(input_folder)]
    print(f'Found {len(directory)} evaluation results.')
    for dir_entry in directory:
        # Construct path for Unmasking directory hierarchy
        path_to_json = glob(os.path.join(dir_entry.path,
                                         'job_*',
                                         'CrossvalResult.*'))[0]
        with open(path_to_json, 'r') as f:
            j = json.load(f)
            results[dir_entry.name] = j['results']

    # if folds and not all(f == folds[0] for f in folds):
    #     sys.exit('Data contains ambiguous folds.')
    # elif folds:
    #     f = folds[0]

    return results


def load_teahan03_results(input_folder):
    results = dict()
    # folds = []
    directory = [d for d in os.scandir(input_folder)]
    for dir_entry in directory:
        with open(dir_entry.path, 'r') as f:
            j = json.load(f)
            print(dir_entry.name[:5], j)
            results[dir_entry.name.split('_pan20')[0]] = j  # Hacky
    return results


def main():
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Plot Authorship Verification evaluation results",
        add_help=True)
    parser.add_argument('--input',
                        '-i',
                        required=True,
                        help='Path to directory of evaluation results')
    parser.add_argument('--output',
                        '-o',
                        required=True,
                        help='Name to tag the output file')
    parser.add_argument('--vocab-sizes',
                        '-v',
                        help='File with vocab sizes')
    args = parser.parse_args()

    if args.vocab_sizes is None:
        out_path = os.path.join('data', f'plot_{now()}_{args.output}_bar')
    else:
        out_path = os.path.join('data', f'plot_{now()}_{args.output}_sct')
    os.makedirs(out_path, exist_ok=True)

    # Load data (Unmasking)

    if str.startswith(os.path.basename(os.path.dirname(args.input)), 'crossval_results'):
        results = load_unmasking_results(args.input)
    else:  # Load data (other)
        results = load_teahan03_results(args.input)

    df = pd.DataFrame(data=results, columns=manual_order)  # Transposed
    df.rename(columns=system_to_label, inplace=True)
    df = df.T
    print(df)

    if args.vocab_sizes is None:
        for measure, values in df.iteritems():
            plt.rcParams['mathtext.fontset'] = 'dejavuserif'
            plt.figure(figsize=(5, 3.5))
            values.plot.bar()
            plt.ylim(0, 1)
            plt.ylabel(measure_to_label[measure])
            plt.grid(axis='y')

            # Add differences to first bar
            for i in range(1, len(values)):
                diff = round(values[i]-values[0], 4)
                if diff < 0:
                    color = 'r'
                    diff = f' -{abs(diff)}'
                elif diff > 0:
                    color = 'g'
                    diff = f' +{diff}'
                else:
                    color = 'k'
                plt.text(i, values[i], diff, ha='center', rotation='vertical', color=color)

            plt.savefig(os.path.join(out_path, f'{measure}.svg'), format='svg', bbox_inches='tight')
            plt.clf()
    else:
        with open(args.vocab_sizes, 'r') as f:
            type_dto = json.load(f)
        for measure, values in df.iteritems():

            plt.rcParams['mathtext.fontset'] = 'dejavuserif'
            plt.figure(figsize=(5, 3.5))
            x_values = []
            y_values = []
            labels = []
            texts = []
            for k, v in values.iteritems():
                x_values.append(type_dto[k])
                y_values.append(v)
                labels.append(k)
                texts.append(plt.text(type_dto[k], v, k))

            plt.ylim(0, 1)
            plt.title(f'{measure_to_label[measure]} vs. vocabulary size')
            plt.xlabel('Vocabulary size (types)')
            plt.ylabel(measure_to_label[measure])
            plt.grid()
            plt.scatter(x_values, y_values)  # s=5 -> size

            m, b = np.polyfit(x_values, y_values, 1)
            p = np.poly1d([m, b])
            plt.plot(x_values, p(x_values), 'k', linewidth=1)
            steps = adjust_text(texts, force_points=1, autoalign='y', arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
            print(f'Label placement iterations for {measure}: {steps}')

            plt.savefig(os.path.join(out_path, f'{measure}.svg'), format='svg', bbox_inches='tight')


if __name__ == "__main__":
    main()
