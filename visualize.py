"""
Script for calculating significance values and visualizing results.
Input .json structure for ONE transcription:
{
    1:{
        f1: [x, x, x, x, x, x, x, x, x, x],
        accuracy: [x, x, x, x, x, x, x, x, x, x],
        precision: [x, x, x, x, x, x, x, x, x, x],
        ...
    },
    2:{
    },
    ...
    avg:{
    f1: x,
    accuracy: x,
    ...
    }
}

Input:
folder of crossval results, one subfolder must be named 'verbatim' /
'verbatimoriginal'
"""
import json
import os
import sys
from glob import glob
from statistics import mean, stdev

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import t as t_dist


manual_order = [
    #'verbatimoriginal',
    'verbatim',
    'ipa',
    'asjp',
    'dolgo',
    'refsoundex',
    'metaphone',
    'soundex',
    'cv',
    'ipa_4grams',
    'punct_4grams',
    'asjp_4grams',
    'dolgo_4grams',
    'cv_4grams',
    'punct',
    'punct_lemma',
    'punct_lemma_stop'
]


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
    'punct_4grams': '$P$ $4$-$grams$',
    'asjp_4grams': '$ASJP$ $4$-$grams$',
    'cv_4grams': '$CV$ $4$-$grams$',
    'punct_4grams': '$P$ $4$-$grams$',
    'verbatimoriginal': '$Verbatim$ $(orig.)$',
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


def load_unmasking_results(input_folder):
    """
    Loader for Unmasking results
    """
    results = dict()
    directory = [d for d in os.scandir(input_folder)]
    print(f'Found {len(directory)} evaluation results.')
    for dir_entry in directory:
        # Construct path for Unmasking directory hierarchy
        path_to_json = glob(os.path.join(dir_entry.path,
                                         'job_*',
                                         'CrossvalResult.*'))[0]
        with open(path_to_json, 'r') as f:
            results[dir_entry.name] = json.load(f)
    return results


def load_teahan03_results(input_folder):
    """
    Loader for teahan03 results
    """
    results = dict()
    directory = [d for d in os.scandir(input_folder)]
    print(f'Found {len(directory)} evaluation results.')
    for dir_entry in directory:
        with open(dir_entry.path, 'r') as f:
            if '_pan20' in dir_entry.name:
                results[dir_entry.name.split('_pan20')[0]] = json.load(f)  # Hacky
            else:
                results[dir_entry.name.split('_gb')[0]] = json.load(f)  # Hacky
    return results


def p():
    """
    Determine p-values, generate table, and plot.
    """
    # results = load_teahan03_results('../teahan03-phonetic/data/evaluated_2021-07-30_22-18-28_ff_r05_final_thisone')  # r = 0.05
    # verbatim = 'verbatimoriginal'
    # OR
    results = load_unmasking_results('../unmasking/data/crossval_results_2021-07-30_16-57-42_30folds_32runs_final')
    # results = load_teahan03_results('../teahan03-phonetic/data/evaluated_2021-07-31_02-15-53_gb_r05_final')  # r = 0.05
    verbatim = 'verbatim'

    nr_samples = results[verbatim]['folds']  # 10
    df = nr_samples - 1  # 9, degrees of freedom
    nr_of_experiments = len(results[verbatim]['results']['0']['f1'] + results[verbatim]['results']['1']['f1'] + results[verbatim]['results']['2']['f1'])  # 30
    alpha_corrected_sign = 0.05 / nr_of_experiments  # bonferroni-corrected
    alpha_corrected_verysign = 0.01 / nr_of_experiments  # bonferroni-corrected
    alpha_corrected_extremelysign = 0.001 / nr_of_experiments  # bonferroni-corrected
    measures = results[verbatim]['results']['0'].keys()
    evaluation = dict()
    print(nr_samples, df, nr_of_experiments, alpha_corrected_extremelysign, measures)
    for system in results.keys():
        evaluation[system] = dict()
        print(f'Determining p-values for {system}')
        if system == verbatim:
            continue
        for measure in measures:
            original = np.array(results[verbatim]['results']['0'][measure] + results[verbatim]['results']['1'][measure] + results[verbatim]['results']['2'][measure])
            changed = np.array(results[system]['results']['0'][measure] + results[system]['results']['1'][measure] + results[system]['results']['2'][measure])
            d = original - changed
            x = mean(d)
            s = stdev(d)
            delta = 0  # Hypothesized mean difference is 0 because null hypothesis is that the means of original and changed are equal
            t = (x - delta) * nr_samples / s

            # Alternate hypothesis: u1 != u2, Increase or decrease in performance is statistically significant.
            p_inc = 2 * t_dist.cdf(-abs(t), df)
            if p_inc < alpha_corrected_extremelysign:
                evaluation[system][measure] = '^{*\\! *\\! *}'
            elif p_inc < alpha_corrected_verysign:
                evaluation[system][measure] = '^{*\\! *}'
            elif p_inc < alpha_corrected_sign:
                evaluation[system][measure] = '^{*}'
            else:
                evaluation[system][measure] = ''

    # Print table with absolute values and asterisks
    measures = ['precision', 'recall', 'f1', 'f_05_u', 'c_at_1']
    print('\\bf System', end=' & ')
    for measure in measures:
        print(f'\\bf {measure_to_label[measure]}', end=' & ')
    print()
    print('\\midrule')
    # Place verbatim here when putting it in the .tex file
    print('\\midrule')
    for system, eval in evaluation.items():
        print(labels[system], end=' & ')
        for measure in measures:

            if system == verbatim:
                print(f'${round(results[system]["results"]["avg"][measure], 4)}$', end=' & ')
            else:
                verbatim_value = results[verbatim]["results"]["avg"][measure]
                current_value = results[system]["results"]["avg"][measure]
                temp = eval[measure]
                if temp != '':
                    if verbatim_value >= current_value:
                        temp = f'{temp}_{{-}}'
                    else:
                        temp = f'{temp}_{{+}}'
                print(f'${round(results[system]["results"]["avg"][measure], 4)}{temp}$', end=' & ')

    # Make plot
    measure_to_plot = 'f1'

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(figsize=(5, 3.5))
    values = dict()
    for system in results.keys():
        values[system] = results[system]['results']['avg'][measure_to_plot]
    print(sorted(values.items(), key=lambda x: manual_order.index(x[0])))
    data = [(labels[k], v) for k, v in sorted(values.items(), key=lambda x: manual_order.index(x[0]))]
    # sys.exit(0)
    # df = DataFrame(data)
    # df.plot.bar()
    plt.bar(*zip(*data), color='#185B8C')  # Dark blue
    plt.xticks(rotation='vertical')
    plt.ylim(0, 1)
    plt.ylabel(measure_to_label[measure_to_plot])
    plt.grid(axis='y')

    # Add differences to first bar
    for i in range(1, len(data)):
        diff = round(data[i][1] - data[0][1], 4)
        if diff < 0:
            color = '#B30000'  # Dark red
            diff = f'  -{abs(diff)}'
        elif diff > 0:
            color = '#006600'  # Dark green
            diff = f'  +{diff}'
        else:
            color = 'k'
            diff = f'  {diff}'
        plt.text(i, 1, diff, ha='center', rotation='vertical', color=color)

    plt.savefig(f'data/{measure_to_plot}_gb_unmasking_correct_order.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    p()