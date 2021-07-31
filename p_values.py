"""
Calculate p-values.
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
from scipy.stats import t as t_dist


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
    Determine p-values.
    """
    # results = load_teahan03_results('../teahan03-phonetic/data/evaluated_2021-07-30_22-18-28_ff_r05_final_thisone')  # r = 0.05
    # verbatim = 'verbatimoriginal'
    # OR
    # results = load_unmasking_results('../unmasking/data/crossval_results_2021-07-30_16-57-42_30folds_32runs_final')
    results = load_teahan03_results('../teahan03-phonetic/data/evaluated_2021-07-31_02-15-53_gb_r05_final')  # r = 0.05
    verbatim = 'verbatim'

    # print(results[verbatim]['results']['0']['f1'])
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
            # print(measure)
            original = np.array(results[verbatim]['results']['0'][measure] + results[verbatim]['results']['1'][measure] + results[verbatim]['results']['2'][measure])
            changed = np.array(results[system]['results']['0'][measure] + results[system]['results']['1'][measure] + results[system]['results']['2'][measure])
            d = original - changed
            x = mean(d)
            s = stdev(d)
            # print(d, x, s)
            # sys.exit(0)
            delta = 0  # Hypothesized mean difference is 0 because null hypothesis is that the means of original and changed are equal
            t = (x - delta) * nr_samples / s
            # print(t)
            # Alternate hypothesis: u1 != u2, Increase or decrease in performance is statistically significant.
            p_inc = 2 * t_dist.cdf(-abs(t), df)
            if p_inc < alpha_corrected_extremelysign:
                evaluation[system][measure] = '***'
            elif p_inc < alpha_corrected_verysign:
                evaluation[system][measure] = '**'
            elif p_inc < alpha_corrected_sign:
                evaluation[system][measure] = '*'
            else:
                evaluation[system][measure] = '='

    # Print table
    measures = ['precision', 'recall', 'f1', 'f_05_u', 'c_at_1']
    print('\\bf System', end=' & ')
    for measure in measures:
        print(f'\\bf {measure_to_label[measure]}', end=' & ')
    print()
    print('\\midrule')
    for system, eval in evaluation.items():
        if system == verbatim:
            continue
        print(labels[system], end=' & ')
        for measure in measures:
            temp = f'${eval[measure]}$'
            if temp == '$$':
                temp = ''
            print(temp, end=' & ')
        print()


if __name__ == '__main__':
    p()