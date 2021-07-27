"""
Naive proof-of-concept Authorship Verification classifier using vocab
size differences as the only parameter in training and classification.
"""
import json
import time

import numpy as np
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from pan20_verif_evaluator import evaluate_all


def now(): return time.strftime("%Y-%m-%d_%H-%M-%S")


def prep():

    # Load dataset
    data = []
    data_ids = set()
    # with open('../unmasking/NAACL-19/corpus/pan20/transcribed/punct/punct_gb.jsonl', 'r') as f:
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/punct/punct_pan20-authorship-verification-training-small.jsonl', 'r') as f:
        # nr = len(data)
        for line in tqdm(f):
            # print(len(data))
            d = json.loads(line)
            if d['id'] in data_ids:
                continue
            data_ids.add(d['id'])
            data.append(d)

    # Load truth
    truth = []
    truth_ids = set()
    # with open('../unmasking/NAACL-19/corpus/pan20/transcribed/punct/gb-truth.jsonl', 'r') as f:
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/punct/pan20-authorship-verification-training-small-truth.jsonl', 'r') as f:
        for line in tqdm(f):
            d = json.loads(line)
            if d['id'] in truth_ids:
                continue
            truth_ids.add(d['id'])
            truth.append(d)

    # data_ids = set([x['id'] for x in data])
    # truth_ids = set([x['id'] for x in truth])
    # print(len(data))
    # print(len(truth))
    # print(len(data_ids))
    # print(len(truth_ids))
    # print(data_ids.difference(truth_ids))
    # c_data = Counter()
    # c_truth = Counter()
    # c_data.update([x['id'] for x in data])
    # c_truth.update([x['id'] for x in truth])
    # print(c_data.most_common(50))
    # print(c_truth.most_common(50))


    # Assert that data and truth are aligned / sorted
    print(f'Data is aligned: {all([data["id"] == truth["id"] for data, truth in tqdm(zip(data, truth))])}')
    print(f'Data length: {len(data)}')
    print(f'Truth length: {len(truth)}')

    # Calculate absolute vocab size differences
    # Feature: Size of symmetric difference of the vocab sets of the pair
    # prepared_data = [len(set(sample['pair'][0].split(' ')).symmetric_difference(set(sample['pair'][1].split(' ')))) for sample in tqdm(data)]
    # Feature: Difference between sizes of vocab sets of a pair
    prepared_data = [abs(len(set(sample['pair'][0].split(' '))) - len(set(sample['pair'][1].split(' ')))) for sample in tqdm(data)]
    # print(prepared_data[:20])

    dto = dict()
    dto['data'] = prepared_data
    with open(f'data/naive_prepared_symdiff_ff_{now()}.json', 'w') as f:
        json.dump(dto, f)

def cv():
    # with open('data/naive_prepared_gb_2021-07-21_14-45-17.json', 'r') as f:
    with open('data/naive_prepared_ff_2021-07-21_14-35-02.json', 'r') as f:
        prepared_data = json.load(f)['data']

    # Load truth
    truth = []
    truth_ids = set()
    # with open('../unmasking/NAACL-19/corpus/pan20/transcribed/punct/gb-truth.jsonl', 'r') as f:
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/punct/pan20-authorship-verification-training-small-truth.jsonl', 'r') as f:
        for line in tqdm(f):
            d = json.loads(line)
            if d['id'] in truth_ids:
                continue
            truth_ids.add(d['id'])
            truth.append(d)


    X = np.array(prepared_data, dtype=np.float64).reshape(-1, 1)
    y = np.array([t['same'] for t in truth], dtype=np.float64)


    # Out-of-fold 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10)
    pred_y = []
    true_y = []
    for train, test in tqdm(kf.split(X, y)):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        # clf = linear_model.LogisticRegression()
        clf = linear_model.LinearRegression()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_y.extend(pred)
        true_y.extend(y_test)

        # for X_inner, y_inner in zip(X_test, y_test):
        #     pred = clf.predict([X_inner])
        #     # All values around 0.5 are transformed to 0.5
        #     if 0.5 - radius <= pred[0, 1] <= 0.5 + radius:
        #         pred[0, 1] = 0.5
        #     pred_y.append(pred[0, 1])
        #     true_y.append(y_inner)
    # pred_y = [0.5 if 0.49 < x < 0.51 else x for x in pred_y]

    # print(f'Number of samples: {len(X)}\n'
    #       f'Number of predictions: {len(pred_y)}\n'
    #       f'Size of ground truth: {len(true_y)}')
    # print(pred_y, true_y)

    # Evaluate
    results = evaluate_all(true_y, pred_y)
    binarized = [1 if x > 0.5 else 0 for x in pred_y]
    results['recall'] = recall_score(true_y, binarized)
    results['precision'] = precision_score(true_y, binarized)
    print(results)

    with open(f'data/naive_eval_ff_{now()}.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def random():
    # Load truth
    truth = []
    truth_ids = set()
    # with open('../unmasking/NAACL-19/corpus/pan20/transcribed/punct/gb-truth.jsonl', 'r') as f:
    with open('../teahan03-phonetic/data/transcribed_remote/transcribed/punct/pan20-authorship-verification-training-small-truth.jsonl', 'r') as f:
        for line in tqdm(f):
            d = json.loads(line)
            if d['id'] in truth_ids:
                continue
            truth_ids.add(d['id'])
            truth.append(d)

    y = np.array([t['same'] for t in truth], dtype=np.float64)

    r = np.random.uniform(low=0, high=1, size=len(y))

    results = evaluate_all(y, r)
    binarized = [1 if x > 0.5 else 0 for x in r]
    results['recall'] = recall_score(y, binarized)
    results['precision'] = precision_score(y, binarized)
    print(results)


if __name__ == '__main__':
    # prep()
    # cv()
    random()