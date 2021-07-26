"""
Goes through the Fan-ficiton dataset and removes texts of length
<20500 and >=22500 characters.
Creates a copy of the dataset.

Values for correctly cleaned verbatim text, Fan-fiction dataset
Zero-length texts: 2
Texts <20500 and >=22500: 1698, percentage: 0.016140377559361988
"""
import argparse
import json
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        prog='transcribe',
        description='Remove extremely long and short samples',
        add_help=True)
    parser.add_argument('-i', '--input', type=str, help='Path to a PAN20 folder of transcriptions')
    args = parser.parse_args()

    # Find IDs of texts to be removed
    lengths = []
    to_be_removed = []
    nr_texts = 0
    with open(os.path.join(args.input, 'verbatimcleaned', 'verbatimcleaned_pan20-authorship-verification-training-small.jsonl')) as f:
        for line in tqdm(f):
            pair = json.loads(line)
            first_length = len(pair['pair'][0])
            second_length = len(pair['pair'][1])
            if not(20500 < first_length <= 22500) or not(20500 < second_length <= 22500):
                to_be_removed.append(pair['id'])
    print(f'Found {len(to_be_removed)} samples to be removed.')

    directory = [d for d in os.scandir(args.input)]
    for dir_entry in directory:
        print(f'Filtering {dir_entry.name}')

        # Filtering data
        with open(os.path.join(args.input, dir_entry.name, f'{dir_entry.name}_pan20-authorship-verification-training-small.jsonl'), 'r') as infile:
            os.makedirs(os.path.join(os.path.dirname(args.input), f'{os.path.basename(args.input)}_lengthfiltered', dir_entry.name), exist_ok=True)
            with open(os.path.join(os.path.dirname(args.input), f'{os.path.basename(args.input)}_lengthfiltered', dir_entry.name, f'{dir_entry.name}_pan20-authorship-verification-training-small.jsonl'), 'w') as outfile:
                for line in tqdm(infile):
                    pair = json.loads(line)
                    if pair['id'] not in to_be_removed:
                        json.dump(pair, outfile)
                        outfile.write('\n')

        # Filtering truth
        with open(os.path.join(args.input, dir_entry.name, f'pan20-authorship-verification-training-small-truth.jsonl'), 'r') as infile:
            with open(os.path.join(os.path.dirname(args.input), f'{os.path.basename(args.input)}_lengthfiltered', dir_entry.name, f'pan20-authorship-verification-training-small-truth.jsonl'), 'w') as outfile:
                for line in tqdm(infile):
                    pair = json.loads(line)
                    if pair['id'] not in to_be_removed:
                        json.dump(pair, outfile)
                        outfile.write('\n')


if __name__ == '__main__':
    main()
