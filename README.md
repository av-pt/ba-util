# ba-util
Utilities for my Bachelor's Thesis

- `clean_ff.py`: Cleaning of Fanfiction dataset
- `converters.py`: Transcriptions of single texts
- `naive_classify.py`: Proof-of-concept implementation of a naive AV classifier
- `pan20_verif_evaluator.py`: Official evaluation utilities from PAN 2020
- `plot_playground.py`: Miscellaneous visalizations
- `remove_long_short.py`: Utility for removing excessively long and short texts from the Fanfiction dataset
- `transcribe.py`: Transcriptions of whole PAN 2020 formatted datasets
- `visualize.py`: Visualize results, generate LaTeX tables, calculate p-values
- `vocab_sizes.py`: Compute and plot vocabulary size reduction

NOTE: Needs `aspell` and `aspell-en` to be installed on the system.

```
pipenv install
pipenv run python -m spacy download en_core_web_sm

# Transcribing the Gutenberg dataset (locations might be different)
pipenv run python ba-util/transcribe.py -i \
    unmasking/NAACL-19/corpus/pan20/gb.jsonl -t \
    unmasking/NAACL-19/corpus/pan20/gb-truth.jsonl -o \
    unmasking/NAACL-19/corpus/pan20/transcribed -s -d gb
# Change -d gb to -d ff for Fan-fiction datset


pipenv run python vocab_sizes.py -i ../unmasking/NAACL-19/corpus/pan20/transcribed/
```
