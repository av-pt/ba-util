# ba-util
Utilities for my Bachelor's Thesis

NOTE: Needs `aspell` and `aspell-en` to be installed on the system.

```
pipenv install
pipenv run python -m spacy download en_core_web_sm

# Transcribing the Gutenberg dataset (locations might be different)
pipenv run python ba-util/transcribe.py -i unmasking/NAACL-19/corpus/pan20/gb.jsonl -t unmasking/NAACL-19/corpus/pan20/gb-truth.jsonl -o unmasking/NAACL-19/corpus/pan20/transcribed -s -d gb
# Change -d gb to -d ff for Fan-fiction datset


pipenv run python vocab_sizes.py -i ../unmasking/NAACL-19/corpus/pan20/transcribed/
```
