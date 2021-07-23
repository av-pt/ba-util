"""
Script to clean the PAN20 Fan-fiction dataset.
Executes the following steps in order:
 - Remove tokens longer than 23 characters
 - Remove tokens with 3 or more punctuation symbols (37510 types)
 - Remove tokens containing symbols that are not in transcribable_ff or punctuation
 - Replace " with ' (" does not transcribe, ' does)
 - Remove excessively long or short texts
Persists the cleaned dataset to a new folder also containing a corrected truth file.
"""
from string import punctuation
import re

import spacy

# Contains lowercase characters that g2p_en can transcribe directly
transcribable_ff_lowercase = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                              'i', 'j',
                              'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'á',
                              'â', 'ã',
                              'ä', 'å', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù',
                              'ú', 'û',
                              'ü', 'ý', 'ÿ', 'ā', 'ă', 'ą', 'ć', 'ĉ', 'č', 'ď', 'ē', 'ĕ', 'ė', 'ę', 'ě', 'ĝ', 'ğ', 'ġ',
                              'ĥ', 'ī',
                              'ĭ', 'į', 'ĺ', 'ļ', 'ľ', 'ń', 'ņ', 'ň', 'ō', 'ŏ', 'ő', 'ŕ', 'ŗ', 'ř', 'ś', 'ŝ', 'ş', 'š',
                              'ţ', 'ť',
                              'ũ', 'ū', 'ŭ', 'ů', 'ű', 'ų', 'ŵ', 'ŷ', 'ź', 'ż', 'ž', 'ơ', 'ư', 'ǎ', 'ǐ', 'ǒ', 'ǔ', 'ǘ',
                              'ǧ', 'ǫ',
                              'ǵ', 'ǹ', 'ǻ', 'ȁ', 'ȅ', 'ȇ', 'ȉ', 'ȋ', 'ȍ', 'ȏ', 'ȓ', 'ș', 'ț', 'ȟ', 'ȧ', 'ȩ', 'ȫ', 'ȯ',
                              'ȳ', 'ḁ',
                              'ḋ', 'ḍ', 'ḑ', 'ḓ', 'ḙ', 'ḛ', 'ḣ', 'ḥ', 'ḧ', 'ḩ', 'ḫ', 'ḭ', 'ḯ', 'ḷ', 'ḻ', 'ḽ', 'ḿ', 'ṁ',
                              'ṃ', 'ṅ',
                              'ṇ', 'ṛ', 'ṟ', 'ṡ', 'ṣ', 'ṩ', 'ṫ', 'ṭ', 'ṯ', 'ṱ', 'ṳ', 'ṷ', 'ṻ', 'ṽ', 'ṿ', 'ẁ', 'ẃ', 'ẇ',
                              'ẍ', 'ẖ',
                              'ẗ', 'ẘ', 'ạ', 'ầ', 'ậ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ị', 'ọ', 'ỏ', 'ộ', 'ớ', 'ờ', 'ở', 'ủ', 'ỳ',
                              'ỵ', 'ỷ']
transcribable = transcribable_ff_lowercase + list(punctuation)

nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])


def clean(doc):
    return ''.join([x.text_with_ws for x in doc if clean_cond(x)])


def clean_cond(token):
    return all(c in transcribable for c in token.text.lower()) and len(token.text) <= 23 and len([c for c in token.text if c in punctuation]) <= 2


def replace_special(text):
    return text.replace('"', "'").replace('...', '. ')


def main():
    pass


if __name__ == '__main__':
    main()
