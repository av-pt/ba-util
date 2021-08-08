"""
Script providing transcribe_horizontal_gb / transcribe_horizontal_ff
functions to transcribe a text to a range of transcription systems.
"""

import atexit
import os
import re
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_suffix_regex

import ujson
from g2p_en import G2p
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pyclts import CLTS
from pyphonetics import Soundex, RefinedSoundex, Metaphone

from clean_ff import clean_cond, clean


def dump_with_message(msg, cache_loaded, cache_changed, obj, file_path, **kwargs):
    if cache_loaded and cache_changed:
        print(msg)
        with open(file_path, 'w') as fp:
            ujson.dump(obj, fp, **kwargs)


def persistent_cache(func):
    """
    Persistent cache decorator.
    Creates a "cache/" directory if it does not exist and writes the
    caches of the given func to the file "cache/<func-name>.cache" once
    on exit.
    """
    file_path = os.path.join('cache', f'{func.__name__}.cache')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        print(f'Loading {file_path}')
        with open(file_path, 'r') as fp:
            cache = ujson.load(fp)
    except (IOError, ValueError):
        cache = {}
    atexit.register(lambda: dump_with_message(f'Writing {file_path}',
                                              True,
                                              True,
                                              cache,
                                              file_path,
                                              indent=4))

    def wrapper(*args):
        if str(args) not in cache:
            cache[str(args)] = func(*args)
        return cache[str(args)]

    return wrapper


clts = CLTS(os.path.join('ba-util', 'clts'))

inner_g2p_en = G2p()


def g2p_en(verbatim):
    verbatim = re.sub(re.compile(r'[0-9]{37,}'), '', verbatim)
    return inner_g2p_en(verbatim)


nlp_no_apostrophe_split = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
nlp_no_apostrophe_split.tokenizer.rules = {key: value for key, value in nlp_no_apostrophe_split.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
suffixes = [suffix for suffix in nlp_no_apostrophe_split.Defaults.suffixes if suffix not in ["'s", "'S", '’s', '’S']]
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp_no_apostrophe_split.tokenizer.suffix_search = suffix_regex.search
nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
nlp.tokenizer.suffix_search = suffix_regex.search

# print(nlp.Defaults.suffixes)
# Remove apostrophe rules, so tokens like "I'm" are not split up


# Arpabet to IPA dict with stress
arpabet2ipa_orig = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AX': 'ə', 'AXR': 'ɚ', 'AY': 'aɪ',
                    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IX': 'ɨ', 'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ',
                    'UH': 'ʊ', 'UW': 'u', 'UX': 'ʉ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'DX': 'ɾ', 'EL': 'l̩',
                    'EM': 'm̩', 'EN': 'n̩', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'H': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l',
                    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'NX': 'ɾ̃', 'P': 'p', 'Q': 'ʔ', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
                    'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w', 'WH': 'ʍ', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'}
primary_stress = {key + '1': 'ˈ' + value for key, value in arpabet2ipa_orig.items()}
secondary_stress = {key + '0': 'ˌ' + value for key, value in arpabet2ipa_orig.items()}
arpabet2ipa = {**arpabet2ipa_orig, **primary_stress, **secondary_stress}

# Arpabet to IPA dict without stress
no_primary_stress = {key + '1': value for key, value in arpabet2ipa_orig.items()}
no_secondary_stress = {key + '0': value for key, value in arpabet2ipa_orig.items()}
no_tertiary_stress = {key + '2': value for key, value in arpabet2ipa_orig.items()}
arpabet2ipa_no_stress = {**arpabet2ipa_orig, **no_primary_stress, **no_secondary_stress, **no_tertiary_stress}

soundex = Soundex()
refsoundex = RefinedSoundex()
metaphone = Metaphone()

detokenizer = TreebankWordDetokenizer()


def g2p_pyphonetics(verbatim, transcription_model):
    transcribed_tokens = []
    tokens = word_tokenize(verbatim)
    for token in tokens:
        if token.upper().isupper():
            transcribed_tokens.append(transcription_model.phonetics(token))
        else:
            transcribed_tokens.append(token)
    # transcription = detokenizer.detokenize(transcribed_tokens)
    transcription = ' '.join(transcribed_tokens)
    return transcription


# Init sound classes
sc = {
    'asjp': clts.soundclass('asjp'),
    'cv': clts.soundclass('cv'),
    'dolgo': clts.soundclass('dolgo')
}


@persistent_cache
def clts_translate(symbol, sound_class_system):
    return clts.bipa.translate(symbol, sc[sound_class_system])


whitespaces_regex = re.compile(r"\s+")


def ipa2sc(ipa_transcription, sound_class_system='dolgo'):
    """
    Takes an IPA transcribed, phoneme segmented string and replaces each
    phoneme to its corresponding sound class declared in 
    sound_class_system.
    Keeps punctuation where applicable.
    sound_class_system in {'art', 'asjp', 'color', 'cv', 'dolgo', 'sca'}
    """
    # Arpabet to IPA and tag s = symbol, p = punctuation
    char_ipa = [(arpabet2ipa_no_stress[symbol], 's') if symbol in arpabet2ipa_no_stress.keys() else (symbol, 'p') for
                symbol in ipa_transcription]
    char_sound_class = ''.join(
        [clts_translate(symbol, sound_class_system) if tag == 's' else ' ' for symbol, tag in char_ipa])
    # char_sound_class = ''.join(
    #     [clts_translate(symbol, sound_class_system) if tag == 's' else ' ' for symbol, tag in char_ipa])
    return whitespaces_regex.sub(' ', char_sound_class).strip()


def transcribe_horizontal_gb(verbatim):
    """
    Transcribes a given text. Returns a dict with the keys being the
    corresponding transcription system names.
    GB: Creates n-grams for Gutenberg dataset.
    """
    transcriptions = dict()

    doc_no_apostrophe_split = nlp_no_apostrophe_split(verbatim.strip())
    doc = nlp(verbatim.strip())
    # Verbatim
    transcriptions['verbatim'] = verbatim

    # Create miscellaneous transcriptions
    transcriptions['punct'] = ' '.join([token.text.lower()
                                        for token in doc_no_apostrophe_split
                                        if not token.is_punct])
    transcriptions['punct_lemma'] = ' '.join([token.lemma_
                                              for token in doc
                                              if not token.is_punct])
    transcriptions['punct_lemma_stop'] = ' '.join([token.lemma_
                                                   for token in doc
                                                   if (not token.is_stop
                                                       and not token.is_punct)])

    # Create IPA transcription
    ipa_transcription = ''
    phonemes = g2p_en(verbatim)
    for symbol in phonemes:
        if symbol in arpabet2ipa_no_stress.keys():
            ipa_transcription += arpabet2ipa_no_stress[symbol]
        elif symbol == ' ' and not ipa_transcription.endswith(' '):
            ipa_transcription += symbol
    transcriptions['ipa'] = ipa_transcription

    # Create Sound Class transcriptions, reusing IPA transcriptions
    for sound_class in {'cv', 'dolgo', 'asjp'}:
        transcriptions[sound_class] = ipa2sc(phonemes, sound_class)

    # Generate space-separated 4-grams
    for system in {'punct', 'ipa', 'asjp', 'dolgo', 'cv'}:
        tight = transcriptions[system].replace(' ', '')
        transcriptions[f'{system}_4grams'] = ' '.join([''.join(x) for x in zip(*[tight[i:] for i in range(4)])])

    # Create Soundex transcriptions
    transcriptions['soundex'] = ' '.join([soundex.phonetics(token.text) for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text)])
    transcriptions['refsoundex'] = ' '.join([refsoundex.phonetics(token.text) for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text)])
    transcriptions['metaphone'] = ' '.join([metaphone.phonetics(token.text) for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text)])

    return transcriptions


# @profile
def transcribe_horizontal_ff(verbatim):
    """
    Transcribes a given text. Returns a dict with the keys being the
    corresponding transcription system names.
    FF: Additional cleaning for Fan-Fiction dataset.
    """
    transcriptions = dict()

    doc_no_apostrophe_split = nlp_no_apostrophe_split(verbatim.strip())
    doc = nlp(verbatim.strip())
    # Verbatim
    transcriptions['verbatimcleaned'] = clean(doc)

    # Create miscellaneous transcriptions
    # transcriptions['punct'] = ' '.join([token.text for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text)])
    transcriptions['punct'] = ' '.join([token.text.lower()
                                        for token in doc_no_apostrophe_split
                                        if not token.is_punct
                                        and clean_cond(token)])
    transcriptions['punct_lemma'] = ' '.join([token.lemma_
                                              for token in doc
                                              if not token.is_punct
                                              and clean_cond(token)])
    transcriptions['punct_lemma_stop'] = ' '.join([token.lemma_
                                                   for token in doc
                                                   if (not token.is_stop
                                                       and not token.is_punct)
                                                       and clean_cond(token)])

    # Create IPA transcription
    ipa_transcription = ''
    phonemes = g2p_en(transcriptions['verbatim'])
    for symbol in phonemes:
        if symbol in arpabet2ipa_no_stress.keys():
            ipa_transcription += arpabet2ipa_no_stress[symbol]
        elif symbol == ' ' and not ipa_transcription.endswith(' '):
            ipa_transcription += symbol
    transcriptions['ipa'] = ipa_transcription

    # Create Sound Class transcriptions, reusing IPA transcriptions
    for sound_class in {'cv', 'dolgo', 'asjp'}:
        transcriptions[sound_class] = ipa2sc(phonemes, sound_class)

    # Create Soundex transcriptions
    transcriptions['soundex'] = ' '.join([soundex.phonetics(token.text) for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text) and clean_cond(token)])
    transcriptions['refsoundex'] = ' '.join([refsoundex.phonetics(token.text) for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text) and clean_cond(token)])
    transcriptions['metaphone'] = ' '.join([metaphone.phonetics(token.text) for token in doc_no_apostrophe_split if any(c.isalpha() for c in token.text) and clean_cond(token)])

    return transcriptions