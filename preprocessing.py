import unicodedata
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import fasttext

LANGUAGES = {
    'hi': 'hindi',
    'bn': 'bengali', 
    'or': 'odia',
    'ta': 'tamil',
    'te': 'telugu'
}

def nfc_normalize(text):
    return unicodedata.normalize('NFC', text)

def transliterate_roman(text, lang_code):
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except:
        return text

def preprocess(text, lang_code='hi'):
    text = nfc_normalize(text)
    normalizer_factory = IndicNormalizerFactory()
    normalizer = normalizer_factory.get_normalizer(lang_code)
    text = normalizer.normalize(text)
    tokens = indic_tokenize.trivial_tokenize(text, lang_code)
    text = ' '.join(tokens)
    return text

if __name__ == '__main__':
    sample = "यह एक परीक्षण वाक्य है"
    print(preprocess(sample, 'hi'))
