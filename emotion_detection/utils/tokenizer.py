import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


_word_tokenizer = RegexpTokenizer(r"\w+")

## to check if nltk needs download
try:
    _stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    _stop_words = set(stopwords.words("english"))

_lemmatizer = WordNetLemmatizer()


def nltk_tokenizer(document):

    """ NLTK tokenizer function
    """

    tokens = [w.lower() for w in tokens if not w.lower() in _stop_words]
    lems = []
    for item in tokens:
        lems.append(_lemmatizer.lemmatize(item))
    return lems


def nltk_tokenizer_df(document):

    """ NLTK tokenizer to work with dataframe
    """
    tokens = _word_tokenizer.tokenize(document)
    tokens = [w.lower() for w in tokens if not w.lower() in _stop_words]
    lems = []
    for item in tokens:
        lems.append(_lemmatizer.lemmatize(item))
    return " ".join(lems)
