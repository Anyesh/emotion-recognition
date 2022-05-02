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

    tokens = [w.lower() for w in tokens if w.lower() not in _stop_words]
    return [_lemmatizer.lemmatize(item) for item in tokens]


def nltk_tokenizer_df(document):

    """ NLTK tokenizer to work with dataframe
    """
    tokens = _word_tokenizer.tokenize(document)
    tokens = [w.lower() for w in tokens if w.lower() not in _stop_words]
    lems = [_lemmatizer.lemmatize(item) for item in tokens]
    return " ".join(lems)
