"""
    A tools file to collect some reusable NLTK functions
"""
import importlib
import nltk
import string

"""
The following is to fix the problem of NLTK not having certain data sets.
 Using this work around after reading these SO answers

 NLTK for the dataset is stale as it was imported before the dataset existed.
  according to this SO answer and comment: https://stackoverflow.com/a/23715469/2912901
 Reload a module: https://stackoverflow.com/a/1254379/2912901
"""
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    importlib.reload(nltk)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    importlib.reload(nltk)


stops = nltk.corpus.stopwords.words('english')


def tokenize(data,method='word'):
    if method == 'word':
        return nltk.word_tokenize(data)
    elif method == 'sentence':
        return nltk.sent_tokenize(data)


def clean_sentence(sentence):
    """
    Strip the sentence of newlines and punctuation
    :param sentence:
    :return:
    """
    # remove newlines
    sentence = sentence.replace('\n', ' ')
    # remove trailing whitespace
    sentence = sentence.strip()
    # remove trailing punctuation.
    sentence = sentence.strip(string.punctuation)
    return sentence


def filter_pos_list_to_list(words, keep_tags):
    out_words = []
    tagged = nltk.pos_tag(words)
    for (word, tag) in tagged:
        if tag in keep_tags:
            out_words.append(word)
    return out_words


def filter_stopwords_list_to_list(words):
    out_words = []
    # stops = stopwords.words('english')
    for word in words:
        if word not in stops:
            out_words.append(word)
    return out_words
