"""
    A tools file to collect some reusable NLTK functions
"""
import nltk
from nltk.corpus import stopwords
import string


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
    # remove trailing punctuation.
    sentence.strip(string.punctuation)
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
    stops = stopwords.words('english')
    for word in words:
        if word not in stops:
            out_words.append(word)
    return out_words
