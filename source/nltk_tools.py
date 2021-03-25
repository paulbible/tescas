"""
    A tools file to collect some reusable NLTK functions
"""
from nltk.corpus import stopwords


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
