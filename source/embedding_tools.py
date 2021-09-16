"""
    A tools file to collect some reusable embedding functions
"""
import numpy as np
from collections import defaultdict
import os
import nltk_tools
import string
import math


def load_embeddings(filename):
    """
    This function loads the embedding from a file and returns 2 things
    1) a word_map, this is a dictionary that maps words to an index.
    2) a matrix of row vectors for each work, index the work using the vector.

    :param filename:
    :return: word_map, matrix
    """
    count = 0
    matrix = []
    word_map = {}
    with open(filename, encoding="utf8") as f:
        # with open(filename) as f:
        for line in f:
            line = line.strip()
            items = line.split()
            word = items[0]
            rest = items[1:]
            # print("word:", word)
            word_map[word] = count
            count += 1

            rest = list(map(float, rest))
            matrix.append(rest)
    matrix = np.array(matrix)
    return word_map, matrix


def reduce_sum_word_list(words, word_map, matrix):
    """
    Take a list of words and summarize them as a vector using 'mean'.
    returns a numpy vector
    :param words:
    :param word_map:
    :param matrix:
    :return:
    """
    vec = np.zeros(matrix.shape[1])
    for word in words:
        word = word.lower()
        if word in word_map:
            index = word_map[word]
            vec = vec + matrix[index]
    return vec


def reduce_sum_word_list_weighted(words, word_map, matrix, word_weight_map):
    """
    Take a list of words and summarize them as a vector using 'mean'.
    returns a numpy vector
    :param words:
    :param word_map:
    :param matrix:
    :param word_weight_map:
    :return:
    """
    vec = np.zeros(matrix.shape[1])
    for word in words:
        word = word.lower()
        if word in word_map:
            index = word_map[word]
            if word in word_weight_map:
                vec = vec + matrix[index]*word_weight_map[word]
            else:
                vec = vec + matrix[index]
    return vec


def cossim(vA, vB):
    """
    Calcuate the cosine similarity value.
    Returns the similarity value, range: [-1, 1]
    :param vA:
    :param vB:
    :return: similarity
    """
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))


def calc_similar_values(word_map, matrix, query):
    """
    Claculate the similarity of every word vector with the query word vector
    :param word_map:
    :param matrix:
    :param query:
    :return:
    """
    if query not in word_map:
        print(query, "not found")
        return None

    values = []
    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        values.append(cossim(test_vector, vector))
    return values


def search_dissimilar(word_map, matrix, query, threshold=-0.90):
    """
    Search through the matrix and get words that are similar to the give word
    using the embedding vectors.
    :param word_map:
    :param matrix:
    :param query:
    :param threshold:
    :return:
    """
    if query not in word_map:
        print(query, "not found")
        return None

    words = ['' for i in range(len(word_map))]
    for word in word_map:
        words[word_map[word]] = word

    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        if cossim(test_vector, vector) <= threshold:
            print(words[i])


def search_similar(word_map, matrix, query, threshold=0.95):
    """
    Search through the matrix and get words that are similar to the give word
    using the embedding vectors.
    :param word_map:
    :param matrix:
    :param query:
    :param threshold:
    :return:
    """
    if query not in word_map:
        print(query, "not found")
        return None

    words = ['' for i in range(len(word_map))]
    for word in word_map:
        words[word_map[word]] = word

    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        if cossim(test_vector, vector) >= threshold:
            print(words[i])


def calculate_tf_weightings(input_folder):
    sentence_db_temp = []
    file_labels_temp = []

    idf_counts = defaultdict(int)
    # Document frequency loop
    files_to_process = os.listdir(input_folder)
    for filename in files_to_process:
        full_filename = os.path.join(input_folder, filename)
        with open(full_filename, encoding='utf-8', errors='ignore') as f:
            data = f.read()
            # get all sentences form the file
            current_sentences = nltk_tools.tokenize(data, 'sentence')

            for sentence in current_sentences:
                sentence_db_temp.append(sentence)
                file_labels_temp.append(filename)

                # remove newlines
                sentence = nltk_tools.clean_sentence(sentence)

                word_list = nltk_tools.tokenize(sentence, 'word')
                for word in set(word_list):
                    # add 1 for each sentence where the sentence is found.
                    idf_counts[word.lower()] += 1

    idf_weights = {}
    doc_count = len(sentence_db_temp)
    for word in idf_counts:
        # create a weight for very word that reflects how many sentences it appears in
        # Words appearing in nearly every sentence will be near 0. Words that are uncommon get higher weighting.
        idf_weights[word] = math.log(doc_count / idf_counts[word])

    return idf_weights, doc_count


def parse_weight_map(filename):
    weight_map = {}
    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split('\t')
            # try:
            #     weight_map[parts[0]] = float(parts[1])
            # except:
            #     print(line)
            #     print(parts)
            #     input()
    return weight_map


def write_weight_map(filename, weight_map):
    with open(filename, 'w', encoding='utf-8') as fout:
        for word in weight_map:
            outstr = word + '\t' + str(weight_map[word]) + '\n'
            fout.write(outstr)


def build_sentence_database(input_folder):
    sentence_db = []
    file_labels = []

    idf_counts = defaultdict(int)
    # Document frequency loop
    files_to_process = os.listdir(input_folder)
    for filename in files_to_process:
        full_filename = os.path.join(input_folder, filename)
        # attempting to process a directory gives an error.
        if os.path.isdir(full_filename):
            continue

        with open(full_filename, encoding='utf-8', errors='ignore') as f:
            data = f.read()
            # get all sentences form the file
            current_sentences = nltk_tools.tokenize(data, 'sentence')

            for sentence in current_sentences:
                sentence_db.append(sentence)
                file_labels.append(filename)

    return sentence_db, file_labels


def calculate_sentence_vectors_tfidf(input_folder, weight_map, word_map, matrix):
    """
    Maskes some use of these ideas
    https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3
    https://hackernoon.com/finding-the-most-important-sentences-using-nlp-tf-idf-3065028897a3

    :param input_folder:
    :param weight_map:
    :param word_map:
    :param matrix:
    :return:
    """
    sentence_db_raw, file_labels_raw = build_sentence_database(input_folder)
    doc_count = len(sentence_db_raw)

    sent_vectors_database = []      # database of sentence vectors
    sentence_database= []           # database list of processed sentences
    sentence_labels = []            # these file names will help identify sentences

    # for each sentence
    for i in range(doc_count):
        sentence = sentence_db_raw[i]
        sentence_label = file_labels_raw[i]

        sentence = nltk_tools.clean_sentence(sentence)

        word_list = nltk_tools.tokenize(sentence, 'word')
        word_list = nltk_tools.filter_stopwords_list_to_list(word_list)

        # discard any empty sentences (skip them)
        if len(word_list) <= 0:
            continue

        # caculate word vector
        sent_vec = reduce_sum_word_list_weighted(word_list, word_map, matrix, weight_map)

        # discard sentences that contain no words from the embedding database, not much can be done here.
        # examples 'H.R.3' and 'Janiyah'
        if np.linalg.norm(sent_vec) < 1e-6:
            continue

        # add the transformed data to the data storage variables
        sentence_database.append(sentence)
        sent_vectors_database.append(sent_vec)
        # add the filename to a list too
        sentence_labels.append(sentence_label)

    return sent_vectors_database, sentence_database, sentence_labels





# newline
