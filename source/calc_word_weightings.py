"""
    This tool scans a corpus and calculates a weighting for each word.
    The tab delimited map is written to output
"""
import os
import string
import math
import nltk
from nltk_tools import tokenize, clean_sentence
import option_helpers as opth
from script_options import calc_word_weightings_ops
from collections import defaultdict
from embedding_tools import write_weight_map


def main():
    # get command line option descriptions and parse options.
    options = calc_word_weightings_ops()
    print_usage_func = opth.print_usage_maker('This is a tool for calculating a weight map based on a corpus.',
                                              options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    # collect input command line arguments and validate
    argument_map = parse_function()
    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    output_filename = opth.validate_required('output', argument_map, print_usage_func)
    # print(input_folder)
    # print(output_filename)

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
            current_sentences = tokenize(data, 'sentence')

            for sentence in current_sentences:
                sentence_db_temp.append(sentence)
                file_labels_temp.append(filename)

                # remove newlines
                sentence = clean_sentence(sentence)

                word_list = tokenize(sentence, 'word')
                for word in set(word_list):
                    # add 1 for each sentence where the sentence is found.
                    idf_counts[word.lower()] += 1

    idf_weights = {}
    doc_count = len(sentence_db_temp)
    for word in idf_counts:
        # create a weight for very word that reflects how many sentences it appears in
        # Words appearing in nearly every sentence will be near 0. Words that are uncommon get higher weighting.
        idf_weights[word] = math.log(doc_count / idf_counts[word])

    # write the weight map to file
    write_weight_map(output_filename, idf_weights)


main()
