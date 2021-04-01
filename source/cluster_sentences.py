"""
    This program clusters sentences and writes the results to a labeled table.
"""
import embedding_tools as et
import option_helpers as opth
from script_options import cluster_sentences_ops
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def main():
    # ######### Parse and validate options ##########
    options = cluster_sentences_ops()
    print_usage_func = opth.print_usage_maker('This is a tool for clusters sentences in an embedding space.',
                                              options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    # collect input command line arguments and validate
    argument_map = parse_function()
    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    output_filename = opth.validate_required('output', argument_map, print_usage_func)
    embedding_filename = opth.validate_required('embedding', argument_map, print_usage_func)
    weighting_filename = opth.validate_required('weightings', argument_map, print_usage_func)
    num_clusters = opth.validate_required_int('num_clusters', argument_map, print_usage_func)

    # ########## Load Embeddings and Word Frequency Weightings ##########
    # load the word weighting map
    weight_map = et.parse_weight_map(weighting_filename)
    # load the word_index_map and the embedding matrix
    word_map, matrix = et.load_embeddings(embedding_filename)

    # ########## Get Sentence and Sentence Vector Databases ##########
    results = et.calculate_sentence_vectors_tfidf(input_folder, weight_map, word_map, matrix)
    sent_vectors_database, sentence_database, sentence_labels = results

    # ########## Format and Numpy Matrix and Cluster ##########
    # convert to matrix and prepare to cluster
    sent_mat = np.array(sent_vectors_database)
    row_number = sent_mat.shape[0]
    # do the actual clustering, calculate distances, cluster linkage.
    y = pdist(sent_mat, 'cosine')  # define distance between points (sentence vectors)
    z = linkage(y, 'ward')  # define linkage, how to group points together

    # get the actual cluster labels
    cluster_labels = fcluster(z, num_clusters, criterion='maxclust')

    # open the output file
    # one of the sentences has a unicode that causes problems. make the output utf-8 too.
    file_out = open(output_filename, 'w', encoding='utf-8')
    header = ['Sentence', 'cluster', 'file']
    file_out.write(','.join(header) + '\n')

    for i in range(row_number):
        # sentences have commas, so take them out.
        sentence_string = sentence_database[i].replace(',', ' ')
        outputs = [sentence_string, str(cluster_labels[i]), sentence_labels[i]]
        # use 'join' to create the row string
        file_out.write(','.join(outputs) + '\n')
    file_out.close()


main()

# newline
