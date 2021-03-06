"""
    A tool to cluster and plot sentences in a cluster.
    This tool help explore a cluster and pick a good k
    for the number of clusters.
"""
import embedding_tools as et
import option_helpers as opth
from script_options import cluster_and_plot_ops
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt


def main():
    # ######### Parse and validate options ##########
    options = cluster_and_plot_ops()
    print_usage_func = opth.print_usage_maker('This is a tool for exploring clusters with plots.',
                                              options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    # collect input command line arguments and validate
    argument_map = parse_function()
    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    embedding_filename = opth.validate_required('embedding', argument_map, print_usage_func)
    weighting_filename = opth.validate_required('weightings', argument_map, print_usage_func)
    save_file = opth.with_default('save', argument_map, None)

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
    # do the actual clustering, calculate distances, cluster linkage.
    y = pdist(sent_mat, 'cosine')  # define distance between points (sentence vectors)
    z = linkage(y, 'ward')  # define linkage, how to group points together

    # ########## Plot and Save Figure ##########
    last = z[-50:, 2]
    last_rev = last[::-1]
    indexes = np.arange(1, len(last) + 1)
    plt.plot(indexes, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(indexes[:-2] + 1, acceleration_rev)

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    else:
        plt.show()


main()

