"""
    This program clusters sentences and writes the results to a labeled table.
"""
import embedding_tools as et
import option_helpers as opth
from script_options import summarize_clusters_core_sentences
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cosine
from collections import defaultdict


def get_cluster_index_map(cluster_labels):
    # Getting cluster subsets,
    cluster_map = defaultdict(list)
    for i in range(len(cluster_labels)):
        cluster_map[cluster_labels[i]].append(i)
    return cluster_map


def top_n_indexes(indexes, sent_mat, n):
    cluster_mat = sent_mat[np.array(indexes), :]
    mean_vector = cluster_mat.mean(axis=0)
    distances = []
    for index in indexes:
        vector = sent_mat[index, :]
        distance = cosine(vector, mean_vector)
        distances.append((distance, index))

    distances.sort(reverse=True)
    num_indexes = min(n, len(distances))
    top_indexes = [i for (d, i) in distances[:num_indexes]]
    return top_indexes


def main():
    # ######### Parse and validate options ##########
    options = summarize_clusters_core_sentences()
    print_usage_func = opth.print_usage_maker('This is a tool for summarizing clusters using core sentences.',
                                              options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    # collect input command line arguments and validate
    argument_map = parse_function()
    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    output_filename = opth.validate_required('output', argument_map, print_usage_func)
    embedding_filename = opth.validate_required('embedding', argument_map, print_usage_func)
    weighting_filename = opth.validate_required('weightings', argument_map, print_usage_func)
    num_clusters = opth.validate_required_int('num_clusters', argument_map, print_usage_func)
    top_n = opth.with_default_int('topN', argument_map, 5)

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
    # print("Clustering Complete")

    cluster_index_map = get_cluster_index_map(cluster_labels)
    # 'fcluster' omits 0 as a cluster label (cluster are 1 to k).
    cluster_keys = range(1, num_clusters + 1)

    output_data = []
    for current_cluster in cluster_keys:
        indexes = cluster_index_map[current_cluster]
        top_indexes = top_n_indexes(indexes, sent_mat, top_n)
        for index in top_indexes:
            sentence = sentence_database[index].replace(',', ' ')
            output_rec = [sentence, str(current_cluster), sentence_labels[index]]
            output_data.append(output_rec)

    header = ['sentence','cluster','speech']
    with open(output_filename, 'w') as fout:
        fout.write(','.join(header) + '\n')
        for output in output_data:
            fout.write(','.join(output) + '\n')


main()

# newline
