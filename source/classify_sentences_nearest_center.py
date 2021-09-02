"""
    This program uses cluster centers to classify new speech sentences.
"""
import embedding_tools as et
import option_helpers as opth
from script_options import classify_sentences_nearest_center_ops
import numpy as np
from scipy.spatial.distance import cosine


def load_cluster_center_vectors(cluster_centers_filename):
    center_vector_map = {}
    with open(cluster_centers_filename) as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            label = parts[0]
            vector_parts = parts[1:]
            vector = np.array(list(map(float, vector_parts)))
            center_vector_map[int(label)] = vector
    return center_vector_map


def classify_nearest_center(vector, center_map, distance=cosine):
    center_distances = []
    for current_cluster in center_map:
        center_vector = center_map[current_cluster]
        distance_value = distance(vector, center_vector)
        center_distances.append((distance_value, current_cluster))
    center_distances.sort()
    # print(center_distances)
    # print(center_distances[0])
    return center_distances[0][1]


def main():
    # ######### Parse and validate options ##########
    options = classify_sentences_nearest_center_ops()
    print_usage_func = opth.print_usage_maker('This is a tool for clusters sentences in an embedding space.',
                                              options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    # collect input command line arguments and validate
    argument_map = parse_function()
    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    output_filename = opth.validate_required('output', argument_map, print_usage_func)
    embedding_filename = opth.validate_required('embedding', argument_map, print_usage_func)
    weighting_filename = opth.validate_required('weightings', argument_map, print_usage_func)
    cluster_centers_filename = opth.validate_required('cluster_centers', argument_map, print_usage_func)

    # ########## Load Embeddings and Word Frequency Weightings ##########
    # load the word weighting map
    weight_map = et.parse_weight_map(weighting_filename)
    # load the word_index_map and the embedding matrix
    word_map, matrix = et.load_embeddings(embedding_filename)

    cluster_centers_map = load_cluster_center_vectors(cluster_centers_filename)

    # ########## Get Sentence and Sentence Vector Databases ##########
    results = et.calculate_sentence_vectors_tfidf(input_folder, weight_map, word_map, matrix)
    sent_vectors_database, sentence_database, sentence_labels = results

    sent_mat = np.array(sent_vectors_database)
    row_number = sent_mat.shape[0]

    with open(output_filename, 'w', encoding='utf-8') as file_out:
        header = ['Sentence', 'cluster', 'file']
        file_out.write(','.join(header) + '\n')

        for index in range(row_number):
            sentence_vector = sent_mat[index, :]
            class_label = classify_nearest_center(sentence_vector, cluster_centers_map)
            sentence_string = sentence_database[index].replace(',', ' ')
            outputs = [sentence_string, str(class_label), sentence_labels[index]]
            file_out.write(','.join(outputs) + '\n')


main()
