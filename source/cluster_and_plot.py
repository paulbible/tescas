"""
    A tool to cluster and plot sentences in a cluster.
    This tool help explore a cluster and pick a good k
    for the number of clusters.
"""
import embedding_tools as et
import option_helpers as opth
from script_options import cluster_and_plot_ops
from collections import defaultdict
from nltk.corpus import stopwords
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from collections import defaultdict
from matplotlib import pyplot as plt


def main():
    options = cluster_and_plot_ops()
    print_usage_func = opth.print_usage_maker('This is a tool for exploring clusters with plots.',
                                              options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    # collect input command line arguments and validate
    argument_map = parse_function()
    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    embedding_filename = opth.validate_required('embedding', argument_map, print_usage_func)
    weighting_filename = opth.validate_required('weightings', argument_map, print_usage_func)
    print(input_folder)
    print(embedding_filename)
    print(weighting_filename)
    weight_map = et.parse_weight_map(weighting_filename)


main()

