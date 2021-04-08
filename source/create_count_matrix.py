"""
    This program takes clustered sentences and creats a count matrix.
    The count matrix contains a row for each document (speech) and has
      a column for each cluster. The values are the number of sentences
      contained within the speech that belong to the given cluster.
"""
import option_helpers as opth
from script_options import create_count_matrix_ops
from collections import defaultdict


def read_names(filename):
    names = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            names.append(line)
    return names


def main():
    options = create_count_matrix_ops()
    program_description = 'Process a table of sentences with cluster ids and speeches\n' \
                          '     into a count matrix for regression analysis.'
    print_usage_func = opth.print_usage_maker(program_description, options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    argument_map = parse_function()

    input_file = opth.validate_required('input', argument_map, print_usage_func)
    output_file = opth.validate_required('output', argument_map, print_usage_func)
    name_list = opth.with_default('list', argument_map, None)

    clusters = set()
    speech_cluster_counts = {}

    # read the cluster file
    with open(input_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'Sentence,cluster,file':
                # skip header
                continue

            # split sentence record
            sentence, cluster, filename = line.split(',')

            # initialize cluster database
            if filename not in speech_cluster_counts:
                speech_cluster_counts[filename] = defaultdict(int)

            # add the cluster
            clusters.add(cluster)

            # add cluster count to speech
            speech_cluster_counts[filename][cluster] += 1

    clusters = list(clusters)
    cluster_ids = [int(cluster) for cluster in clusters]
    cluster_ids.sort()

    if name_list:
        filenames = read_names(name_list)
    else:
        filenames = list(speech_cluster_counts.keys())

    with open(output_file, 'w') as f:
        header = ['speech']
        for c in cluster_ids:
            header.append('c_' + str(c))

        f.write(','.join(header) + '\n')

        for filename in filenames:
            cluster_counts = speech_cluster_counts[filename]
            outputs = [filename]
            for c in cluster_ids:
                if str(c) not in cluster_counts:
                    outputs.append('0')
                else:
                    outputs.append(str(cluster_counts[str(c)]))
            f.write(','.join(outputs) + '\n')


main()

# newline
