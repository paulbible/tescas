"""
    This tool will calculate word pair statistics to summarize clusters.
"""
import option_helpers as opth
from script_options import summarize_clusters_word_pairs_ops
from nltk_tools import stops, tokenize, clean_sentence
from collections import defaultdict


def create_word_pair_databases(input_file):
    cluster_word_database = {}
    cluster_word_total = defaultdict(int)

    all_data_word_counts = defaultdict(int)
    total_words = 0

    pair_word_noise = {"ca_n\'t", "wo_n't"}

    with open(input_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'Sentence,cluster,file':
                # skip header
                continue
            # print(line)

            # split sentence record
            sentence, cluster, filename = line.split(',')
            if cluster == '':
                continue

            # initialize cluster databases
            if cluster not in cluster_word_database:
                cluster_word_database[cluster] = defaultdict(int)

            word_list = tokenize(sentence, 'word')
            word_count = len(word_list)
            # skip if there are not pairs in the sentence
            if word_count < 2:
                continue

            for i in range(word_count - 2):
                word_1 = word_list[i].lower()
                word_2 = word_list[i + 1].lower()
                word_1 = clean_sentence(word_1)
                word_2 = clean_sentence(word_2)
                if len(word_1) == 0 or word_1 in stops:
                    continue

                if len(word_2) == 0 or word_2 in stops:
                    continue

                pair_key = word_1 + '_' + word_2

                if pair_key in pair_word_noise:
                    continue

                cluster_word_database[cluster][pair_key] += 1
                cluster_word_total[cluster] += 1
                all_data_word_counts[pair_key] += 1
                total_words += 1
    return cluster_word_database, cluster_word_total, all_data_word_counts, total_words


def main():
    options = summarize_clusters_word_pairs_ops()
    program_description = 'Summarize the word distributions of sentence clusters\n' \
                          '    using frequent word pairs.'
    print_usage_func = opth.print_usage_maker(program_description, options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    argument_map = parse_function()

    input_file = opth.validate_required('input', argument_map, print_usage_func)
    output_file = opth.validate_required('output', argument_map, print_usage_func)
    count_limit = opth.with_default_int('cutoff', argument_map, 1)
    percent_limit = opth.with_default_float('percent', argument_map, 0.0)
    top_n = opth.with_default_int('topN', argument_map, 10)

    result = create_word_pair_databases(input_file)
    cluster_word_database, cluster_word_totals, all_data_word_counts, total_words = result

    ################
    # main filtering
    header = ['cluster', 'word', 'count', 'cluster_freq', 'total_freq']
    output_rows = []

    for cluster in cluster_word_totals:
        # total pairs in the cluster
        total = float(cluster_word_totals[cluster])
        # mapping from a pair to its count
        word_map = cluster_word_database[cluster]

        # sort all cluster word pairs by frequency
        sorted_pairs = sorted(word_map.items(), key=lambda item: item[1], reverse=True)

        # limit to report
        limit = min(top_n, len(word_map))
        # current count of reported pairs
        count = 0
        for k, v in sorted_pairs:
            cluster_frequency = v / total
            dataset_frequency = all_data_word_counts[k] / total_words

            if v < count_limit:
                continue

            if cluster_frequency < percent_limit:
                continue

            # if filters pass, add word pair data to output rows
            outs = [cluster, k, str(v), str(round(cluster_frequency, 5)),
                    str(round(dataset_frequency, 5))]
            output_rows.append(outs)
            count += 1
            if count >= limit:
                break

    with open(output_file, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in output_rows:
            f.write(','.join(row) + '\n')


main()

# newline
