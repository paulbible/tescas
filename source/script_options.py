"""
    a python file to collect command options for each script
    Note: maybe not the best fix for this, but it reduces clutter.
"""
import option_helpers as opth


def calc_word_weightings_ops():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'An input folder with each document represented as a single text file.'
    options['input']['input_name'] = '<input_folder>'
    options['output']['description'] = 'An output file name for the tab-delimited, 2-column weightings map.'
    return options


def cluster_and_plot_ops():
    # Create some options for the tool.
    options = opth.default_option_map_input()
    # modify default descriptions for input and output
    options['input']['description'] = 'An input folder with each document represented as a single text file.'
    options['input']['input_name'] = '<input_folder>'
    options['embedding'] = {
        'order': 2,
        'short': 'e',
        'long': 'embedding',
        'input_name': '<embedding_file>',
        'description': 'The file containing the vector space embedded vocabulary.',
        'optional': False
    }
    options['weightings'] = {
        'order': 3,
        'short': 'w',
        'long': 'weightings',
        'input_name': '<weightings_file>',
        'description': 'The file containing the vocabulary word weightings.',
        'optional': False
    }
    options['save'] = {
        'order': 4,
        'short': 's',
        'long': 'save',
        'input_name': '<image_file>',
        'description': 'A file to save the plot as an image (png).',
        'optional': True
    }
    return options


def cluster_sentences_ops():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'An input folder with each document represented as a single text file.'
    options['input']['input_name'] = '<input_folder>'
    options['output']['description'] = 'An output file name for the clustered sentence table.'
    options['embedding'] = {
        'order': 3,
        'short': 'e',
        'long': 'embedding',
        'input_name': '<embedding_file>',
        'description': 'The file containing the vector space embedded vocabulary.',
        'optional': False
    }
    options['weightings'] = {
        'order': 4,
        'short': 'w',
        'long': 'weightings',
        'input_name': '<weightings_file>',
        'description': 'The file containing the vocabulary word weightings.',
        'optional': False
    }
    options['num_clusters'] = {
        'order': 5,
        'short': 'k',
        'long': 'num_clusters',
        'input_name': '<num_clusters>',
        'description': 'The number of sentence clusters to partition the sentences into.',
        'optional': False
    }
    return options


def summarize_clusters_word_pairs_ops():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'A comma separated file of sentences as rows with clusters labeled.'
    options['input']['input_name'] = '<cluster_file>'
    options['output']['description'] = 'An output file name for the word distribution data (CSV).'
    # add custom options
    options['topN'] = {
        'order': 3,
        'short': 'n',
        'long': 'topN',
        'input_name': '<number>',
        'description': 'Return the N word pairs with the highest count for each cluster. (default: 10)',
        'optional': True
    }
    options['cutoff'] = {
        'order': 4,
        'short': 'c',
        'long': 'cutoff',
        'input_name': '<count_limit>',
        'description': 'Filter word pairs with less that <count_limit> occurrences. (default: 1)',
        'optional': True
    }
    options['percent'] = {
        'order': 5,
        'short': 'p',
        'long': 'percent',
        'input_name': '<percent_limit>',
        'description': 'Filter word pairs that account for less than <percent_limit> of cluster words (default: 0.0)',
        'optional': True
    }
    return options


def summarize_clusters_core_sentences():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'An input folder with each document represented as a single text file.'
    options['input']['input_name'] = '<input_folder>'
    options['output']['description'] = 'An output file name for the core sentence data (CSV).'
    options['embedding'] = {
        'order': 3,
        'short': 'e',
        'long': 'embedding',
        'input_name': '<embedding_file>',
        'description': 'The file containing the vector space embedded vocabulary.',
        'optional': False
    }
    options['weightings'] = {
        'order': 4,
        'short': 'w',
        'long': 'weightings',
        'input_name': '<weightings_file>',
        'description': 'The file containing the vocabulary word weightings.',
        'optional': False
    }
    options['num_clusters'] = {
        'order': 5,
        'short': 'k',
        'long': 'num_clusters',
        'input_name': '<num_clusters>',
        'description': 'The number of sentence clusters to partition the sentences into.',
        'optional': False
    }
    options['topN'] = {
        'order': 6,
        'short': 'n',
        'long': 'topN',
        'input_name': '<number>',
        'description': 'Return the N sentences closest to the cluster mean. (default: 5)',
        'optional': True
    }
    return options


def create_count_matrix_ops():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'A comma separated file of sentences as rows with clusters labeled.'
    options['input']['input_name'] = '<cluster_file>'
    options['output']['description'] = 'An output file name for count matrix data (CSV).'
    # add custom options
    options['list'] = {
        'order': 3,
        'short': 'l',
        'long': 'list',
        'input_name': '<list>',
        'description': 'A list of names to order the output. These are a text file of file names.',
        'optional': True
    }
    return options

# newline
