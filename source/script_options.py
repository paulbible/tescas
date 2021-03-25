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
    return options


def cluster_text_tool_ops():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'An input folder with each document represented as a single text file.'
    options['input']['input_name'] = '<input_folder>'
    options['output']['description'] = 'An output file name for the clustered sentence table.'
    # add custom options
    options['embedding'] = {
        'order': 3,
        'short': 'e',
        'long': 'embedding',
        'input_name': '<embedding_file>',
        'description': 'The file containing the vector space embedded vocabulary.',
        'optional': False
    }
    options['num_clusters'] = {
        'order': 4,
        'short': 'k',
        'long': 'num_clusters',
        'input_name': '<num_clusters>',
        'description': 'The number of sentence clusters to partition the sentences into.',
        'optional': False
    }
    options['stopfilter'] = {
        'order': 5,
        'short': 's',
        'long': 'stopfilter',
        'input_name': None,
        'description': 'Filter out stopwords (a, an ,the, etc.) before clustering',
        'optional': True
    }
    options['pos'] = {
        'order': 6,
        'short': 'p',
        'long': 'pos',
        'input_name': None,
        'description': 'Keep only words that are a major part of speech (Nouns, Verbs, Adjectives, and Adverbs).',
        'optional': True
    }
    options['tfidf'] = {
        'order': 6,
        'short': 't',
        'long': 'tfidf',
        'input_name': None,
        'description': 'Apply an inverse document frequency weighting. log(#sentences/#setnences with word)',
        'optional': True
    }
    return options
