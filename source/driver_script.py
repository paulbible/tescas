"""
    This is a driver script that will run the pipeline.
    This script can be edited to accommodate the user's environment.
"""
import subprocess
import os
# from datetime import datetime


def main():
    #############################################################
    # Set data set locations here.
    # use a unique folder for each modeling / training data set.
    data_set_folder = 'sotu'
    embedding = 'glove.6B.50d.txt'
    save_cluster_plot = True
    cluster_number = 15

    #############################################################
    # Configure which stages to run
    stages = 3      # total number of stages
    # # run all stages
    # stages_to_run = set(list(range(stages)))

    # set the following list to run specific parts of the pipeline
    stages_to_run = {2}

    #############################################################
    # Set up your environment here
    # your environment, use ../venv/Scripts/python.exe for pycharm virtual environment
    # if python 3 is in your path, you may use py, py3, python3, or just python depending on your installation.
    python_command = '../venv/Scripts/python.exe'
    R_command = 'C:\\Program Files\\R\\R-4.0.2\\bin\\x64\\Rscript.exe'

    # these folders should be fixed in the directory
    speeches_folder = '../data/speeches'
    weightings_folder = '../data/word_weightings'
    embeddings_folder = '../data/embeddings'
    images_foler = '../data/images'
    runs_folder = '../data/runs'

    speech_folder = os.path.join(speeches_folder, data_set_folder)
    weight_filename = data_set_folder + '_weights.txt'
    word_weight_file = os.path.join(weightings_folder, weight_filename)
    embedding_filename = os.path.join(embeddings_folder, embedding)
    cluster_plot_name = data_set_folder + '_cluster_plot.png'
    cluster_plot_filename = os.path.join(images_foler, cluster_plot_name)

    # now = datetime.now()
    # time_string = now.strftime('%Y-%m-%d_%H-%M')
    # output_folder_name = '_'.join([data_set_folder, time_string])
    # output_folder_path = os.path.join(runs_folder, output_folder_name)
    output_folder_path = os.path.join(runs_folder, data_set_folder)
    print(output_folder_path)

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    sentence_clusters_name = data_set_folder + '_clustered_sentences.csv'
    sentence_clusters_filename = os.path.join(output_folder_path, sentence_clusters_name)

    if 0 in stages_to_run:
        print('##### Calculating Work Weightings for speeches in', os.path.basename(speech_folder))
        args = [python_command, 'calc_word_weightings.py',
                '-i', speech_folder,
                '-o', word_weight_file]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 0')
        print()

    if 1 in stages_to_run:
        print('##### Clustering and making a plot', os.path.basename(speech_folder))
        args = [python_command, 'cluster_and_plot.py',
                '-i', speech_folder,
                '-e', embedding_filename,
                '-w', word_weight_file]
        if save_cluster_plot:
            args.extend(['-s', cluster_plot_filename])
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 1')
        print()

    if 2 in stages_to_run:
        print('##### Clustering Sentences in', os.path.basename(speech_folder))
        args = [python_command, 'cluster_sentences.py',
                '-i', speech_folder,
                '-o', sentence_clusters_filename,
                '-e', embedding_filename,
                '-w', word_weight_file,
                '-k', str(cluster_number)]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 2')
        print()



main()
