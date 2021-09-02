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
    # data_set_folder = 'ATE'
    test_set_folder = 'test'

    # a text file providing an ordered list of speeches
    # this file should be in the speeches folder
    data_set_document_list = 'sotu_order_list.txt'
    # data_set_document_list = 'ATE_order_list.txt'
    embedding = 'glove.6B.50d.txt'
    save_cluster_plot = True
    cluster_number = 123
    # cluster_number = 23
    # cluster_number = 16

    #############################################################
    # Configure which stages to run in the processing pipeline
    stages = 8      # total number of stages
    # # The pipeline stages are:
    # 0 Calculate word weightings
    # 1 Explore the corpus clustering plot, helps choose number of clusters
    # 2 Perform sentence clustering
    # 3 Summarize the cluster content using pair word usage statistics.
    # 4 Summarize the cluster content using core sentences
    # 5 Convert clustered sentences into cluster count matrix
    # 6 Extract central vectors for nearest neighbor classification.
    # 7 Classify test data with cluster centers.

    # # run all stages
    # stages_to_run = set(list(range(stages)))

    # set the following list to run specific parts of the pipeline
    stages_to_run = {3, 4}

    #############################################################
    # Set up your environment here
    # your environment, use ../venv/Scripts/python.exe for pycharm virtual environment
    # if python 3 is in your path, you may use py, py3, python3, or just python depending on your installation.
    python_command = '../venv/Scripts/python.exe'
    R_command = 'C:\\Program Files\\R\\R-4.0.2\\bin\\x64\\Rscript.exe'

    #############################################################
    # Fixed folders and automatically created files
    # Do not modify.
    # these folders should be fixed in the directory
    speeches_folder = '../data/speeches'
    weightings_folder = '../data/word_weightings'
    embeddings_folder = '../data/embeddings'
    images_foler = '../data/images'
    runs_folder = '../data/runs'

    # these files and folder are generated specific to a data set
    speech_folder = os.path.join(speeches_folder, data_set_folder)
    speech_list_filename = os.path.join(speeches_folder, data_set_document_list)
    weight_filename = data_set_folder + '_weights.txt'
    word_weight_file = os.path.join(weightings_folder, weight_filename)
    embedding_filename = os.path.join(embeddings_folder, embedding)
    cluster_plot_name = data_set_folder + '_cluster_plot.png'
    cluster_plot_filename = os.path.join(images_foler, cluster_plot_name)

    # now = datetime.now()
    # time_string = now.strftime('%Y-%m-%d_%H-%M')
    # output_folder_name = '_'.join([data_set_folder, time_string])
    # output_folder_path = os.path.join(runs_folder, output_folder_name)
    run_output_folder_path = os.path.join(runs_folder, data_set_folder)
    print(run_output_folder_path)

    if not os.path.isdir(run_output_folder_path):
        os.mkdir(run_output_folder_path)

    # cluster output data files
    sentence_clusters_name = data_set_folder + '_clustered_sentences.csv'
    sentence_clusters_filename = os.path.join(run_output_folder_path, sentence_clusters_name)
    word_pairs_summary_name = data_set_folder + '_cluster_word_pairs.csv'
    word_pairs_summary_filename = os.path.join(run_output_folder_path, word_pairs_summary_name)
    core_sentence_summary_name = data_set_folder + '_cluster_core_sentences.csv'
    core_sentence_summary_filename = os.path.join(run_output_folder_path, core_sentence_summary_name)

    # regression files
    count_matrix_name = data_set_folder + '_cluster_count_matrix.csv'
    count_matrix_filename = os.path.join(run_output_folder_path, count_matrix_name)

    # Classification files, center vectors
    central_vectors_name = data_set_folder + '_center_vectors.csv'
    central_vectors_filename = os.path.join(run_output_folder_path, central_vectors_name)

    test_folder = os.path.join(speeches_folder, test_set_folder)

    test_sentence_clusters_name = test_set_folder + '_clustered_sentences.csv'
    test_sentence_clusters_filename = os.path.join(run_output_folder_path, test_sentence_clusters_name)


    #########################################################################################
    ## BEGIN PIPELINE

    ######################################
    # ##### Stage 0: Word Weightings #####
    if 0 in stages_to_run:
        print('##### Calculating Work Weightings for speeches in', os.path.basename(speech_folder))
        args = [python_command, 'calc_word_weightings.py',
                '-i', speech_folder,
                '-o', word_weight_file]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 0')
        print()

    #########################################
    # ##### Stage 1: Cluster Jump Plot  #####
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

    #########################################
    # ##### Stage 2: Cluster Sentences  #####
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

    ######################################################
    # ##### Stage 3: Summarize Clusters, Word Pairs  #####
    if 3 in stages_to_run:
        print('##### Analyzing word pair distributions for clusters in', os.path.basename(speech_folder))
        args = [python_command, 'summarize_clusters_word_pairs.py',
                '-i', sentence_clusters_filename,
                '-o', word_pairs_summary_filename,
                '-n', str(15)]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 3')
        print()

    #########################################################
    # ##### Stage 4: Summarize Clusters, Core Sentences #####
    if 4 in stages_to_run:
        print('##### Summarizing Clusters with Core Sentences in ', os.path.basename(speech_folder))
        # speech_list_filename
        args = [python_command, 'summarize_clusters_core_sentences.py',
                '-i', speech_folder,
                '-o', core_sentence_summary_filename,
                '-e', embedding_filename,
                '-w', word_weight_file,
                '-k', str(cluster_number),
                '-n', str(15)]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 4')
        print()

    ######################################################
    # ##### Stage 5: Create Cluster count matrix     #####
    if 5 in stages_to_run:
        print('##### Creating cluster count matrix for ', os.path.basename(speech_folder))
        # speech_list_filename
        args = [python_command, 'create_count_matrix.py',
                '-i', sentence_clusters_filename,
                '-o', count_matrix_filename,
                '-l', speech_list_filename]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 5')
        print()

    ##############################################
    # ##### Stage 6: Extract Cluster Centers #####
    if 6 in stages_to_run:
        print('##### Extracting Cluster Center vectors ', os.path.basename(speech_folder))
        # speech_list_filename
        args = [python_command, 'extract_cluster_centers.py',
                '-i', speech_folder,
                '-o', central_vectors_filename,
                '-e', embedding_filename,
                '-w', word_weight_file,
                '-k', str(cluster_number)]
        print('## running: ', ' '.join(args))
        subprocess.run(args)
        print('## completed stage 6')
        print()

    ###############################################################
    # ##### Stage 7: Classify Sentences with Cluster Centers  #####
    if 7 in stages_to_run:
        print('##### Classifying Sentences in', os.path.basename(test_folder))
        args = [python_command, 'classify_sentences_nearest_center.py',
                '-i', test_folder,
                '-o', test_sentence_clusters_filename,
                '-e', embedding_filename,
                '-w', word_weight_file,
                '-c', central_vectors_filename]
        print('## running: ', ' '.join(args))
        # subprocess.run(args)
        print('## completed stage 2')
        print()



main()
