import tkinter as tk
import os
import datetime
from tkinter import Menu
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import embedding_tools as et
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import pandas
import pandastable
# from threading import *


class TescasGui:
    def __init__(self):
        # data storage
        self.messages = []
        self.data_set_directory = None
        self.embedding_filename = None
        # embedding variables
        self.word_map = None
        self.embed_matrix_vectors = None
        # data set varaibles
        self.word_weightings = None
        self.sentence_vectors_database = None
        self.sentence_database = None
        self.sentence_labels = None
        self.sentence_matrix = None

        # tkinter app
        self.root = tk.Tk()
        self.root.title('TESCAS')

        self.fig_width = 4
        self.fig_height = 4

        # ### Initialize images for buttons #################
        self.embedding_icon_image_raw = Image.open('../data/resources/images/vectors.png')
        self.embedding_icon_image_raw = self.embedding_icon_image_raw.resize((48, 48), Image.ANTIALIAS)
        self.embedding_icon_image = ImageTk.PhotoImage(self.embedding_icon_image_raw)

        self.loading_icon_image_raw = Image.open('../data/resources/images/speech_folder.png')
        self.loading_icon_image_raw = self.loading_icon_image_raw.resize((48, 48), Image.ANTIALIAS)
        self.loading_icon_image = ImageTk.PhotoImage(self.loading_icon_image_raw)

        self.cluster_icon_image_raw = Image.open('../data/resources/images/cluster_data.png')
        self.cluster_icon_image_raw = self.cluster_icon_image_raw.resize((48, 48), Image.ANTIALIAS)
        self.cluster_icon_image = ImageTk.PhotoImage(self.cluster_icon_image_raw)

        # summary tool
        self.summary_icon_image_raw = Image.open('../data/resources/images/summary_clipboard.png')
        self.summary_icon_image_raw = self.summary_icon_image_raw.resize((48, 48), Image.ANTIALIAS)
        self.summary_icon_image = ImageTk.PhotoImage(self.summary_icon_image_raw)

        # Create top level frame ###############
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X)

        # ### Button Frame, left side tool bar #################
        button_frame = tk.Frame(top_frame, highlightbackground='black', highlightthickness=1)
        button_frame.pack(side=tk.LEFT, fill=tk.Y)

        load_embedding_button = tk.Button(button_frame, text='embedding', image=self.embedding_icon_image,
                                          compound="top",
                                          command=self.load_embedding_file)
        load_embedding_button.pack(side=tk.TOP, fill=tk.X)

        load_button = tk.Button(button_frame, text='load', image=self.loading_icon_image, compound="top",
                           command=self.load_data_set)
        load_button.pack(side=tk.TOP, fill=tk.X)

        cluster_button = tk.Button(button_frame, text='cluster', image=self.cluster_icon_image, compound="top",
                                   command=self.cluster_and_plot)
        cluster_button.pack(side=tk.TOP, fill=tk.X)

        summary_button = tk.Button(button_frame, text='Summary', image=self.summary_icon_image, compound="top",
                                   command=self.cluster_and_plot)
        summary_button.pack(side=tk.TOP, fill=tk.X)

        # ### Middle frame content #################
        content_frame = tk.Frame(top_frame)
        content_frame.pack(side=tk.LEFT)
        # create figure
        self.figure = Figure(figsize=(self.fig_width, self.fig_height))
        # plt = figure.add_subplot(111)
        # data = np.random.rand(100,2)
        # plt.scatter(data[:, 0], data[:, 1])

        self.canvas = FigureCanvasTkAgg(self.figure, master=content_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # ### Configure the bottom of application status bar. #################
        # Setting pack of the containing frame is important
        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # This StringVar acts as an updatable string connected to a label.
        # data set label
        self.data_set_label_text = tk.StringVar()
        self.data_set_label_text.set('Data set:')
        self.dataset_label = tk.Label(status_frame, textvariable=self.data_set_label_text,
                                      anchor=tk.W,
                                 highlightbackground='black', highlightthickness=1)
        self.dataset_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # embedding status label
        # data set label
        self.embedding_label_text = tk.StringVar()
        self.embedding_label_text.set('Embedding:')
        self.embedding_label = tk.Label(status_frame, textvariable=self.embedding_label_text,
                                      highlightbackground='black', highlightthickness=1)
        self.embedding_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # doc count label
        self.doc_count_label_text = tk.StringVar()
        self.doc_count_label_text.set('Documents: 0')
        doc_count_label = tk.Label(status_frame, textvariable=self.doc_count_label_text,
                                   highlightbackground='black', highlightthickness=1)
        doc_count_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # sentence count label
        self.sentence_count_label_text = tk.StringVar()
        self.sentence_count_label_text.set('Sentences: 0')
        sentence_count_label = tk.Label(status_frame, textvariable=self.sentence_count_label_text,
                                        anchor=tk.E,
                                        highlightbackground='black', highlightthickness=1)
        sentence_count_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ### Message panel. #################
        # bottom frame content
        text_area_frame = tk.Frame(self.root)
        text_area_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)

        self.text_area = ScrolledText(text_area_frame, height=10)
        self.text_area.pack(fill=tk.BOTH)
        # disable to make read only
        self.text_area.config(state=tk.DISABLED)

        # custom initialization
        self.initialize()

    # ###  Helper Functions #####################
    @staticmethod
    def now_string():
        '''
        Modified from https://www.programiz.com/python-programming/datetime/current-datetime
        :return:
        '''
        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        return dt_string

    def start_timestamp(self):
        dt_string = TescasGui.now_string()
        self.insert_text('-- Started at %s' % dt_string)

    def end_timestamp(self):
        dt_string = TescasGui.now_string()
        self.insert_text('-- Finished at %s' % dt_string)
        self.insert_text('Done')

    def initialize(self):
        self.insert_text('Welcome to TESCAS')

    def insert_text(self, message, newline=True):
        self.text_area.config(state=tk.NORMAL)
        new_message = message
        if newline:
            new_message += '\n'
        self.text_area.insert(tk.INSERT, new_message)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview_moveto(1.0)

    # ### Other Helpers ##################
    def has_data_set(self):
        return self.data_set_directory

    def has_embedding_file(self):
        return self.embedding_filename

    def update_data_set_label(self):
        if self.has_data_set():
            data_name = os.path.basename(self.data_set_directory)
            self.data_set_label_text.set('Data set: %s' % data_name)

    def update_doc_count(self, count):
        if self.has_data_set():
            self.doc_count_label_text.set('Documents: %s' % str(count))

    def update_embedding_label(self):
        if self.has_embedding_file():
            base_name = os.path.basename(self.embedding_filename)
            self.embedding_label_text.set('Embedding: %s' % base_name)

    def update_sentence_count(self, count):
        self.sentence_count_label_text.set('Sentences: %s' % str(count))

    def build_sentence_database(self):
        '''
        This function should be called to recreate the sentence database for a data set
        :return:
        '''
        if self.has_data_set() and self.has_embedding_file():
            base_name = os.path.basename(self.data_set_directory)
            self.insert_text('Begin Building Sentence Embedding Databse for %s ...' % base_name)
            self.start_timestamp()
            self.root.update_idletasks()
            results = et.calculate_sentence_vectors_tfidf(self.data_set_directory,
                                                          self.word_weightings,
                                                          self.word_map,
                                                          self.embed_matrix_vectors)
            self.sentence_vectors_database, self.sentence_database, self.sentence_labels = results
            self.end_timestamp()
            self.root.update_idletasks()
            self.update_sentence_count(len(self.sentence_database))
            self.sentence_matrix = np.array(self.sentence_vectors_database)

    # ### Button commands ##############
    def load_data_set(self):
        filepath = askdirectory(title='Choose your data set')
        if filepath:
            print(filepath)
            self.insert_text('Accessing %s' % filepath)
            self.data_set_directory = filepath
            self.update_data_set_label()

            filenames = os.listdir(filepath)
            base_name = os.path.basename(filepath)
            doc_count = len(filenames)
            self.update_doc_count(doc_count)
            self.insert_text('Documents found: %s' % str(doc_count))

            # Word Weighting Calculation
            self.insert_text('Begin Word Weighting Calculation for %s ...' % base_name)
            self.start_timestamp()
            # call this to update the GUI while processing
            self.root.update_idletasks()
            self.word_weightings, sentence_count = et.calculate_tf_weightings(filepath)
            self.insert_text('-- Word Weightings calculated for %s' % base_name)
            self.end_timestamp()
            self.root.update_idletasks()

            # attempt to build sentence database
            self.build_sentence_database()

    def load_embedding_file(self):
        filepath = askopenfilename(title='Choose your embedding file')
        if filepath:
            print(filepath)
            self.insert_text('Accessing embedding file %s' % filepath)
            self.embedding_filename = filepath
            base_name = os.path.basename(filepath)
            self.update_embedding_label()

            # load the word embedding vectors
            self.insert_text('Begin loading embedding vectors %s ...' % base_name)
            self.start_timestamp()
            self.root.update_idletasks()
            self.word_map, self.embed_matrix_vectors = et.load_embeddings(self.embedding_filename)
            self.end_timestamp()
            self.root.update_idletasks()

            # attempt to build sentence database
            self.build_sentence_database()

    def cluster_and_plot(self):
        if self.sentence_matrix is not None:
            self.insert_text('Begin clustering data set ...')
            self.start_timestamp()
            self.root.update_idletasks()
            y = pdist(self.sentence_matrix, 'cosine')  # define distance between points (sentence vectors)
            z = linkage(y, 'ward')  # define linkage, how to group points together
            self.end_timestamp()

            # ########## Plot and Save Figure ##########
            last = z[-50:, 2]
            last_rev = last[::-1]
            indexes = np.arange(1, len(last) + 1)
            plt = self.figure.add_subplot(111)
            plt.plot(indexes, last_rev)

            acceleration = np.diff(last, 2)  # 2nd derivative of the distances
            acceleration_rev = acceleration[::-1]
            plt.plot(indexes[:-2] + 1, acceleration_rev)

            plt.set_label('Cluster SSE')
            plt.set(xlabel='Cluster Number', ylabel='SSE')

            # self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            self.canvas.draw()


gui = TescasGui()
tk.mainloop()
