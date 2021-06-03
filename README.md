# TESCAS
This is a Tool for Embedding Space Cluster Analysis of Speeches (TESCAS). It was developed by William C. McAuley and Paul W. Bible in Python 3.

## Dependencies
The TESCAS tool requires the following software and packages. Once python is installed the python pakages can be installed uing `pip`.

1) [Python 3](https://www.python.org/) (developed for Python 3.7)
2) [numpy](https://numpy.org/), `pip install numpy`
3) [matplotlib](https://matplotlib.org//), `python -m pip install -U matplotlib`
4) [NLTK](https://www.nltk.org/), `pip install --user -U nltk`
5) [scipy](https://www.scipy.org/), `python -m pip install --user scipy`
6) [scikit-learn](https://scikit-learn.org/stable/index.html) `pip install -U scikit-learn`

## Usage
To use TESCAS, place your text speeches into a folder then modify the `driver_script.py`.

Check that the path to your python command is correct in `driver_script.py`, and modify the section that looks like this:

```
    #############################################################
    # Set data set locations here.
    # use a unique folder for each modeling / training data set.
    data_set_folder = 'sotu'
    embedding = 'glove.6B.50d.txt'
    save_cluster_plot = True
    cluster_number = 15
```

### Configure which stages to run in the processing pipeline

#### The pipeline stages are:
* **0** Calculate word weightings
* **1** Explore the corpus clustering plot, helps choose number of clusters
* **2** Perform sentence clustering
* **3** Summarize the cluster content using pair word usage statistics.
* **4** Summarize the cluster content using core sentences.
* **5** Create a matrix of cluster counts for each speech.
* **6** Extract the cluster centers for use in a classifier

Either uncomment this line to run all stages:
```
stages_to_run = set(list(range(stages)))
```
Or set the specific stages to run here:
```
stages_to_run = {0, 2, 3}
```

