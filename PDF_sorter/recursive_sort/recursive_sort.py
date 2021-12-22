import numpy as np
import networkx as nx
import os
import shutil
from shutil import copyfile
import community.community_louvain as community_louvain

import PDF_sorter.utils.utils as utils


def recursive_sorter_from_graph(
    G,
    PDF_DIR,
    DESTINATION_DIR,
    n_largest_names,
    n_largest_description,
    min_graph_size,
    partition_resolution,
    word_list,
    iteration,
    max_depth,
):

    # if the graph is too small to partition stop here
    if (len(G) < min_graph_size) | (iteration + 1 > max_depth):

        move_articles(G, PDF_DIR, DESTINATION_DIR)

        # text file description of the folder
        utils.write_description(DESTINATION_DIR, word_list)

        return

    for node in G.nodes():
        G.nodes[node]["partition"] = np.nan

    # partition the graph
    partition = community_louvain.best_partition(G, resolution=partition_resolution)

    for node in G.nodes():
        G.nodes[node]["partition"] = partition[node]

    word_prob = utils.prob_in_communities(G)
    partition = nx.get_node_attributes(G, "partition")
    partition_list = np.unique([partition[k] for k in partition.keys()])

    # pour éviter les betises
    word_prob += 1e-10

    # prepare the folders
    if os.path.exists(DESTINATION_DIR):
        shutil.rmtree(DESTINATION_DIR)

    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)

    # run the partition proper
    for partition_number in partition_list:

        # need to string the name
        partition_name = str(int(partition_number))
        s = utils.get_unique_words(word_prob, partition_name)
        name_frame = s.nlargest(n_largest_names, columns=0)

        # make the folder
        folder_name = ""
        for ind in name_frame.index:
            folder_name += ind + "_"

        word_frame = s.nlargest(n_largest_description, columns=0)
        word_list = [i for i in word_frame.index]

        if not os.path.exists(os.path.join(DESTINATION_DIR, folder_name)):
            os.mkdir(os.path.join(DESTINATION_DIR, folder_name))

        LOCAL_DESTINATION_DIR = os.path.join(DESTINATION_DIR, folder_name)

        # make local graph
        subgraph = utils.get_subgraph(G, "partition", partition_number)

        recursive_sorter_from_graph(
            G=subgraph,
            PDF_DIR=PDF_DIR,
            DESTINATION_DIR=LOCAL_DESTINATION_DIR,
            n_largest_names=n_largest_names,
            n_largest_description=n_largest_description,
            min_graph_size=min_graph_size,
            partition_resolution=partition_resolution,
            word_list=word_list,
            iteration=iteration + 1,
            max_depth=max_depth,
        )

    return


def move_articles(G, PDF_DIR, DESTINATION_DIR):

    for article_name in G.nodes():

        src = os.path.join(PDF_DIR, article_name)
        dst = os.path.join(DESTINATION_DIR, article_name)

        copyfile(src, dst)

    return


def recursive_sort_papers_from_dict(
    DICTNAME,
    DICTDIR,
    PDF_DIR,
    SORTED_DIR,
    max_common_words,
    n_largest_names,
    partition_resolution,
    n_largest_description,
    min_graph_size,
    no_keywords=["http", "ncbi", "experi", "biorxiv", "pubm", "elsevi", "refhub"],
    iteration=0,
    max_depth=5,
):

    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open(os.path.join(DICTDIR, DICTNAME), "rb") as fp:
        paper_dictionnary = pickle.load(fp)

    word_count = utils.make_word_list(paper_dictionnary)
    no_keywords += utils.get_most_common_words(word_count, max_common_words)

    G = utils.make_graph_from_dict(paper_dictionnary, no_keywords)

    partition = community_louvain.best_partition(G, resolution=partition_resolution)

    for node in G.nodes():
        G.nodes[node]["partition"] = partition[node]

    recursive_sorter_from_graph(
        G,
        PDF_DIR=PDF_DIR,
        DESTINATION_DIR=SORTED_DIR,
        n_largest_names=n_largest_names,
        n_largest_description=n_largest_description,
        min_graph_size=min_graph_size,
        partition_resolution=partition_resolution,
        word_list=[],
        iteration=iteration,
        max_depth=max_depth,
    )

    return

def recursive_sort_papers_based_on_contents(
    PDF_DIR: str,
    SORTED_DIR: str,
    max_common_words: int,
    num_words: int,
    num_topics: int,
    n_largest_names: int,
    partition_resolution: float,
    n_largest_description: int,
    SAVEDICT: bool,
    DICTDIR: str,
    min_graph_size: int,
    no_keywords=["http", "ncbi", "experi", "biorxiv", "pubm", "elsevi", "refhub"],
    iteration=0,
    max_depth=5,
):

    """
    Recursive sort from PDFs contained in a folder.
    Generates a new folder for each of the topics.
    
    Parameters:
    -----------
     - PDF_DIR: str
        Path to the folder containing the PDFs.
     - SORTED_DIR: str
        Path to the folder where the sorted PDFs will be stored.
     - max_common_words: int
        Number of most common words to be removed from the analysis.
     - num_words: int
        Number of words to be used for the topic modeling.
     - num_topics: int
        Number of topics to be used for the topic modeling.
     - n_largest_names: int
        Number of most common words to be used for the description of the folder.
     - partition_resolution: float
        Resolution of the partition in the Louvain algorithm.
     - n_largest_description: int
        Number of most common words to be used for the description of the folder.
     - SAVEDICT: bool
        Whether to save the dictionnary of the papers.
     - DICTDIR: str
        Path to the folder where the dictionnary will be stored.
     - min_graph_size: int
        Minimum size of the graph to be considered in the sub-partitions.
     - no_keywords: list
        List of keywords to be removed from the analysis.
     - iteration: int
        Initial iteration number.
     - max_depth: int
        Maximum depth of the recursion.
        
    Returns:
    --------
     - None
    """

    paper_dictionnary = utils.make_dict_from_papers(PDF_DIR, num_words, num_topics)

    if SAVEDICT:
        utils.savedict(paper_dictionnary, DICTDIR)

    word_count = utils.make_word_list(paper_dictionnary)
    no_keywords += utils.get_most_common_words(word_count, max_common_words)

    G = utils.make_graph_from_dict(paper_dictionnary, no_keywords)

    partition = community_louvain.best_partition(G, resolution=partition_resolution)

    for node in G.nodes():
        G.nodes[node]["partition"] = partition[node]

    recursive_sorter_from_graph(
        G,
        PDF_DIR=PDF_DIR,
        DESTINATION_DIR=SORTED_DIR,
        n_largest_names=n_largest_names,
        n_largest_description=n_largest_description,
        min_graph_size=min_graph_size,
        partition_resolution=partition_resolution,
        word_list=[],
        iteration=iteration,
        max_depth=max_depth,
    )

    return

