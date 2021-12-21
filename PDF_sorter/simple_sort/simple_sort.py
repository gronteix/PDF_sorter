import numpy as np
import networkx as nx
import PDF_sorter.utils.utils as utils


def sort_papers_based_on_contents(
    PDF_DIR,
    SORTED_DIR,
    max_common_words,
    num_words,
    num_topics,
    n_largest_names,
    partition_resolution,
    n_largest_description,
    SAVEDICT: bool,
    DICTDIR,
    no_keywords=["http", "ncbi", "experi", "biorxiv", "pubm", "elsevi", "refhub"],
):

    paper_dictionnary = utils.make_dict_from_papers(PDF_DIR, num_words, num_topics)

    if SAVEDICT:
        utils.savedict(paper_dictionnary, DICTDIR)

    word_count = utils.make_word_list(paper_dictionnary)
    no_keywords += utils.get_most_common_words(word_count, max_common_words)

    G = utils.make_graph_from_dict(paper_dictionnary, no_keywords)

    utils.sort_papers_from_graph(
        G,
        PDF_DIR,
        SORTED_DIR,
        n_largest_names,
        n_largest_description,
        partition_resolution,
    )

    return


def sort_papers_from_dict(
    DICTNAME,
    PDF_DIR,
    SORTED_DIR,
    max_common_words,
    n_largest_names,
    partition_resolution,
    n_largest_description,
    no_keywords=["http", "ncbi", "experi", "biorxiv", "pubm", "elsevi", "refhub"],
):

    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open(DICTNAME, "rb") as fp:
        paper_dictionnary = pickle.load(fp)

    word_count = utils.make_word_list(paper_dictionnary)
    no_keywords += utils.get_most_common_words(word_count, max_common_words)

    G = utils.make_graph_from_dict(paper_dictionnary, no_keywords)

    partition = utils.community_louvain.best_partition(
        G, resolution=partition_resolution
    )

    for node in G.nodes():
        G.nodes[node]["partition"] = partition[node]

    utils.sort_papers_from_graph(
        G,
        PDF_DIR,
        SORTED_DIR,
        n_largest_names,
        n_largest_description,
        partition_resolution,
    )

    return
