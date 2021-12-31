import glob
import os
import shutil
from collections import Counter
from shutil import copyfile

import community.community_louvain as community_louvain
import gensim
import networkx as nx
import nltk
import numpy as np
import pandas
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_numeric,
    strip_punctuation,
)
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm

import PDF_sorter.open_PDF.open_PDF as open_PDF

nltk.download("wordnet")
stemmer = PorterStemmer()


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))


def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result


def get_all_keywords(paper_dictionnary):

    keyword_list = []

    for paper in paper_dictionnary.keys():

        keywords_paper = paper_dictionnary[paper]["keywords"]

        keyword_list += keywords_paper

    return keyword_list


def make_word_list(paper_dictionnary):

    keyword_list = get_all_keywords(paper_dictionnary)

    return Counter(keyword_list)


def get_most_common_words(word_count, n):

    common_word_list = []

    most_common_words = word_count.most_common(n)

    for word_tuple in most_common_words:

        common_word_list.append(word_tuple[0])

    return common_word_list


def get_topics_paper(text, num_topics=10, num_words=2):

    """
    This function returns the topics of a text using
    a LDA approach.

    Input:
    ------
        text: the text to be analyzed
        num_topics: number of topics to be used in the LDA model
        num_words: number of words to be used in the dictionary
    Output:
    -------
        keywords: the keywords of the text
    """

    text = preprocess(text)
    dictionary = gensim.corpora.Dictionary(np.array([text]))

    # generate BOW
    bow_corpus = [dictionary.doc2bow([doc]) for doc in text]

    lda_model = gensim.models.LdaMulticore(
        bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2
    )

    lda_topics = lda_model.show_topics(num_words=num_words)

    keywords = []
    filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

    for topic in lda_topics:
        for word in preprocess_string(topic[1], filters):
            keywords.append(word)

    return keywords


def make_dict_from_papers(PDF_DIR, num_words, num_topics):

    """
    This function makes a dictionary from the PDF
    files in the directory PDF_DIR.

    Input:
    ------
        PDF_DIR: directory of the PDF files
        num_words: number of words to be used in the dictionary
        num_topics: number of topics to be used in the LDA model
    Output:
    -------
        dictionary: a dictionary of the papers and their keywords
    """

    paper_dictionnary = {}
    num_words = 5
    num_topics = 5

    error_files = []

    for fname in tqdm(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):

        try:

            text = open_PDF.open_PDF_tika(fname)
            keywords = get_topics_paper(
                text, num_words=num_words, num_topics=num_topics
            )

            paper_name = os.path.basename(fname)
            dir_name = os.path.dirname(fname)

            paper_dictionnary[paper_name] = {}
            paper_dictionnary[paper_name]["directory"] = dir_name
            paper_dictionnary[paper_name]["full_path"] = fname
            paper_dictionnary[paper_name]["keywords"] = keywords

        except:

            error_files.append(fname)

        if len(error_files) > 0:
            print("Error for: ")
            print(error_files)

    return paper_dictionnary


def make_graph_from_dict(paper_dictionnary, no_keywords):

    """
    This function makes a graph from the dictionary of papers.
    """

    G = nx.Graph()
    G.add_nodes_from(paper_dictionnary.keys())

    for key_n in paper_dictionnary.keys():

        for keyword in paper_dictionnary[key_n]["keywords"]:

            # explore other articles
            for article in paper_dictionnary.keys():

                # not current article
                if article != key_n:

                    # if keyword in both articles then link
                    if (keyword in paper_dictionnary[article]["keywords"]) & (
                        keyword not in no_keywords
                    ):

                        edge_info = G.get_edge_data(article, key_n)
                        if edge_info is None:
                            G.add_edge(article, key_n, weight=1)
                            G[article][key_n]["keywords"] = [keyword]
                        elif keyword not in edge_info["keywords"]:
                            w = edge_info["weight"]
                            G.add_edge(article, key_n, weight=w + 1)
                            G[article][key_n]["keywords"] += [keyword]
    return G


def get_all_the_words(G):

    """
    This function returns all the words in the graph.
    """

    word_list = []

    for u, v, d in G.edges(data=True):

        word_list += d["keywords"]

    return Counter(word_list)


def get_word_prob(counter_dict):

    """
    This function returns the probability of a word composing
    the dictionnary counter_dict.
    """

    total_word_number = np.sum([counter_dict[k] for k in counter_dict.keys()])

    counter_frame = pandas.DataFrame()

    for word in counter_dict.keys():

        counter_frame.loc[word, "prob"] = counter_dict[word] / total_word_number

    return counter_frame


def get_subgraph(G, attribute, attribute_value):

    """
    This function returns the subgraph of G with the attribute
    attribute_value.

    Input:
    ------
        G: networkx graph
        attribute: attribute of the nodes
        attribute_value: value of the attribute

    Output:
    -------
        subgraph: the subgraph of G with the attribute attribute_value
    """

    node_list = []

    for x, y in G.nodes(data=True):

        if y[attribute] == attribute_value:

            node_list.append(x)

    return G.subgraph(node_list)


def prob_in_communities(G):

    """
    This function returns the probability of all words to appear
    in the communities of G.

    Input:
    ------
        G: networkx graph

    Output:
    -------
        prob_frame: dataframe containing the probabilities
                    for each word to appear. Each row is a word.
                    Each column is a community.
    """

    partition = nx.get_node_attributes(G, "partition")
    partition_list = np.unique([partition[k] for k in partition.keys()])

    counter_dict = get_all_the_words(G)
    tot_graph_counter_frame = get_word_prob(counter_dict)

    tot_graph_counter_frame.columns = ["whole_graph"]

    for partition_value in partition_list:

        subgraph = get_subgraph(G, "partition", partition_value)

        counter_dict = get_all_the_words(subgraph)
        counter_frame = get_word_prob(counter_dict)

        if not counter_frame.empty:
            counter_frame.columns = [str(int(partition_value))]

            tot_graph_counter_frame = tot_graph_counter_frame.merge(
                counter_frame, left_index=True, right_index=True, how="outer"
            )

        else:

            counter_frame = pandas.DataFrame(
                data=np.zeros(len(tot_graph_counter_frame))
            )
            counter_frame.columns = [str(int(partition_value))]
            counter_frame.index = tot_graph_counter_frame.index.values

            tot_graph_counter_frame = tot_graph_counter_frame.merge(
                counter_frame, left_index=True, right_index=True, how="outer"
            )

    return tot_graph_counter_frame.fillna(0)


def get_unique_words(word_prob, column):

    if not column in word_prob.columns:
        print(column)
        print(word_prob)

    entropy_frame = pandas.DataFrame()

    for j in word_prob.columns:

        if column != j:

            s = -np.log(word_prob[j] / word_prob[column])
            s = pandas.DataFrame(s)
            s.columns = [j]
            entropy_frame = entropy_frame.merge(
                s, left_index=True, right_index=True, how="outer"
            )

    return pandas.DataFrame(entropy_frame.sum(axis=1))


def sort_papers_from_graph(
    G, PDF_DIR, SORTED_DIR, n_largest_names, n_largest_description, partition_resolution
):

    """
    From the network G, create the folders relative to
    each community and sort the papers in each folder.

    Input:
    ------
        G: networkx graph
        PDF_DIR: directory where the PDFs are
        SORTED_DIR: directory where the sorted PDFs will be
        n_largest_names: number of most common names to keep
        n_largest_description: number of most common
            descriptions to write in readme
        partition_resolution: resolution of the Louvain community
            partitioning.

    Output:
    -------
        None
    """

    partition = community_louvain.best_partition(G, resolution=partition_resolution)

    for node in G.nodes():
        G.nodes[node]["partition"] = partition[node]

    word_prob = prob_in_communities(G)
    partition = nx.get_node_attributes(G, "partition")
    partition_list = np.unique([partition[k] for k in partition.keys()])

    # pour éviter les betises
    word_prob += 1e-10

    if os.path.exists(SORTED_DIR):
        shutil.rmtree(SORTED_DIR)

    if not os.path.exists(SORTED_DIR):
        os.makedirs(SORTED_DIR)

    for partition_number in partition_list:

        # need to string the name
        partition_name = str(int(partition_number))

        s = get_unique_words(word_prob, partition_name)
        name_frame = s.nlargest(n_largest_names, columns=0)

        # make the folder
        folder_name = ""
        for ind in name_frame.index:
            folder_name += ind + "_"

        if not os.path.exists(os.path.join(SORTED_DIR, folder_name)):
            os.mkdir(os.path.join(SORTED_DIR, folder_name))

        # text file description of the folder
        word_frame = s.nlargest(n_largest_description, columns=0)
        word_list = [i for i in word_frame.index]
        write_description(os.path.join(SORTED_DIR, folder_name), word_list)

        for article_name in partition.keys():

            src = os.path.join(PDF_DIR, article_name)
            dst = os.path.join(SORTED_DIR, folder_name, article_name)

            if partition[article_name] == partition_number:

                copyfile(src, dst)

    return


def write_description(DIR, word_list):

    with open(os.path.join(DIR, "readme.txt"), "w") as f:
        for line in word_list:
            f.write(line)
            f.write("\n")

    return


def savedict(dict_to_save, DIR):

    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    if os.path.exists(DIR):
        shutil.rmtree(DIR)

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    with open(os.path.join(DIR, "paper_dictionnary.p"), "wb") as fp:
        pickle.dump(dict_to_save, fp, protocol=pickle.HIGHEST_PROTOCOL)
