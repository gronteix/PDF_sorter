import os
import fire
from PDF_sorter.recursive_sort import recursive_sort

def main(
    pdf_dir: str,
    sorted_dir: str = "",
    max_common_words: int = 1,
    num_words: int = 10,
    num_topics: int = 10,
    n_largest_names: int = 2,
    partition_resolution: float = 0.9,
    n_largest_description: int = 10,
    savedict: bool = True,
    dict_dir: str = "",
    min_graph_size: int = 10,
    no_keywords: list = [
        "http",
        "ncbi",
        "experi",
        "biorxiv",
        "pubm",
        "elsevi",
        "refhub",
    ],
    iteration: int = 0,
    max_depth: int = 5,
):

    """
    Sorts PDFs into folders based on their common words.
    
    Parameters:
    -----------
    - pdf_dir : str
        Directory containing the PDFs to sort.
    - sorted_dir : str, optional
        Directory to save the sorted PDFs. The default is "".
    - max_common_words : int, optional
        The maximum number of common words to use for sorting. The default is 1.
    - num_words : int, optional
        The number of words to use for sorting. The default is 10.
    - num_topics : int, optional
        The number of topics to use for sorting. The default is 10.
    - n_largest_names : int, optional
        The number of names to use for sorting. The default is 2.
    - partition_resolution : float, optional
        The partition resolution to use for the Louvain community clustering. The default is 0.9.
    - n_largest_description : int, optional
        The number of descriptions to use for sorting. The default is 10.
    - savedict : bool, optional
        Whether to save the dictionary containing the PDF information. The default is True.
    - dict_dir : str, optional
        Directory to save the dictionary. The default is "".
    - min_graph_size : int, optional
        The minimum number of nodes in the graph to use for sorting. The default is 10.
    - no_keywords : list, optional
        List of keywords to ignore. The default is [].
    - iteration : int, optional
        The iteration number to use for sorting. The default is 0.
    - max_depth : int, optional
        The maximum depth to use for sorting. The default is 5.
    
    Returns:
    --------
    - None."""

    # Recursive sort from PDFs contained in a folder.
    if sorted_dir == "":
        if not os.path.exists(os.path.join(pdf_dir, "sorted_papers")):
            os.mkdir(os.path.join(pdf_dir, "sorted_papers"))
        sorted_dir = os.path.join(pdf_dir, "sorted_papers")

    if dict_dir == "":
        if not os.path.exists(os.path.join(pdf_dir, "dictionnary")):
            os.mkdir(os.path.join(pdf_dir, "dictionnary"))
        dict_dir = os.path.join(pdf_dir, "dictionnary")

    recursive_sort.recursive_sort_papers_based_on_contents(
        PDF_DIR=pdf_dir,
        SORTED_DIR=sorted_dir,
        max_common_words=max_common_words,
        num_words=num_words,
        num_topics=num_topics,
        n_largest_names=n_largest_names,
        partition_resolution=partition_resolution,
        n_largest_description=n_largest_description,
        SAVEDICT=savedict,
        DICTDIR=dict_dir,
        min_graph_size=min_graph_size,
        no_keywords=no_keywords,
        iteration=iteration,
        max_depth=max_depth,
    )

    return


if __name__ == "__main__":
    fire.Fire(main)
