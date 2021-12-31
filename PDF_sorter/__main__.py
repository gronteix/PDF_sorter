import os
import fire
from PDF_sorter.recursive_sort import recursive_sort


def main(
    pdf_dir: str = "",
    sorted_dir: str = "",
    max_common_words: int = 1,
    num_words: int = 10,
    num_topics: int = 10,
    n_largest_names: int = 2,
    partition_resolution: float = 0.9,
    n_largest_description: int = 10,
    savedict: bool = True,
    dict_dir: str = "dictionnary",
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

    # Recursive sort from PDFs contained in a folder.
    if sorted_dir == "":
        if not os.path.exists(pdf_dir):
            os.mkdir(os.path.join(pdf_dir, "sorted_papers"))
        sorted_dir = os.path.join(pdf_dir, "sorted_papers")

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
