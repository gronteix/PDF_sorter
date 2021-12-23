import click
import os
from PDF_sorter.recursive_sort import recursive_sort

@click.command()
@click.argument('pdf_dir', nargs=1)

@click.option('--sorted_dir', 
    type=str,
    default='', 
    help='Repository where the sorted papers will be stored')

@click.option('--max_common_words', 
    default=1, 
    help='Number of most common words to remove from the analysis')

@click.option('--num_words',
    type=int,
    default=10,
    help='Number of words to use for the topic modeling')

@click.option('--num_topics',
    type=int,
    default=10,
    help='Number of topics to use for the topic modeling')

@click.option('--n_largest_names',
    type=int,
    default=2, 
    help='Number of most common names in partition to name the folder')

@click.option('--partition_resolution',
    type=float,
    default=0.9,
    help='Resolution of the partition in the Louvain algorithm. A smaller value will lead to a more precise partition, but will take more time.')

@click.option('--n_largest_description',
    type=int,
    default=10,
    help='Number of most common words in partition to describe the folder contents')

@click.option('--savedict',
    type=bool,
    default=True,
    help='Whether to save the dictionnary containing the paper descriptions.')

@click.option('--dict_dir',
    type=str,
    default='dictionnary',
    help='Path to the folder where the dictionnary will be stored.')

@click.option('--min_graph_size',
    type=int,
    default=10,
    help='Minimum number of papers in a partition to be considered for further classification.')

@click.option(
    '--no_keywords', '-nk',
    type=list,
    default=["http", "ncbi", "experi", "biorxiv", "pubm", "elsevi", "refhub"],
    show_default=True,
    help='Keywords to exlude from articles')

@click.option('--iteration',
    type=int,
    default=0,
    help='Initial iteration number.')

@click.option('--max_depth',
    type=int,
    default=5,
    help='Maximum depth of the recursive sort.')

def main(
    pdf_dir,
    sorted_dir,
    max_common_words,
    num_words,
    num_topics,
    n_largest_names,
    partition_resolution,
    n_largest_description,
    savedict,
    dict_dir,
    min_graph_size,
    no_keywords,
    iteration,
    max_depth):

    # Recursive sort from PDFs contained in a folder.
    if sorted_dir == '':
        if not os.path.exists(pdf_dir):
            os.mkdir(os.path.join(pdf_dir, 'sorted_papers'))
        sorted_dir = os.path.join(pdf_dir, 'sorted_papers')

    recursive_sort.recursive_sort_papers_based_on_contents(
        PDF_DIR = pdf_dir,
        SORTED_DIR = sorted_dir,
        max_common_words = max_common_words,
        num_words = num_words,
        num_topics = num_topics,
        n_largest_names = n_largest_names,
        partition_resolution = partition_resolution,
        n_largest_description = n_largest_description,
        SAVEDICT = savedict,
        DICTDIR = dict_dir,
        min_graph_size = min_graph_size,
        no_keywords=no_keywords,
        iteration=iteration,
        max_depth=max_depth,
    )

    return

if __name__ == "__main__":

    main()
   