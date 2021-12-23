# This is **`PDF_sorter`**

Some people might have spent their past years in research downloading endless stacks of PDF documents without any proper sorting system. This *can* have consequences when writing a bibliography or a review article.

This Python module is an attempt to solve this problem. It extracts keywords from PDF documents and sorts the articles based on common topics. After years of procrastination, your folder where your papers have been dumped is now sorted!

## Installation

Download the folder from GitHub, navigate to the module on the command prompt and type:

```
pip install .
```

## Requirements

The current program uses [`tika`](https://github.com/chrismattmann/tika-python), which requires at least Java 8 to be [installed](https://java.com/en/download/help/download_options.html) on the computer. Earlier versions of Java will lead to errors in the PDF parsing.

## Usage

After installation, the program can be called from the command line. The principal argument `$PATH_TO_PDFs` is the path to the directory containing the PDFs to be sorted. A typical call is as follows:

```
python -m PDF_sorter $PATH_TO_PDFs
```

This will generate a folder containing the sorted PDFs in the `$PATH_TO_PDFs` folder according to their keywords. A typical result would be:

```
.

├── sorted_papers           # Folder containing the sorted papers
│   ├── keyword1_keyword2_  # Nested sub-directories
│   ├── keyword3_keyword4_  # keywords 1..6 are common paper topics
│   └── keyword5_keyword6_  

```

For further options call:

```
python -m PDF_sorter --help
```

In particular the precision of the clustering in the Louvain community clustering can be tailored with the option `--partition_resolution`. 