#!/usr/bin/env python

from setuptools import setup

setup(
    name="PDF_sorter",
    version="0.0.1",
    description="Sorts PDFs into folders based on their common words",
    author="Gustave Ronteix",
    author_email="gustave.ronteix@pasteur.fr",
    url="https://github.com/gronteix/PDF_sorter",
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "pytest",
        "tqdm",
        "pandas",
        "click",
        "pip",
        "black",
        "gensim",
        "tox",
        "nltk",
        "tika",
        "pytest",
        "fire",
    ],
    packages=[
        "PDF_sorter",
        "PDF_sorter.utils",
        "PDF_sorter.simple_sort",
        "PDF_sorter.recursive_sort",
        "PDF_sorter.open_PDF",
    ],
)
