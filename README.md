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