from tika import parser


def open_PDF_tika(pdf_file):

    """
    This function opens a PDF file using tika and returns the
    data as a numpy array.
    """

    pdf_data = parser.from_file(pdf_file)
    pdf_data = pdf_data["content"]
    return pdf_data


def clean_text(pdf_data):
    """
    This function cleans the text data.
    """
    pdf_data = pdf_data.replace("\n", " ")
    pdf_data = pdf_data.replace("\r", " ")
    pdf_data = pdf_data.replace("\t", " ")
    pdf_data = pdf_data.replace("\xa0", " ")
    pdf_data = pdf_data.replace("-", "")
    pdf_data = pdf_data.replace('"', "")
    pdf_data = pdf_data.replace("[", "")
    pdf_data = pdf_data.replace("]", "")
    pdf_data = pdf_data.replace(").", "")
    pdf_data = pdf_data.replace("].", "")
    return pdf_data


def remove_mathematical_expressions_text(pdf_data):
    """
    This function removes mathematical expressions from the text data.
    """
    import re

    pdf_data = re.sub(r"\$.*?\$", "", pdf_data)
    return pdf_data


def remove_numbers_from_text(pdf_data):
    """
    This function removes the numbers from the text data.
    """
    import re

    pdf_data = re.sub(r"\d+", "", pdf_data)
    return pdf_data


def extract_concepts_from_text(pdf_data):
    """
    This function extracts the concepts from the text data.
    """
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    import string

    punctuation = set(string.punctuation)
    pdf_data = pdf_data.lower()
    pdf_data = pdf_data.split()
    pdf_data = [word for word in pdf_data if word not in stop_words]
    pdf_data = [word for word in pdf_data if word not in punctuation]
    pdf_data = [word for word in pdf_data if len(word) > 4]
    return pdf_data
