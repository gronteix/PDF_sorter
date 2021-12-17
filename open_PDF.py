import numpy as np

def open_PDF(pdf_file):
    """
    This function opens a PDF file and returns the data as a numpy array.
    """
    import PyPDF2
    pdf_file = open(pdf_file, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    pdf_data = []
    for page in range(pdf_reader.numPages):
        pdf_data.append(pdf_reader.getPage(page).extractText())
    pdf_data = np.array(pdf_data)
    return pdf_data