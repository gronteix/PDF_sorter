import numpy as np
import os
import glob
import open_PDF
import stemmer
import gensim
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import tqdm

nltk.download('wordnet')
stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    
    result=[]
    
    for token in gensim.utils.simple_preprocess(text):
        
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

def get_topics_paper(text,
                     num_topics = 10,
                     num_words = 2):

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

    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                       num_topics = num_topics, 
                                       id2word = dictionary,                                    
                                       passes = 10,
                                       workers = 2)

    lda_topics = lda_model.show_topics(num_words=num_words)

    keywords = []
    filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

    for topic in lda_topics:
        for word in preprocess_string(topic[1], filters):        
            keywords.append(word)
        
    return keywords

def make_dict_from_papers(PDF_DIR,
                          num_words,
                          num_topics):

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

    for fname in tqdm(glob.glob(os.path.join(PDF_DIR, '*.pdf'))):

        try:

            text = open_PDF.open_PDF_tika(fname)
            keywords = get_topics_paper(text, 
                         num_words = num_words, 
                         num_topics = num_topics)

            paper_name = os.path.basename(fname)
            dir_name = os.path.dirname(fname)

            paper_dictionnary[paper_name] = {}
            paper_dictionnary[paper_name]['directory'] = dir_name
            paper_dictionnary[paper_name]['full_path'] = fname
            paper_dictionnary[paper_name]['keywords'] = keywords

        except:

            error_files.append(fname)
           
        if len(error_files)>0:
            print('Error for: ')
            print(error_files)
            
    return paper_dictionnary