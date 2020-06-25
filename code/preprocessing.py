"""
Document that contain all the functions to preprocess the text.
This includes, tokenize, delete stopwords, lower_case and stemming.
"""
import unicodedata
import re
import copy

import nltk

nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem import PorterStemmer
from embeddings import FTEmbeddings


class Preprocessing:
    '''Class to preprocess the data.
    '''
    @staticmethod
    def pipeline1(data):
        '''First pipeline of preprocessing. Preprocess all the data without embeddings.

        Arguments:
            - data: data to preprocess
        Returns:
            - pdata: data preprocessed
        '''
        pdata = copy.deepcopy(data)
        pdata = Preprocessing.preprocess_all_sentences(pdata)
        pdata = Preprocessing.tokenize(pdata)
        pdata = Preprocessing.delete_stopwords(pdata)
        pdata = Preprocessing.stemming(pdata)
        pdata = Preprocessing.pad_sentences(pdata)
        return pdata

    @staticmethod
    def pipeline2(data):
        '''Second pipeline of preprocessing. Preprocess all the data with embeddings.

        Arguments:
            - data: data to preprocess
        Returns:
            - pdata: data preprocessed
        '''
        pdata = copy.deepcopy(data)
        pdata = Preprocessing.preprocess_all_sentences(pdata)
        pdata = Preprocessing.tokenize(pdata)
        pdata = Preprocessing.delete_stopwords(pdata)
        pdata = Preprocessing.stemming(pdata)
        pdata = Preprocessing.pad_sentences(pdata)
        embeddings = Preprocessing.calculate_embeddings(pdata)
        return pdata, embeddings

    @staticmethod
    def tokenize(text):
        '''Function that tokenize the text.

        Arguments:
            - text: text to be tokenize
        Returns:
            - A list with all the text tokenized
        '''
        return [word_tokenize(sentence) for sentence in text]

    # Converts the unicode file to ascii
    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    @staticmethod
    def preprocess_sentence(w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    @staticmethod
    def preprocess_all_sentences(data):
        '''Function to apply preprocess_sentence function to every instance.
        Arguments:
            - data: All the data to preprocess
        Returns:
             - data preprocessed.
        '''
        for i in range(len(data)):
            data[i] = self.preprocess_sentence(data[i])
        return data

    @staticmethod
    def stemming( text):
        '''Function to get the stem for every word
        Arguments:
            - text: list of lists with the text tokenized.
        Returns:
            - list of lists with the stem applied.
        '''
        stemmer = PorterStemmer()
        return [[stemmer.stem(token) for token in sentence] for sentence in text]

    @staticmethod
    def delete_stopwords(text):
        '''Function to delete the stopwords in english
        Arguments:
            - text: text to delete the stopwords. The sentences must be tokenized first.
        Returns:
            - text without stopwords
        '''
        # Pick all the stopwords in english
        stop_words_en = list(get_stop_words('en'))
        nltk_words_en = list(stopwords.words('english'))
        all_stop_words = stop_words_en + nltk_words_en
        # Delete all the stopwords
        return [[w for w in word if w not in all_stop_words] for word in text]

    @staticmethod
    def pad_sentences(text, max_len):
        '''Function to pad the sentences from the text to a max_len
        Arguments:
            - text: list of lists that will be padded to the max_len argument.
            - max_len: int that indicates the max lenght for every sentence.
        Return:
            - all the text padded
        '''
        for i in range(len(text)):
            if len(text[i]) > max_len:
                text[i] = text[i][:max_len]
            elif len(text[i]) < max_len:
                if isinstance(text[i], str):
                    text[i] = text[i] + ''.join([str(0) for j in range(max_len-len(text[i]))])
                elif isinstance(text[i], list):
                    text[i] += [0 for i in range(max_len - len(text[i]))]
        return text

    @staticmethod
    def calculate_embeddings(text):
        '''Calculate fasttext embeddings for the data
        Params:
            - text: list o lists. The text must be tokenized before entering here.
        Returns:
            - embeddings from the text.
        '''
        ft = FTEmbeddings()
        embeddings = ft.apply_vectors(text)
        return embeddings
