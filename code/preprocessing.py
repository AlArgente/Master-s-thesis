"""
Document that contain all the functions to preprocess the text.
This includes, tokenize, delete stopwords, lower_case and stemming.
"""
import unicodedata
import re
import copy

import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem import PorterStemmer
from factory_embeddings import FactoryEmbeddings


class Preprocessing:
    '''Class to preprocess text data.
    '''

    @staticmethod
    def pipeline(data):
        '''First pipeline of preprocessing. Preprocess all the data without embeddings.

        Arguments:
            - data: data to preprocess
        Returns:
            - pdata_stem: Data with all the preprocess applied.
            - pdata_join: Same as pdata_stem but each instance ins joined as sentence.
        '''
        # Safe copy from the data to preprocess
        pdata = copy.deepcopy(data)
        print('Preprocesado de las sentencias.')
        # Tokenize the punctuation
        pdata = Preprocessing.preprocess_all_sentences(pdata)
        print('Conversión de contracciones.')
        # Remove all the possible contractions
        pdata = Preprocessing.remove_all_contractions(pdata)
        print('Tokenización.')
        # Tokenize the data.
        pdata = Preprocessing.tokenize(pdata)
        print('Eliminación de stopwords.')
        # Delete stopwords from the data
        pdata = Preprocessing.delete_stopwords(pdata)
        print('Stemming.')
        # Stem the data
        pdata_stem, pdata_join = Preprocessing.stemming(pdata)
        return pdata_stem, pdata_join

    @staticmethod
    def pipeline_simple(data):
        '''Seconf pipeline of preprocessing. Preprocess all the data without embeddings.

                Arguments:
                    - data: data to preprocess
                Returns:
                    - pdata: data with sencences preprocesed
                '''
        # Safe copy from the data to preprocess
        pdata = copy.deepcopy(data)
        print('Preprocesado de las sentencias.')
        # Tokenize the punctuation
        pdata = Preprocessing.preprocess_all_sentences(pdata)
        print('Conversión de contracciones.')
        # Remove all the possible contractions
        pdata = Preprocessing.remove_all_contractions(pdata)
        print('Return pipeline simple')
        return pdata


    @staticmethod
    def tokenize(text):
        '''Function that tokenize the text.

        Arguments:
            - text: text to be tokenize
        Returns:
            - A list with all the text tokenized
        '''
        # Uso nltk tokenizer to tokenize
        return [word_tokenize(sentence) for sentence in text]

    # Converts the unicode file to ascii
    @staticmethod
    def unicode_to_ascii(s):
        """Function that transform the data from unidecode to ascii
        """
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    @staticmethod
    def preprocess_sentence(w):
        """Function that split the punctuation.
        """
        w = Preprocessing.unicode_to_ascii(w.lower().strip())

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
        # w = '<start> ' + w + ' <end>'
        return w

    @staticmethod
    def remove_all_contractions(text):
        """Function to apply remove_contractions function to all the data
        """
        for i in range(len(text)):
            text[i] = Preprocessing.remove_contractions(text[i])
        return text

    @staticmethod
    def remove_contractions(text):
        """
        Remove all the possible contractions in the text so we can tokenize it
        better and to delete more stopwords.
        """
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how does",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            " u ": " you ",
            " ur ": " your ",
            " n ": " and "}
        for word in text.split():
            if word.lower() in contractions:
                text = text.replace(word, contractions[word.lower()])
        return text

    @staticmethod
    def preprocess_all_sentences(data):
        '''Function to apply preprocess_sentence function to every instance.
        Arguments:
            - data: All the data to preprocess
        Returns:
             - data preprocessed.
        '''
        for i in range(len(data)):
            data[i] = Preprocessing.preprocess_sentence(data[i])
        return data

    @staticmethod
    def stemming(text):
        '''Function to get the stem for every word
        Arguments:
            - text: list of lists with the text tokenized.
        Returns:
            - list of lists with the stem applied.
        '''
        stemmer = PorterStemmer()
        # Stem all the data
        stem_list = [[stemmer.stem(token) for token in sentence] for sentence in text]
        # Generate the instances with join
        stem_join = [' '.join(sentence) for sentence in stem_list]
        return stem_list, stem_join

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
    def pad_sentences(text, max_len=10):
        '''Function to pad the sentences from the text to a max_len.
        Actually this function is not used because I'm applying pad_sequences from TensorFlow.
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
                    text[i] = text[i] + ''.join([str(0) for j in range(max_len - len(text[i]))])
                elif isinstance(text[i], list):
                    text[i] += [0 for i in range(max_len - len(text[i]))]
        return text

    @staticmethod
    def calculate_embeddings(text, type='glove'):
        '''Calculate fasttext embeddings for the data.
        Not used because it's needed to calculate the vocabulary and that it's done in basemodel.py
        Params:
            - text: list o lists. The text must be tokenized before entering here.
        Returns:
            - embeddings from the text. Matrix of embeddings, with each sequence with his corresponding array of
            embeddings.
        '''
        emb = FactoryEmbeddings()
        emb.load_embeddings(type)
        embeddings = emb.embeddings.calc_embeddings(text=text)
        return embeddings
