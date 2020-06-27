import pandas as pd
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from factory_embeddings import FactoryEmbeddings

class BaseModel(Layer):

    def __init__(self, max_len, path_train, path_test, path_dev, batch_size, embedding_size, emb_type, vocabulary=None,
                 embedding_matrix=None,**kwargs):
        super(BaseModel).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.path_train = path_train
        self.path_test = path_test
        self.path_dev = path_dev
        self.batch_size = batch_size
        self.embedding_matrix = embedding_matrix
        self.embedding_size = embedding_size
        self.emb_type = emb_type

    def load_data(self):
        """Load the 3 DataFrames, train-test-dev. The data here will be preprocessed, previously tokenized,
        stopwords deleted and stem.
        """
        self.train = pd.read_csv(self.path_train, sep='\t')
        self.test = pd.read_csv(self.path_test, sep='\t')
        self.dev = pd.read_csv(self.path_dev, sep='\t')

    def pad_sentences(self):
        """Function that pad all the sentences  to the max_len parameter from the class
        """
        pass

    def load_vocabulary(self):
        """Function that extract the vocabulary from the train DataFrame
        """
        pass

    def create_embeddings(self):
        """Function that create the embedding matrix
        """
        self.emb = FactoryEmbeddings()