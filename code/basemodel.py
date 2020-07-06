import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from factory_embeddings import FactoryEmbeddings
from abc import abstractmethod

class BaseModel(Model):

    def __init__(self, max_len, path_train, path_test, path_dev, epochs, learning_rate, optimizer,
                 load_embeddings, batch_size=32, embedding_size='300', emb_type='fasttext',
                 vocabulary=None, vocab_size=None, max_sequence_len=None):
        super(BaseModel).__init__()
        self._vocabulary = vocabulary
        self.max_len = max_len
        self.path_train = path_train
        self.path_test = path_test
        self.path_dev = path_dev
        self.batch_size = batch_size
        self.embeddings_matrix = None
        self.embedding_size = embedding_size
        self.emb_type = emb_type
        self.max_sequence_len = max_sequence_len
        self._vocab_size = None
        self.train = None
        self.test = None
        self.dev = None
        self.MAX_NB_WORDS = vocab_size
        self.X_train = None
        self.X_test = None
        self.X_dev = None
        self.y_train = None
        self.y_test = None
        self.y_dev = None
        self._vocab_index = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.OPTIMIZERS = {
            'adam': Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            'rmsprop': RMSprop(learning_rate == self.learning_rate)
        }
        self.optimizer = self.OPTIMIZERS[optimizer]
        self.load_embeddings = load_embeddings

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def vocab_index(self):
        return self._vocab_index

    @property
    def testpred(self):
        return self.y_test

    @property
    def devpred(self):
        return self.y_dev

    def load_data(self):
        """Load the 3 DataFrames, train-test-dev. The data here will be preprocessed, previously tokenized,
        stopwords deleted and stem.
        """
        self.train = pd.read_csv(self.path_train, sep='\t')
        self.train.loc[self.train['label'] == -1, 'label'] = 0
        self.test = pd.read_csv(self.path_test, sep='\t')
        self.test.loc[self.test['label'] == -1, 'label'] = 0
        self.dev = pd.read_csv(self.path_dev, sep='\t')
        self.dev.loc[self.dev['label'] == -1, 'label'] = 0

    def pad_sentences(self):
        """Function that pad all the sentences  to the max_len parameter from the class.
        First it tokenize the data with tensorflow tokenizer and then apply the pad_sequences function
        """
        tokenizer = Tokenizer(num_words=self.max_len, lower=True, char_level=False)
        word_seq_train = tokenizer.texts_to_sequences(self.train['text'])
        word_seq_test = tokenizer.texts_to_sequences(self.test['text'])
        word_seq_dev = tokenizer.texts_to_sequences(self.dev['text'])
        if not self.load_embeddings:
            word_index = tokenizer.word_index

        self.X_train = pad_sequences(word_seq_train, maxlen=self.max_sequence_len)
        self.X_test = pad_sequences(word_seq_test, maxlen=self.max_sequence_len)
        self.X_dev = pad_sequences(word_seq_dev, maxlen=self.max_sequence_len)
        self.y_train = tf.keras.utils.to_categorical(self.train['label'], num_classes=2)
        print(self.y_train.shape)
        # self.y_train = self.train['label'].astype('category')
        self.y_test = tf.keras.utils.to_categorical(self.test['label'], num_classes=2)
        # self.y_test = self.test['label'].astype('category')
        self.y_dev = tf.keras.utils.to_categorical(self.dev['label'], num_classes=2)
        # self.y_dev = self.dev['label'].astype('category')

    def load_vocabulary(self):
        """Function that extract the vocabulary from the train DataFrame. This functions will be used with it's needed
        to use random embeddings.
        """
        # TODO: Calculate vocabulary from data
        pass

    def create_embeddings_matrix(self):
        """Function that create the embedding matrix
        """
        self.emb = FactoryEmbeddings()
        self.emb.load_embeddings(self.emb_type)
        self._vocabulary = self.emb.embeddings.vocabulary
        self.embeddings_matrix = np.asarray(self.emb.embeddings.embeddings_matrix, dtype=np.float32)
        new_words_train = 0
        new_words_test = 0
        new_words_dev = 0
        for sentence in self.train['text_stem']:
            for word in sentence:
                if self._vocabulary.get(word) == None:
                    self._vocabulary[word] = self._vocabulary['NEWWORD']
                    new_words_train+=1
        for sentence in self.test['text_stem']:
            for word in sentence:
                if self._vocabulary.get(word) == None:
                    self._vocabulary[word] = self._vocabulary['NEWWORD']
                    new_words_test += 1
        for sentence in self.dev['text_stem']:
            for word in sentence:
                if self._vocabulary.get(word) == None:
                    self._vocabulary[word] = self._vocabulary['NEWWORD']
                    new_words_dev += 1
        self._vocab_size = len(self.vocabulary)
        self.max_len = self._vocab_size
        self._vocab_index = list(self._vocabulary.values())

    def prepare_data(self):
        print('Loading data')
        self.load_data()
        print('Loading Vocabulary and Embeddings Matrix')
        self.create_embeddings_matrix()
        print('Padding sentences')
        self.pad_sentences()
        print('Tras padding')
        print(self.X_train[:20])

    @abstractmethod
    def fit(self, with_validation=False):
        """Fit the model.
        Arguments:
            - with_validation (bool): If True test data is applied as validation set
        """
        pass