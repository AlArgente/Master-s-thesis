"""
File for the attention model created
"""
from __future__ import print_function

import io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, Dropout, Layer, Dense, MaxPool1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from ast import literal_eval
import tensorflow_datasets as tfds
from collections import Counter

OPTIMIZERS = {
        'adam' : Adam,
        'rmsprop' : RMSprop
    }

class CNNRNNModel(Layer):
    '''Class that contain the attention model created for the Proppy database.
    '''
    def __init__(self, batch_size, epochs, filters, kernel_size, optimizer, max_sentence_len, lstm_units,
                 path_train, path_test, path_dev, embedding_matrix=None, learning_rate = 1e-3, pool_size=4,
                 vocabulary=None, embedding_size=300, max_len=1900, load_embeddings=False, buffer_size=3,
                 emb_type='glove'):
        """Init function for the model.
        """
        super(CNNRNNModel, self).__init__(vocabulary=vocabulary, max_len=max_len, path_train=path_train,
                                          path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                          embedding_matrix=embedding_matrix, embedding_size=embedding_size,
                                          emb_type=emb_type)
        self.epochs = epochs
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.optimizer = OPTIMIZERS[optimizer](learning_rate=self.learning_rate)
        self.max_sentence_len = max_sentence_len
        self.lstm_units = lstm_units
        self.load_embeddings = load_embeddings
        self.buffer_size = buffer_size
        self.callbacks = None
        self.checkpoint_filepath = './checkpoints/checkpoint'
        self.model_save = ModelCheckpoint(filepath=self.checkpoint_filepath, save_weights_only=True, mode='max',
                                          monitor='val_acc', save_best_only=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.2, verbose=0, mode='auto',
                                           min_lr=1e-5)
        self.callbacks = [self.reduce_lr, self.model_save]
    def build(self, input_shape=300):
        """Build the model with all the layers.
        """
        self.model = Sequential()
        if self.load_embeddings:
            self.model.add(
                self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size))
                Conv1D(self.filters, self.kernel_size, activation='relu', padding='same', input_shape=(None,self.max_len,input_shape)))
        else:
            self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size))
            self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same'))

        # self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same', input_shape=input_shape))
        # La capa superior se añade si los embeddings son los de fasttext, sino la capa de abajo
        # self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same'))
        self.model.add(Dropout(0.3))
        self.model.add(MaxPool1D(pool_size=self.pool_size))
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu', return_sequences=True)))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu')))
        self.model.add(Dropout(0.3))  
        self.model.add(Dense(1, activation='softmax'))

        self.model.compile(loss=BinaryCrossentropy(),
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        print('Model built.')
        self.model.summary()


    def fit(self, X_train, y_train, X_test=None, y_test = None):
        """Fit the model
        """
        print('ESTOY EN EL FIT')
        y_train = tf.keras.utils.to_categorical(y_train)
        if self.load_embeddings == False:
            if X_test is None and y_test is None:
                self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                               callbacks=self.callbacks, shuffle=True)
            else:
                assert type(X_train) == type(X_test)
                assert type(y_train) == type(y_test)
                X_test = X_test.apply(lambda x: np.asarray(literal_eval(x)))
                self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                               callbacks=self.callbacks, shuffle=True, validation_data=(X_test, y_test))

    def predict(self, X_test):
        """Make the prediction for the data given
        Arguments:
            X_test: test preprocessed to predict.
        Returns:
            The prediction done by the model.
        """
        return self.model.predict(X_test, batch_size=self.batch_size, verbose=0)

    def encode_text(self, text):
        self.encoder = tfds.features.text.TokenTextEncoder(text)

    def encode(self, text_tensor, label):
        encoded_text =self.encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(self, text, label):
        pass

    def generate_vocabulary(self, text):
        min_count = 5
        flat_words = []
        for sentence in text:
            flat_words += sentence

        counts = Counter(list(flat_words))
        counts = pd.DataFrame(counts.most_common())
        counts.columns = ['word', 'count']
        counts = counts[counts['count'] > min_count]

        vocab = pd.Series(range(len(counts)), index=counts['word']).sort_index()

        self.vocab_size = vocab
        filtered_text = []
        for doc in text:
            doc = [word for word in doc if word in vocab.index]
            if len(doc):
                filtered_text.append(doc)
        text = filtered_text

        for i, doc in enumerate(text):
            doc[i] = [vocab.loc[word] for word in doc]