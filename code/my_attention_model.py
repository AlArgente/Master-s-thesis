"""
File for the attention model created
"""
from __future__ import print_function

import io
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, Dropout, Layer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences

OPTIMIZERS = {
        'adam' : Adam(),
        'rmsprop' : RMSprop()
    }

class AttentionModel(Layer):
    '''Class that contain the attention model created for the Proppy database.
    '''
    def __init__(self, batch_size, epochs, buffer_size, max_features, max_len, filters, kernel_size, optimizer,
                 max_sentence_len, lstm_units, embedding_size=300, load_embeddings=False):
        """Init function for the model.
        """
        super(AttentionModel, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer_size = buffer_size
        self.max_features = max_features
        self.max_len = max_len
        self.filters = filters
        self.kernel_size = kernel_size
        self.optimizer = OPTIMIZERS[optimizer]
        self.max_sentence_len = max_sentence_len
        self.lstm_units = lstm_units
        self.embedding_size = embedding_size
        self.load_embeddings = load_embeddings

    def build(self, input_shape=300):
        """Build the model with all the layers.
        """
        self.model = Sequential()
        if self.load_embeddings:
            self.data = self.__load_vectors__()
        else:
            self.model.add(tf.keras.layers.Embedding(self.max_features, self.embedding_size, input_length=self.max_len))
        '''
        self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same', input_shape=input_shape)) 
        # La capa superior se añade si los embeddings son los de fasttext, sino la capa de abajo
        self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same'))
        self.model.add(Dropout(0.3))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu', return_sequences=True)))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu')))
        self.model.add(Dropout(0.3))  
        self.model.add(Dense(1), activation='softmax')      
        '''

    def fit(self):
        """Fit the model
        """
        self.model.compile(optimizer=self.optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
