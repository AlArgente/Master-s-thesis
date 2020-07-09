"""
File for the attention model created
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, \
    Dropout, Layer, Dense, MaxPool1D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from basemodel import BaseModel


class CNNRNNModel(BaseModel):
    '''Class that contain the attention model created for the Proppy database.
    '''

    def __init__(self, batch_size, epochs, filters, kernel_size, optimizer, max_sequence_len, lstm_units,
                 path_train, path_test, path_dev, vocab_size, learning_rate=1e-3, pool_size=4, rate=0.2,
                 embedding_size=300, max_len=1900, load_embeddings=True, buffer_size=3, emb_type='fasttext'):
        """Init function for the model.
        """
        super(CNNRNNModel, self).__init__(max_len=max_len, path_train=path_train,
                                          path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                          embedding_size=embedding_size, emb_type=emb_type, vocab_size=vocab_size,
                                          max_sequence_len=max_sequence_len, epochs=epochs, learning_rate=learning_rate,
                                          optimizer=optimizer, load_embeddings=load_embeddings, rate=rate)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.buffer_size = buffer_size
        self.callbacks = None
        self.checkpoint_filepath = '../checkpoints/checkpoint'
        self.model_save = ModelCheckpoint(filepath=self.checkpoint_filepath, save_weights_only=True, mode='max',
                                          monitor='val_acc', save_best_only=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, verbose=0, mode='auto',
                                           min_lr=(self.learning_rate/100))
        self.callbacks = [self.reduce_lr, self.model_save]
        # self.callbacks = None

    def call(self):
        """Build the model with all the layers.
        The idea is to build a CNN-RNN model. With de Conv1D layer I try to focus the attention from the model to
        the most important words from the sequence. MaxPooling is used for better performance.
        Then I use 2 BLSTM layers to learn better the information from the data. I apply Dropout to prevent overfitting.
        Arguments:
            - input_shape: Added but not need to use cause max_sequence_len is used.
        """
        self.model = Sequential()
        if self.load_embeddings:
            # Load fasttext or glove embeddings and use it to train the model
            # self.model.add(
                # tf.keras.layers.Embedding(self.embeddings_matrix.shape[0], self.embeddings_matrix.shape[1],
                #                          weights=[self.embeddings_matrix], input_length=self.max_sequence_len,
                #                          trainable=False))
            self.model.add(
                tf.keras.layers.Embedding(self.nb_words, self.embedding_size,
                                          weights=[self.embeddings_matrix], input_length=self.max_sequence_len,
                                          trainable=False))
            self.model.add(
               Conv1D(self.filters, self.kernel_size, activation='relu', padding='same',
                      kernel_regularizer=l2(0.0001)))
            self.model.add(MaxPool1D(pool_size=self.pool_size))
        else:
            # Initialize the model with random embeddings and train them.
            # Fisrt: Create vocabulary
            # Second:
            self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size,
                                                     input_length=self.max_sequence_len, trainable=True))
            self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same',
                                  kernel_regularizer=l2(0.0001)))

        # self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same', input_shape=input_shape))
        # La capa superior se añade si los embeddings son los de fasttext, sino la capa de abajo
        # self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same'))
        self.model.add(Dropout(self.rate))
        """
        self.model.add(Conv1D(self.filters, self.kernel_size, activation='relu', padding='same',
                              kernel_regularizer=l2(0.0001)))
        self.model.add(MaxPool1D(pool_size=self.pool_size))
        self.model.add(Dropout(self.rate))
        """
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu', return_sequences=True,
                                          kernel_regularizer=l2(0.0001))))
        self.model.add(Dropout(self.rate))
        """
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu', return_sequences=True,
                                          kernel_regularizer=l2(0.0001))))
        self.model.add(Dropout(self.rate))
        """
        self.model.add(Bidirectional(LSTM(self.lstm_units, activation='relu', kernel_regularizer=l2(0.0001))))
        self.model.add(Dropout(self.rate))
        # self.model.add(GlobalMaxPool1D())
        """
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0001)))
        self.model.add(Dropout(self.rate))
        """
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(loss=BinaryCrossentropy(),
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        self.model.summary()

    def fit(self, with_validation=False):
        """Fit the model using the keras fit function.
        Arguments:
            - with_validation (bool): If True test data is applied as validation set
        """
        if not with_validation:
            self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                           callbacks=self.callbacks, shuffle=True, class_weight=self.class_weights)
        else:
            self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                           callbacks=self.callbacks, shuffle=True, validation_data=(self.X_dev, self.y_dev),
                           class_weight=self.class_weights)
        print('Salgo de fit')

    def fit_as_tensors(self, with_validation=False):
        """Fit the model using the keras fit function. The data must be loaded using the prepare_data_as_tensors
        function.
        Arguments:
            - with_validation (bool): If True test data is applied as validation set
        """
        if not with_validation:
            self.model.fit(self.train_dataset, epochs=self.epochs, verbose=1, callbacks=self.callbacks,
                           class_weight=self.class_weights)
        else:
            self.model.fit(self.train_dataset, epochs=self.epochs, verbose=1, callbacks=self.callbacks,
                           class_weight=self.class_weights, validation_data=self.val_dataset)
