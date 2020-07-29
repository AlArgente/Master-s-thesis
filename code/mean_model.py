"""
File for the attention model created
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, \
    Dropout, Layer, Dense, MaxPool1D, Concatenate, LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from basemodel import BaseModel
from banhdanauattention import BahdanauAttention

SEED = 42


class MeanModel(BaseModel):
    '''Class that contain the attention model created for the Proppy database.
    '''

    def __init__(self, batch_size, epochs, optimizer, max_sequence_len, lstm_units,
                 path_train, path_test, path_dev, vocab_size=None,
                 learning_rate=1e-3, pool_size=4, rate=0.2, filters=64, kernel_size=5,
                 embedding_size=300, max_len=1900, load_embeddings=True, buffer_size=3, emb_type='fasttext',
                 length_type='median', dense_units=128, both_embeddings=False):
        """Init function for the model.
        """
        super(MeanModel, self).__init__(max_len=max_len, path_train=path_train,
                                        path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                        embedding_size=embedding_size, emb_type=emb_type, vocab_size=vocab_size,
                                        max_sequence_len=max_sequence_len, epochs=epochs,
                                        learning_rate=learning_rate,
                                        optimizer=optimizer, load_embeddings=load_embeddings, rate=rate,
                                        length_type=length_type, dense_units=dense_units,
                                        both_embeddings=both_embeddings, buffer_size=buffer_size,
                                        filters=filters, kernel_size=kernel_size, pool_size=pool_size
                                        )
        self.lstm_units = lstm_units

    def call(self):
        # First Model
        # Prepraring the embeddings input
        sequence_input = tf.keras.layers.Input(shape=(self.max_sequence_len,), dtype="int32", name="seq_input")
        token_embeddings_glove = tf.keras.layers.Embedding(self.nb_words, self.embedding_size,
                                                           weights=[self.embeddings_matrix],
                                                           input_length=self.max_sequence_len,
                                                           trainable=False, name='embeddings')
        # embedding_sequence = Concatenate(axis=1, name='full_embeddings')([embedding_sequence_glove,
        #                                                                   embedding_sequence_ft])
        embedding_sequence = token_embeddings_glove(sequence_input)
        # Add the BiLSTM layer and get the states
        x = Bidirectional(LSTM(units=self.lstm_units, activation='tanh',
                               return_sequences=True,
                               recurrent_dropout=0,
                               recurrent_activation='sigmoid',
                               unroll=False, use_bias=True,
                               kernel_regularizer=l2(0.0001)),
                          name='bilstm')(embedding_sequence)
        pool = GlobalMaxPool1D()(x)
        dense = Dense(units=self.dense_units, activation='relu', kernel_regularizer=l2(0.0001),
                      name='dense_layer')(pool)
        drop_fm = Dropout(self.rate)(dense)
        fm = Model(inputs=sequence_input, outputs=drop_fm)

        # Second Model
        # Preparing the document mean
        sequente_input_mean_emb = tf.keras.layers.Input(shape=(self.embedding_size,), dtype=tf.float32, name='mean_emb')
        dense_layer = Dense(self.dense_units, activation='relu', kernel_regularizer=l2(0.0001))(
            sequente_input_mean_emb)
        drop_sm = Dropout(self.rate)(dense_layer)
        sm = Model(inputs=sequente_input_mean_emb, outputs=drop_sm)

        # Combine the models
        combined = Concatenate()([fm.output, sm.output])
        predropout = Dense(self.dense_units/2, activation='relu', kernel_regularizer=l2(0.0001))(combined)
        dropout = Dropout(self.rate)(predropout)
        # Pred layer
        prediction = Dense(units=2, activation='softmax', name='pred_layer')(dropout)
        self.model = Model(inputs=[fm.input, sm.input], outputs=prediction)
        self.model.compile(loss=BinaryCrossentropy(),
                           optimizer=self.optimizer,
                           metrics=self.METRICS)

        self.model.summary()
        # tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=48, to_file='mean_model.png')
