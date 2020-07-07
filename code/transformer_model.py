# TODO: Create a transformer model based on encoder-decoder.
# TODO: Probably trnasformer model will be only based on the encoder from the transformer architecture.

"""
File for the attention model created
"""
from __future__ import print_function

import time
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, Dropout, Layer, Dense, MaxPool1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from ast import literal_eval
import tensorflow_datasets as tfds
from collections import Counter
from basemodel import BaseModel
from transformerlayer import EncoderLayer

# TODO: Probably delete some arguments because of some argument must be repeated in the parameters.
# TODO: Create the fit function, the LearningRateScheduler, the loss function and add the metrics.
class TransformerEncoder(BaseModel):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 maximum_position_encoding, epochs, batch_size, optimizer,
                 max_sequence_len, path_train, path_test, path_dev, embedding_size=300,
                 max_len=1900, load_embeddings=True, emb_type='fasttext', rate=0.3, learning_rate = 1e-3):
        super(TransformerEncoder, self).__init__(max_len=max_len, path_train=path_train,
                                          path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                          embedding_size=embedding_size, emb_type=emb_type, vocab_size=vocab_size,
                                          max_sequence_len=max_sequence_len, epochs=epochs, learning_rate=learning_rate,
                                          optimizer=optimizer, load_embeddings=load_embeddings)

        self.d_model = d_model
        self.num_layers = num_layers
        # TODO: Add the next params to BaseModel (refactoring).
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.OPTIMIZERS = {
            'adam': Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            'rmsprop': RMSprop(learning_rate == self.learning_rate)
        }
        self.optimizer = self.OPTIMIZERS[optimizer]
        self.load_embeddings = load_embeddings
        self.callbacks = None
        self.checkpoint_filepath = './checkpoints/checkpoint'
        self.model_save = ModelCheckpoint(filepath=self.checkpoint_filepath, save_weights_only=True, mode='max',
                                          monitor='val_acc', save_best_only=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.2, verbose=0, mode='auto',
                                           min_lr=1e-5)
        # self.callbacks = [self.reduce_lr, self.model_save]
        self.callbacks = None

        # Layers for the transformer encoder
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, :-1]

    def fit(self, with_validation=False):
        """Fit the model. Implement
        Arguments:
            - with_validation (bool): If True test data is applied as validation set
        """
        train_step_signature = [
            tf.TensorShape(shape=(None, None), dtype=tf.int64),
            tf.TensorShape(shape=(None, None), dtype=tf.int64),
        ]

        for epoch in range(self.epochs):
            start = time.time()

