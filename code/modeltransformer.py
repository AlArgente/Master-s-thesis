"""
File for the attention model created
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, \
    Dropout, Layer, Dense, MaxPool1D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from basemodel import BaseModel
from tokenposembeddings import TokenAndPositionEmbedding
from transformerblock import TransformerBlock

SEED = 42


class TransformerModel(BaseModel):
    """Class that contain the transformer model created for the Proppy database.
    """

    def __init__(self, batch_size, epochs, optimizer, max_sequence_len, lstm_units,
                 path_train, path_test, path_dev, filters=64, kernel_size=5,
                 vocab_size=None, learning_rate=1e-3, pool_size=4, rate=0.2, l2_rate=1e-5,
                 embedding_size=300, max_len=1900, load_embeddings=True, buffer_size=3, emb_type='fasttext',
                 length_type='median', dense_units=128, attheads=12, att_layers=2):
        """Init function for the model.
        """
        super(TransformerModel, self).__init__(max_len=max_len, path_train=path_train,
                                               path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                               embedding_size=embedding_size, emb_type=emb_type, vocab_size=vocab_size,
                                               max_sequence_len=max_sequence_len, epochs=epochs,
                                               learning_rate=learning_rate,
                                               optimizer=optimizer, load_embeddings=load_embeddings, rate=rate,
                                               length_type=length_type, dense_units=dense_units,
                                               filters=filters, kernel_size=kernel_size, pool_size=pool_size,
                                               buffer_size=buffer_size, l2_rate=l2_rate
                                               )
        self.lstm_units = lstm_units
        self.attheads = attheads
        self.att_layers = att_layers
        # Create N transformer layers

    def call(self):
        self.inputs = tf.keras.layers.Input(shape=(self.max_sequence_len,), dtype="int32", name="seq_input")
        self.embedding_layer = TokenAndPositionEmbedding(self.max_sequence_len, self.nb_words, self.embedding_size,
                                                         self.embeddings_matrix)
        self.transformer_layers = [TransformerBlock(self.embedding_size, self.attheads, self.dense_units, self.rate)
                                   for _ in range(self.att_layers)]
        x = self.embedding_layer(self.inputs)
        # Create the transformer layers
        for i in range(self.att_layers):
            x = self.transformer_layers[i](x)
        # For a single transformer layer
        # transformer_block = TransformerBlock(self.embedding_size, self.attheads, self.dense_units, self.rate)
        # x = transformer_block(x)
        # Now shape = # (None, input_seq_len, embeddings_dim)
        x = Dense(self.dense_units, activation='relu', kernel_regularizer=l2(self.l2_rate), name='dense_layer1')(x)
        x = GlobalMaxPool1D()(x)
        # Now shape = (None, embeddings_dim)
        # x = Dropout(self.rate)(x)
        # x = Dense(self.dense_units)(x)
        # Now shape = (None, self.dense_units)
        x = Dropout(self.rate)(x)
        output = Dense(2, activation='softmax')(x)
        # Now shape = (None, 2)
        self.model = Model(inputs=self.inputs, outputs=output, name='transformer_model')
        self.model.compile(loss=BinaryCrossentropy(),
                           optimizer=self.optimizer,
                           metrics=self.METRICS)

        self.model.summary()
        # tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=48, to_file='transformer_model.png')
