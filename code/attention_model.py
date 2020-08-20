"""
File for the attention model created
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, \
    Dropout, Layer, Dense, MaxPool1D, Concatenate, LayerNormalization, SpatialDropout1D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from basemodel import BaseModel
from banhdanauattention import BahdanauAttention

SEED = 42


class AttentionModel(BaseModel):
    '''Class that contain the attention model created for the Proppy database.
    '''

    def __init__(self, batch_size, epochs, optimizer, max_sequence_len, lstm_units,
                 path_train, path_test, path_dev, vocab_size=None, learning_rate=1e-3,
                 filters=64, kernel_size=5, pool_size=4, rate=0.2, l2_rate=1e-5,
                 embedding_size=300, max_len=1900, load_embeddings=True, buffer_size=3, emb_type='fasttext',
                 length_type='median', dense_units=128, both_embeddings=False, att_units=0):
        """Init function for the model.
        """
        super(AttentionModel, self).__init__(max_len=max_len, path_train=path_train,
                                             path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                             embedding_size=embedding_size, emb_type=emb_type, vocab_size=vocab_size,
                                             max_sequence_len=max_sequence_len, epochs=epochs,
                                             learning_rate=learning_rate,
                                             optimizer=optimizer, load_embeddings=load_embeddings, rate=rate,
                                             length_type=length_type, dense_units=dense_units,
                                             both_embeddings=both_embeddings,
                                             filters=filters, kernel_size=kernel_size, pool_size=pool_size,
                                             buffer_size=buffer_size, l2_rate=l2_rate
                                             )
        self.lstm_units = lstm_units

    def attention(self, query, key, value):
        """Function that computes the Scaled Dot-Product Attention
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def scaled_dot_product_attention(self, inputs):
        query = Dense(self.lstm_units*2, name='query')(inputs)
        key = Dense(self.lstm_units*2, name='key')(inputs)
        values = Dense(self.lstm_units*2, name='values')(inputs)
        att, weights = self.attention(query, key, values)
        return att, weights

    def call(self):
        # Prepraring the embeddings input
        sequence_input = tf.keras.layers.Input(shape=(self.max_sequence_len,), dtype="int32", name="seq_input")
        # Preparing the document mean
        sequente_input_mean_emb = tf.keras.layers.Input(shape=(self.embedding_size,), dtype='float32', name='mean_emb')
        sequente_input_mean_emb = Dense(self.dense_units, activation='relu', kernel_regularizer=l2(self.l2_rate))(
            sequente_input_mean_emb)
        token_embeddings_glove = tf.keras.layers.Embedding(self.nb_words, self.embedding_size,
                                                           weights=[self.embeddings_matrix],
                                                           input_length=self.max_sequence_len,
                                                           trainable=False, name='embeddings')
        # embedding_sequence = Concatenate(axis=1, name='full_embeddings')([embedding_sequence_glove,
        #                                                                   embedding_sequence_ft])
        embedding_sequence = token_embeddings_glove(sequence_input)
        embedding_sequence = SpatialDropout1D(0.2)(embedding_sequence)
        # Add the BiLSTM layer and get the states
        x, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(units=self.lstm_units, activation='tanh',
                                                                             return_sequences=True,
                                                                             recurrent_dropout=0,
                                                                             recurrent_activation='sigmoid',
                                                                             unroll=False, use_bias=True,
                                                                             kernel_regularizer=l2(self.l2_rate),
                                                                             return_state=True),
                                                                        name='bilstm')(embedding_sequence)
        # Add BahdanauAttention
        # Pass states to the attention_layer
        # att, _ = BahdanauAttention(self.att_units)(forward_h, x) # It only works with forward_h
        # Scaled Dot Product Attention
        att, _ = self.scaled_dot_product_attention(x)
        # Norm the attetion
        out1 = LayerNormalization(epsilon=1e-6, name='normalization_att_bilstm')(x + att)
        max_pooling = GlobalMaxPool1D()(out1)
        dropout = Dropout(self.rate, name='dropout')(max_pooling)
        # Add the Dense layer
        dense = Dense(units=self.dense_units, activation='relu', kernel_regularizer=l2(self.l2_rate),
                      name='dense_layer')(dropout)
        # concat = Concatenate()([pool, sequente_input_mean_emb])
        # Pred layer
        prediction = Dense(units=2, activation='softmax', name='pred_layer')(dense)
        self.model = Model(inputs=[sequence_input], outputs=[prediction], name='attention_model')
        self.model.compile(loss=BinaryCrossentropy(),
                           optimizer=self.optimizer,
                           metrics=self.METRICS)

        self.model.summary()
        # tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=48, to_file='attention_model.png')
