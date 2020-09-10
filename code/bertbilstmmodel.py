"""
File for the attention model created
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
from ast import literal_eval
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, GlobalAveragePooling1D, GlobalMaxPool1D, \
    Dropout, Layer, Dense, MaxPool1D, Concatenate, LayerNormalization, SpatialDropout1D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from basemodel import BaseModel
from banhdanauattention import BahdanauAttention
from attention_layers import Attention, SelfAttention
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, precision_score, \
    recall_score
from nela_features.nela_features import NELAFeatureExtractor
import calendar
# Visualization
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML  # To use with Jupyter Notebook
import seaborn as sns

sns.set()

SEED = 42


class LocalAttentionModelNela(BaseModel):
    '''Class that contain the attention model created for the Proppy database.
    '''

    def __init__(self, batch_size, epochs, optimizer, max_sequence_len, lstm_units,
                 path_train, path_test, vocab_size=None, l2_rate=1e-5, path_dev=None,
                 learning_rate=1e-3, pool_size=4, rate=0.2, filters=64, kernel_size=5,
                 embedding_size=300, max_len=1900, load_embeddings=True, buffer_size=3, emb_type='fasttext',
                 length_type='median', dense_units=128, both_embeddings=False):
        """Init function for the model.
        """
        super(LocalAttentionModelNela, self).__init__(max_len=max_len, path_train=path_train,
                                                      path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                                      embedding_size=embedding_size, emb_type=emb_type,
                                                      vocab_size=vocab_size,
                                                      max_sequence_len=max_sequence_len, epochs=epochs,
                                                      learning_rate=learning_rate,
                                                      optimizer=optimizer, load_embeddings=load_embeddings, rate=rate,
                                                      length_type=length_type, dense_units=dense_units,
                                                      both_embeddings=both_embeddings, buffer_size=buffer_size,
                                                      filters=filters, kernel_size=kernel_size, pool_size=pool_size,
                                                      l2_rate=l2_rate
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

    def attention_context(self, query, key, value):
        """Function that computes the Scaled Dot-Product Attention
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def scaled_dot_product_attention(self, inputs):
        query = Dense(self.lstm_units * 2, name='query')(inputs)
        key = Dense(self.lstm_units * 2, name='key')(inputs)
        values = Dense(self.lstm_units * 2, name='values')(inputs)
        att, weights = self.attention(query, key, values)
        return att, weights

    def scaled_dot_product_attention_second(self, inputs, contextvec):
        query_emb = Dense(self.lstm_units * 2, name='query_emb')(contextvec)
        key_emb = Dense(self.lstm_units * 2, name='key_emb')(inputs)
        values_emb = Dense(self.lstm_units * 2, name='values_emb')(inputs)
        att, weights = self.attention_context(query_emb, key_emb, values_emb)
        return att, weights

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
        embedding_sequence = SpatialDropout1D(0.2)(embedding_sequence)
        # embedding_sequence = Conv1D(self.filters, self.kernel_size, activation='relu')(embedding_sequence)
        # Add the BiLSTM layer and get the states
        x, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(units=self.lstm_units, activation='tanh',
                                                                             return_sequences=True,
                                                                             recurrent_dropout=0,
                                                                             recurrent_activation='sigmoid',
                                                                             unroll=False, use_bias=True,
                                                                             kernel_regularizer=l2(self.l2_rate),
                                                                             return_state=True),
                                                                        name='bilstm')(embedding_sequence)
        # Concatenate hidden states
        # context = Concatenate(axis=1)([forward_h, backward_h, forward_c, backward_c])
        # context = tf.expand_dims(context, 1)
        # print('Concat shape: ', context.shape)
        ################## LOCAL ATTENTION ###################
        # Preparing the input
        hidden_state = Concatenate()([forward_h, backward_h])
        attention_input = [x, hidden_state]
        encoder_output, att_weights = Attention(context='many-to-one',
                                                alignment_type='local-p*',
                                                window_width=100,
                                                score_function='scaled_dot',
                                                name='attention_layer')(attention_input)
        # out = LayerNormalization(epsilon=1e-6)([encoder_output+att_weights])
        ########################## NEW ###############################
        out = Dense(self.dense_units, activation='tanh', kernel_regularizer=l2(self.l2_rate), name='dense')(
            encoder_output)
        out = Dropout(rate=self.rate, name='dropout')(out)
        # out = Flatten(name='Flatten')(out)
        # out = Dropout(rate=self.rate, name='dropout')(out)
        # encoder_output = Flatten()(encoder_output)
        # out = GlobalMaxPool1D()(encoder_output)
        out = GlobalMaxPool1D()(out)
        # concat = Dense(self.dense_units/2, activation='relu', kernel_regularizer=l2(self.l2_rate))(concat)
        ###################### ADDING NELA FEATURES #####################
        nela_input = tf.keras.layers.Input(shape=(126,), dtype="float32", name="nela_input")
        dense_nela = tf.keras.layers.Dense(self.dense_units, activation='relu', kernel_regularizer=l2(self.l2_rate),
                                           name='dense_nela')(nela_input)
        nela_out = Concatenate()([out, dense_nela])
        # final = Concatenate(axis=1)([final, context])
        # Pred layer
        # prediction = Dense(units=2, activation='softmax', name='pred_layer')(out)
        prediction = Dense(units=2, activation='softmax', name='pred_layer')(nela_out)
        self.model = Model(inputs=[sequence_input, nela_input], outputs=[prediction], name='local-attention-model-nela')
        # self.model = Model(inputs=[sequence_input], outputs=[prediction], name='local-attention-model-nela')
        self.model.compile(loss=BinaryCrossentropy(),
                           optimizer=self.optimizer,
                           metrics=self.METRICS)

        self.model.summary()
        # tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=48, to_file='local_attention_model.png')

    def calculate_nela_features(self):
        self.nela_features_train = []
        self.nela_features_test = []
        # Create the Nela Feature extractor
        nela_train = pd.read_csv('../data/train_nela_features.csv')
        nela_test = pd.read_csv('../data/test_nela_features.csv')
        nela_features_train = np.asarray([np.array(literal_eval(feature)) for feature in nela_train['nela_features']], dtype=np.float32)
        nela_train['nela_features'] = nela_features_train
        self.nela_features_train = nela_features_train
        nela_features_test = np.asarray([np.array(literal_eval(feature)) for feature in nela_test['nela_features']], dtype=np.float32)
        nela_test['nela_features'] = nela_features_test
        self.nela_features_test = nela_features_test
        print(len(self.nela_features_test))
        self.X_train = [self.X_train, self.nela_features_train]
        self.X_test = [self.X_test, self.nela_features_test]
        # If we're going for exp 1
        if self.path_dev is not None:
            self.nela_features_dev = []
            nela_dev = pd.read_csv('../data/dev_nela_features.csv')
            nela_features_dev = np.asarray([np.array(literal_eval(feature)) for feature in nela_dev['nela_features']], dtype=np.float32)
            nela_dev['nela_features'] = nela_features_dev
            self.nela_features_dev = nela_features_dev
            print(len(self.nela_features_dev))
            self.X_dev = [self.X_dev, self.nela_features_dev]

    def prepare_data(self):
        super(LocalAttentionModelNela, self).prepare_data()
        self.calculate_nela_features()

    def plot_attention(self):
        if self.model is None:
            raise ValueError("The model must be fitted before using this function.")
        # New model for getting att_weights
        model_att = Model(inputs=self.model.inputs,
                          outputs=[self.model.outputs, self.model.get_layer('attention_layer').output])
        print('Llego aquí en plot_attention')
        # Get a random instance
        idx = np.random.randint(low=0, high=len(self.X_test))
        tokenized_sample = np.trim_zeros(self.X_test[idx])
        pred, attention = model_att.predict(self.X_test[idx:idx + 1], verbose=0)
        pred = pred[0]
        # Get the decoded sentence
        sentence = dict(map(reversed, self.tokenizer.word_index.items()))
        decoded_sentence = [sentence[word] for word in sentence]

        # Get the prediction
        label = np.argmax((pred > 0.5).astype(int).squeeze())
        labelsid = ['Propaganda', 'No Propaganda']

        # Get word attentions using attention vector
        token_attention_dic = {}
        max_score = 0.0
        min_score = 0.0

        attentions_text = attention[0][-len(tokenized_sample):]
        attentions_text = (attentions_text - np.min(attentions_text)) / (
                np.max(attentions_text) - np.min(attentions_text))
        # print(attentions_text[0])
        attentions_text = attentions_text[0]

        for token, att_Score in zip(decoded_sentence, attentions_text):
            token_attention_dic[token] = att_Score

        # Build HTML String to viualize attentions
        # USE THIS ONLY IN Jupyter Notebook
        html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
        for token, attention_t in token_attention_dic.items():
            html_text += "<span style='background-color:{};'>{} <span> ".format(self.attention2color(attention_t[0]),
                                                                                token)

        # Display text enriched with attention scores
        display(HTML(html_text))

        # PLOT PROPAGANDA SCORE
        _labels = ['propaganda', 'no propaganda']
        plt.figure(figsize=(5, 2))
        plt.bar(np.arange(len(_labels)), pred.squeeze(), align='center', alpha=0.5,
                color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
        plt.xticks(np.arange(len(_labels)), _labels)
        plt.ylabel('Scores')
        figname = 'att_plots/attention_score' + str(idx) + '.png'
        plt.savefig(figname)
        plt.show()

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

    def attention2color(self, attention_score):
        r = 255 - int(attention_score * 255)
        color = self.rgb_to_hex((255, r, r))
        return str(color)

    def predict(self):
        """Make the prediction for the test/dev data. It uses the data from the own class.
        """
        # Actually it works as a test function to prove that the code is working.
        print('TEST SET')
        preds = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
        real_preds = []
        for p in preds:
            if p[0] > p[1]:
                real_preds.append(0)
            else:
                real_preds.append(1)
        print(real_preds[:10])
        y_true = []
        for p in self.y_test:
            if p[0] > p[1]:
                y_true.append(0)
            else:
                y_true.append(1)

        print(classification_report(y_true, real_preds))
        print('Roc auc score: ', roc_auc_score(y_true, real_preds))
        print('Accuracy: ', accuracy_score(y_true, real_preds))
        print('Precision-Propaganda: ', precision_score(y_true, real_preds))
        print('Recall-Propaganda: ', recall_score(y_true, real_preds))
        print('F1-Propaganda: ', f1_score(y_true, real_preds))
        print('Macro F1-Propaganda: ', f1_score(y_true, real_preds, average='macro'))

    def predict_dev(self):
        """Make the prediction for the test/dev data. It uses the data from the own class.
                """
        # Actually it works as a test function to prove that the code is working.
        print('DEV SET')
        preds = self.model.predict(self.X_dev, batch_size=self.batch_size, verbose=0)
        real_preds = []
        for p in preds:
            if p[0] > p[1]:
                real_preds.append(0)
            else:
                real_preds.append(1)
        print(real_preds[:10])
        y_true = []
        for p in self.y_dev:
            if p[0] > p[1]:
                y_true.append(0)
            else:
                y_true.append(1)

        print(classification_report(y_true, real_preds))
        print('Roc auc score: ', roc_auc_score(y_true, real_preds))
        print('Accuracy: ', accuracy_score(y_true, real_preds))
        print('Precision-Propaganda: ', precision_score(y_true, real_preds))
        print('Recall-Propaganda: ', recall_score(y_true, real_preds))
        print('F1-Propaganda: ', f1_score(y_true, real_preds))
        print('Macro F1-Propaganda: ', f1_score(y_true, real_preds, average='macro'))
