import time

import bert
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import Model
from basemodel import BaseModel
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, precision_score, \
    recall_score


class BertModel(BaseModel):
    def __init__(self, max_len, path_train, path_test, path_dev, epochs, learning_rate, optimizer,
                 load_embeddings, batch_size=16, embedding_size='300', emb_type='glove',
                 vocab_size=None, max_sequence_len=None, rate=0.2, trainable=True, length_type='median'):
        super(BertModel, self).__init__(max_len=max_len, path_train=path_train,
                                        path_test=path_test, path_dev=path_dev, batch_size=batch_size,
                                        embedding_size=embedding_size, emb_type=emb_type, vocab_size=vocab_size,
                                        max_sequence_len=max_sequence_len, epochs=epochs, learning_rate=learning_rate,
                                        optimizer=optimizer, load_embeddings=load_embeddings, rate=rate,
                                        length_type=length_type)

        self.trainable = trainable
        if self.length_type == 'fixed':
            pass
        elif self.length_type == 'mean':
            self.max_sequence_len = self.mean_length
        elif self.length_type == 'mode':
            self.max_sequence_len = self.mode_length
        elif self.length_type == 'median':
            self.max_sequence_len = self.median_length
        elif self.length_type == 'max':
            self.max_sequence_len = 512  # Max sequence len for BERT.
        # Prepare the layers for the model
        self.input_word_ids = tf.keras.layers.Input(shape=(self.max_sequence_len,), dtype=tf.int32,
                                                    name="input_word_ids")
        self.input_masks = tf.keras.layers.Input(shape=(self.max_sequence_len,), dtype=tf.int32,
                                                 name="input_mask")
        self.input_type_ids = tf.keras.layers.Input(shape=(self.max_sequence_len,), dtype=tf.int32,
                                                    name="input_type_ids")
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                         trainable=self.trainable)
        self.vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bert.bert_tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)
        _, self.sequence_output = self.bert_layer([self.input_word_ids, self.input_masks,
                                                   self.input_type_ids])

    def call(self):
        self.clf_output = self.sequence_output[:, 0, :]
        self.out = tf.keras.layers.Dense(2, activation='softmax')(self.clf_output)
        self.model = tf.keras.models.Model(inputs=[self.input_word_ids,
                                                   self.input_masks,
                                                   self.input_type_ids], outputs=self.out)
        # self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(optimizer=self.optimizer,
                           loss=BinaryCrossentropy(),
                           metrics=self.METRICS)
        # tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=48, to_file='bertmodel.png')
        self.model.summary()

    def fit(self, with_validation=False):
        if not with_validation:
            self.history = self.model.fit(self.train_inputs,
                                          self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                          shuffle=True, class_weight=self.class_weights)
        else:
            self.history = self.model.fit(self.train_inputs,
                                          self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                          shuffle=True, class_weight=self.class_weights,
                                          validation_data=(self.dev_inputs, self.y_dev))

    def predict(self):
        preds = self.model.predict(self.test_inputs, batch_size=self.batch_size)
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
        print('Precision: ', precision_score(y_true, real_preds))
        print('Recall: ', recall_score(y_true, real_preds))
        print('F1 Global: ', f1_score(y_true, real_preds))

    def load_data(self):
        """Load the data from the paths given. This function override the BaseModel load_data function.
        """
        self.data_train = pd.read_csv(self.path_train, sep='\t')
        self.data_train.loc[self.data_train['label'] == -1, 'label'] = 0
        self.data_test = pd.read_csv(self.path_test, sep='\t')
        self.data_test.loc[self.data_test['label'] == -1, 'label'] = 0
        self.data_dev = pd.read_csv(self.path_dev, sep='\t')
        self.data_dev.loc[self.data_dev['label'] == -1, 'label'] = 0
        self.train = self.data_train.text.values
        self.test = self.data_test.text.values
        self.dev = self.data_dev.text.values
        # Prepare the data for the model.
        # self.train_inputs = self._encoder(self.train)
        # self.test_inputs = self._encoder(self.test)
        # self.dev_inputs = self._encoder(self.dev)
        # self.train_inputs = self._encoder(self.dev) # For a fast test
        # Prepare the data using the bert_encode function
        self.train_inputs = self._bert_encode(self.train)
        self.test_inputs = self._bert_encode(self.test)
        self.dev_inputs = self._bert_encode(self.dev)
        # self.train_inputs = self._bert_encode(self.dev) # For a fast test
        # Labels #
        # self.y_train = self.data_train['label'].values
        self.y_train = tf.keras.utils.to_categorical(self.data_train['label'], num_classes=2)
        # self.y_test = self.data_test['label'].values
        self.y_test = tf.keras.utils.to_categorical(self.data_test['label'], num_classes=2)
        # self.y_dev = self.data_dev['label'].values
        self.y_dev = tf.keras.utils.to_categorical(self.data_dev['label'], num_classes=2)
        # self.y_train = tf.keras.utils.to_categorical(self.data_dev['label'], num_classes=2) # For a fast test

    def _get_masks(self, tokens):
        """Get the mask for the tokens.
        Args:
            tokens: list of string with all the tokens to be masked.
        """
        if len(tokens) > self.max_sequence_len:
            raise IndexError("Token length more than max seq length!")
        return [1] * len(tokens) + [0] * (self.max_sequence_len - len(tokens))

    def _get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second
        """
        if len(tokens) > self.max_sequence_len:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (self.max_sequence_len - len(tokens))

    def _get_ids(self, tokens):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_sequence_len - len(token_ids))
        return input_ids

    def _tokenize_sentence(self, sentence):
        return self.tokenizer.tokenize(text=sentence)

    def _pad_sentence(self, sentence):
        if len(sentence) > self.max_sequence_len:
            sentence = sentence[:self.max_sequence_len - 1]
        return sentence

    def _encoder(self, text):

        inputs_id, input_masks, input_segments = [], [], []

        for sentence in text:
            tokens = self._tokenize_sentence(sentence)
            tokens_padded = self._pad_sentence(tokens)
            inputs_id.append(self._get_ids(tokens=tokens_padded))
            input_masks.append(self._get_masks(tokens=tokens_padded))
            input_segments.append(self._get_segments(tokens=tokens_padded))

        return np.asarray(inputs_id, dtype='int32'), np.asarray(input_masks, dtype='int32'), \
               np.asarray(input_segments, dtype='int32')

    def _encode_sentence(self, sentence):
        tokens = list(self.tokenizer.tokenize(sentence))
        tokens.append('[SEP]')
        return self.tokenizer.convert_tokens_to_ids(tokens)[:self.max_sequence_len - 1]

    def _bert_encode(self, data):
        sentence = tf.ragged.constant([self._encode_sentence(sentence=s) for s in np.array(data)])

        cls = [self.tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence.shape[0]
        input_word_ids = tf.concat([cls, sentence], axis=-1)

        input_mask = tf.ones_like(input_word_ids).to_tensor()

        type_cls = tf.zeros_like(cls)
        type_s = tf.zeros_like(sentence)
        input_type_ids = tf.concat([type_cls, type_s], axis=-1).to_tensor()
        inputs = {
            'input_word_ids': input_word_ids.to_tensor(),
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        }

        return inputs
