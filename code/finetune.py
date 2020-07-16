import time

import torch
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import BertForSequenceClassification, AlbertForSequenceClassification, \
    ElectraForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import BertTokenizer, AlbertTokenizer, ElectraTokenizer, DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification, TFBertForSequenceClassification
from transformers import TFAlbertForSequenceClassification
from transformers import BertConfig, AlbertConfig, ElectraConfig, DistilBertConfig
from tensorflow.keras.optimizers import Adam, RMSprop
from transformers import AdamW, get_linear_schedule_with_warmup, AdamWeightDecay
from transformers import TrainingArguments, Trainer, TFTrainer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


class FineTuningModel:
    def __init__(self, path_train, path_test, path_dev, epochs, max_sequence_len, batch_size, optimizer,
                 tr_size=0.8, emb_type='fasttext', max_len=100000,
                 learning_rate=5e-5, eps=1e-8, model_to_use='bert', api='tf'):
        self.path_train = path_train
        self.path_test = path_test
        self.path_dev = path_dev
        self.epochs = epochs
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size
        self.tr_size = tr_size
        self.learning_rate = learning_rate
        self.epsilon = eps
        self.model_to_use = model_to_use
        self.device = torch.device('cpu')
        self.api = api
        self.emb_type = emb_type
        self.max_len = max_len
        self.select_tokenizer()
        self.__select_optimizer()
        self.pos = 5737
        self.neg = 45557
        self.total = 51294
        self.weight_for_0 = (1 / self.neg) * (self.total) / 2.0
        self.weight_for_1 = (1 / self.pos) * (self.total) / 2.0
        self.class_weights = {0: self.weight_for_0, 1: self.weight_for_1}
        print('Weight for class 0: {:.2f}'.format(self.weight_for_0))
        print('Weight for class 1: {:.2f}'.format(self.weight_for_1))
        self.METRICS = [
            # tf.keras.metrics.TruePositives(name='tp'),
            # tf.keras.metrics.FalsePositives(name='fp'),
            # tf.keras.metrics.TrueNegatives(name='tn'),
            # tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]

    def __select_optimizer(self):
        if self.api == 'torch':
            # self.optimizer = AdamW(lr=learning_rate, betas=[0.9, 0.98], eps=eps, params=None)
            print('Not implemented yet.')
        elif self.api == 'tf':
            self.optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def select_tokenizer(self):
        """Function to prepare the Tokenizer
        """
        if self.model_to_use.lower() == 'bert':
            print('Se usará Bert')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                                           add_special_tokens=True, max_length=self.max_sequence_len,
                                                           pad_to_max_length=True)
        elif self.model_to_use.lower() == 'albert':
            print('Se usará Albert')
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1', do_lower_case=True,
                                                             add_special_tokens=True, max_length=self.max_sequence_len,
                                                             pad_to_max_length=True)
        elif self.model_to_use.lower() == 'electra':
            print('Se usará Electra')
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator',
                                                              do_lower_case=True,
                                                              add_special_tokens=True, max_length=self.max_sequence_len,
                                                              pad_to_max_length=True)
        elif self.model_to_use.lower() == 'distilbert':
            print('Se usará Distilbert')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                                 add_special_tokens=True,
                                                                 max_length=self.max_sequence_len,
                                                                 pad_to_max_length=True)
        else:
            print('Model not avaiable yet.')

    def call(self, api=None):
        self.api = api if api is not None else self.api
        if self.api == 'torch':
            self.__call_model_torch()
        elif self.api == 'tf':
            self.__call_model_tf()
            input_ids = tf.keras.layers.Input(shape=(self.max_sequence_len,), name='input_token', dtype='int32')
            input_masks_ids = tf.keras.layers.Input(shape=(self.max_sequence_len,), name='masked_token', dtype='int32')
            X = self.model(input_ids, input_masks_ids)
            self.model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=X)
            self.model.summary()
            self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                               optimizer=self.optimizer,
                               metrics=self.METRICS)
        else:
            print('The api must be "torch" or "tf".')

    def __call_model_torch(self):
        if self.model_to_use.lower() == 'bert':
            self.config = BertConfig(num_labels=2)
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=self.config)
        elif self.model_to_use.lower() == 'albert':
            self.config = AlbertConfig(num_labels=2)
            self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v1', config=self.config)
        elif self.model_to_use.lower() == 'electra':
            self.config = ElectraConfig(num_labels=2)
            self.model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator',
                                                                          config=self.config)
        elif self.model_to_use.lower() == 'distilbert':
            self.config = DistilBertConfig(num_labels=2)
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                             config=self.config)
        else:
            print('Model not avaiable yet.')

    def __call_model_tf(self):
        if self.model_to_use.lower() == 'bert':
            self.config = BertConfig(num_labels=2)
            self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=self.config)
        elif self.model_to_use.lower() == 'albert':
            self.config = AlbertConfig(num_labels=2)
            self.model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v1', config=self.config)
        elif self.model_to_use.lower() == 'electra':
            print('Electra not avaiable for sequence classification with Tensorflow yet.')
        elif self.model_to_use.lower() == 'distilbert':
            self.config = DistilBertConfig(num_labels=2)
            self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                               config=self.config)
        else:
            print('Model not avaiable yet.')

    def load_data(self, api=None):
        """Function that load the data
        """
        self.api = api if api is not None else self.api

        self.data_train = pd.read_csv(self.path_train, sep='\t')
        self.data_train.loc[self.data_train['label'] == -1, 'label'] = 0
        self.data_test = pd.read_csv(self.path_test, sep='\t')
        self.data_test.loc[self.data_test['label'] == -1, 'label'] = 0
        self.data_dev = pd.read_csv(self.path_dev, sep='\t')
        self.data_dev.loc[self.data_dev['label'] == -1, 'label'] = 0
        self.train = self.data_train.text.values
        self.test = self.data_test.text.values
        self.dev = self.data_dev.text.values

        # Tokenize the data for the transformer model
        self.train = self._tokenize(self.train)
        self.test = self._tokenize(self.test)
        self.dev = self._tokenize(self.dev)

        # We have to prepare the dataset in different ways for torch or tensorflow, so we prepare the tensors
        # for the api selected.
        if self.api == 'torch':
            self.__load_data_torch()
        elif self.api == 'tf':
            self.__load_data_tf()
        else:
            print('The api must be "torch" or "tf".')

    def __load_data_torch(self):
        print('Not implemented yet.')

    def __load_data_tf(self):
        """Prepare the data for the execution with tensorflow.
        """
        self.y_train = tf.keras.utils.to_categorical(self.data_train['label'], num_classes=2)
        self.y_test = tf.keras.utils.to_categorical(self.data_test['label'], num_classes=2)
        self.y_dev = tf.keras.utils.to_categorical(self.data_dev['label'], num_classes=2)

    def _tokenize(self, text):
        """Tokenize the data with the tokenizer for the model selected.
        """
        inputs_id, input_masks, input_segments = [], [], []

        for sentence in text:
            inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_sequence_len,
                                                pad_to_max_length=True, return_attention_mask=True, truncation=True,
                                                return_token_type_ids=True)

            inputs_id.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])

        return np.asarray(inputs_id, dtype='int32'), np.asarray(input_masks, dtype='int32'), \
               np.asarray(input_segments, dtype='int32')

    def fit(self):
        # It gives error fitting the model.
        if self.api == 'torch':
            self.__fit_torch()
        elif self.api == 'tf':
            self.__fit_tf()

    def __fit_torch(self):
        print('Not implemented yet.')

    def __fit_tf(self):
        self.history = self.model.fit(self.train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                      verbose=1, shuffle=True, class_weight=self.class_weights)
