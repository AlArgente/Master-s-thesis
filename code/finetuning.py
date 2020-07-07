import pandas as pd
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, ElectraForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import BertTokenizer, AlbertTokenizer, ElectraTokenizer, DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification, TFBertForSequenceClassification
from transformers import TFAlbertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

class FineTuningModels:
    """Class with different models to fine-tune.
    The models aviable right now are: BERT - ALBERT - ELECTRA - DistilBERT
    This class uses the transformer package and is avaiable for both TensorFlow and PyTorch packages, but
    here I fisrt I will use PyTorch models.
    """
    def __init__(self, path_train, path_test, path_dev, epochs, max_sequence_len, batch_size, optimizer, tr_size = 0.8,
                 learning_rate = 5e-5, eps = 1e-8, model_to_use='bert'):
        self.path_train = path_train
        self.path_test = path_test
        self.path_dev = path_dev
        self.epochs = epochs
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.tr_size = tr_size
        self.learning_rate = learning_rate
        self.epsilon = eps
        self.model_to_use = model_to_use
        self.device = torch.device('cpu')
        if model_to_use.lower() == 'bert':
            print('Se usará Bert')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        elif model_to_use.lower() == 'albert':
            print('Se usará Albert')
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1', do_lower_case=False)
            # TODO: Probar con large
        elif model_to_use.lower() == 'electra':
            print('Se usará Electra')
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', do_lower_case=True)
        elif model_to_use.lower() == 'distilbert':
            print('Se usará Distilbert')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.load_data()

    def load_data(self):
        """Load the 3 DataFrames, train-test-dev. The data here will be preprocessed, previously tokenized,
        stopwords deleted and stem.
        """
        self.data_train = pd.read_csv(self.path_train, sep='\t')
        self.data_train.loc[self.train['label'] == -1, 'label'] = 0
        self.data_test = pd.read_csv(self.path_test, sep='\t')
        self.data_test.loc[self.test['label'] == -1, 'label'] = 0
        self.data_dev = pd.read_csv(self.path_dev, sep='\t')
        self.data_dev.loc[self.dev['label'] == -1, 'label'] = 0
        self.train = self.data_train.text.values
        self.test = self.data_test.text.values
        self.dev = self.data_dev.text.values
        self.y_train = self.data_train['labels']
        self.y_test = self.data_test['labels']
        self.y_dev = self.data_dev['labels']

        self.x_train_inputs_ids, self.x_train_attention_mask = self.tokenizer_map(self.train, True)
        self.x_test_inputs_ids, self.x_test_attention_mask = self.tokenizer_map(self.test)
        self.x_dev_inputs_ids, self.x_dev_attention_mask = self.tokenizer_map(self.dev)

        # The train dataset will be used for train and validation
        self.dataset = TensorDataset(self.x_train_inputs_ids, self.x_train_attention_mask, self.y_train)

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset)- train_size

        self.train_dataset, self.val_dataset = random_split(dataset=self.dataset, [train_size, val_size])

        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.batch_size
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.batch_size
        )
        # Generate the dataset for test data
        self.test_dataset = TensorDataset(self.x_test_inputs_ids, self.x_test_attention_mask)
        self.test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.batch_size
        )
        # Generate the dataset for the dev data
        self.dev_dataset = TensorDataset(self.x_dev_inputs_ids, self.x_dev_attention_mask)
        self.dev_sampler = SequentialSampler(self.dev_dataset)
        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            sampler=self.dev_sampler,
            batch_size=self.batch_size
        )

    def call(self):
        if self.model_to_use.lower == 'bert':
            self.model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)
        elif self.model_to_use == 'albert':
            self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v1', num_labels=2,
                                                                         output_attentions=False,
                                                                         output_hidden_states=False)
        elif self.model_to_use == 'electra':
            self.model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator',
                                                                          num_labels=2,
                                                                          output_attentions=False,
                                                                          output_hidden_states=False)
        elif self.model_to_use == 'distilbert':
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2,
                                                                             output_attentions=False,
                                                                             output_hidden_states=False)

        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.epsilon)
        self.total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.total_steps)

    def fit(self):
        # TODO: Fit for train and validation
        pass


    def tokenizer_map(self, text, labels=False):
        inputs_ids = []
        attention_mask = []

        for sequence in text:
            encoded_dict = self.tokenizer.encode_plus(
                sequence,
                add_special_tokens=True,
                truncation='longest_first',
                max_length=self.max_sequence_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            inputs_ids.append(encoded_dict['inputs_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        inputs_ids = torch.cat(inputs_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        # We only want this for training labels.
        if labels:
            self.y_train = torch.tensor(self.y_train)

        return inputs_ids, attention_mask