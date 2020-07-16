import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, ElectraForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import BertTokenizer, AlbertTokenizer, ElectraTokenizer, DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification, TFBertForSequenceClassification
from transformers import TFAlbertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import time
import datetime
from transformers import TrainingArguments, Trainer, TFTrainer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


class FineTuningModel:
    """Class with different models to fine-tune.
    The models aviable right now are: BERT - ALBERT - ELECTRA - DistilBERT
    This class uses the transformer package and is avaiable for both TensorFlow and PyTorch packages, but
    here I fisrt I will use PyTorch models.
    """
    def __init__(self, path_train, path_test, path_dev, epochs, max_sequence_len, batch_size, optimizer, tr_size = 0.8,
                 learning_rate = 5e-5, eps = 1e-8, model_to_use='bert', api='tf'):
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
        self.api = api
        if self.model_to_use.lower() == 'bert':
            print('Se usará Bert')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif self.model_to_use.lower() == 'albert':
            print('Se usará Albert')
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1', do_lower_case=False)
            # TODO: Probar con large
        elif self.model_to_use.lower() == 'electra':
            print('Se usará Electra')
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', do_lower_case=True)
        elif self.model_to_use.lower() == 'distilbert':
            print('Se usará Distilbert')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        else:
            print('Model not avaiable right now.')
        self.load_data()

    def load_data(self):
        """Load the 3 DataFrames, train-test-dev. The data here will be preprocessed, previously tokenized,
        stopwords deleted and stem.
        """
        pass

    def __load_data_torch(self):
        """Function to load
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
        self.y_train = self.data_train['label']
        self.y_test = self.data_test['label']
        self.y_dev = self.data_dev['label']

        self.x_train_inputs_ids, self.x_train_attention_mask = self.tokenizer_map(self.train, True)
        self.x_test_inputs_ids, self.x_test_attention_mask = self.tokenizer_map(self.test)
        self.x_dev_inputs_ids, self.x_dev_attention_mask = self.tokenizer_map(self.dev)

        # The train dataset will be used for train and validation
        self.dataset = TensorDataset(self.x_train_inputs_ids, self.x_train_attention_mask, self.y_train)

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        lengths = [train_size, val_size]
        self.train_dataset, self.val_dataset = random_split(dataset=self.dataset, lengths=[train_size, val_size])

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

    def __load_data_tf(self):
        self.train = pd.read_csv(self.path_train, sep='\t')
        self.train.loc[self.train['label'] == -1, 'label'] = 0
        self.test = pd.read_csv(self.path_test, sep='\t')
        self.test.loc[self.test['label'] == -1, 'label'] = 0
        self.dev = pd.read_csv(self.path_dev, sep='\t')
        self.dev.loc[self.dev['label'] == -1, 'label'] = 0
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.train_dataset = self.train_dataset.shuffle(len(self.X_train)).batch(self.batch_size)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.X_dev, self.y_dev))
        self.val_dataset = self.val_dataset.shuffle(len(self.X_dev)).batch(self.batch_size)

    def call(self):
        if self.model_to_use.lower() == 'bert':
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)
            print('Bert Cargado.')
            print(self.model)
        elif self.model_to_use.lower() == 'albert':
            self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v1', num_labels=2,
                                                                         output_attentions=False,
                                                                         output_hidden_states=False)
        elif self.model_to_use.lower() == 'electra':
            self.model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator',
                                                                          num_labels=2,
                                                                          output_attentions=False,
                                                                          output_hidden_states=False)
        elif self.model_to_use.lower() == 'distilbert':
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2,
                                                                             output_attentions=False,
                                                                             output_hidden_states=False)
        else:
            print('Model not avaiable right now.')
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.epsilon)
        self.total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.total_steps)

    def fit(self):
        # TODO: Fit for train and validation
        training_stats = []

        # Measure the total training time for the whole run
        start = time.time()

        # For each epoch...
        for epoch in range(self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.epochs))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            # Put the model into training mode. Don't be mislead--the call to `train` just changes the *mode*,
            # it doesn't *perform* the training.

            # `dropout` and `batchnorm` layers behave differently during training

            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                if step % 50 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader),
                                                                                elapsed))
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the device(gpu in our case) using the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device).to(torch.int64)
                b_input_mask = batch[1].to(self.device).to(torch.int64)
                b_labels = batch[2].to(self.device).to(torch.int64)

                # Always clear any previously calculated gradients before performing a backward pass.
                # PyTorch doesn't do this automatically because accumulating the gradients is 'convenient while
                # training RNNs'.
                # source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                self.model.zero_grad()
                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

                # It returns different numbers of parameters depending on what arguments given and what flags are set.
                # For our useage here, it returns the loss (because we provided labels) and the 'logits'--the model
                # outputs prior to activation.
                loss, logit = self.model(b_input_ids,
                                         # token_type_ids=None,
                                         attention_mask=b_input_mask,
                                         labels=b_labels)
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0 This is to help prevent the 'exploding gradients' problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.

                # The optimizer dictates the 'update rule'--how the parameters are modified based on their gradients, the learning rate, etc.

                self.optimizer.step()

                # Update the learning rate.

                self.scheduler.step()

            # Calculate the average loss over all of the batches.

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # Measure how long this epoch took.

            training_time = self.format_time(time.time() - t0)

            print('')
            print('  Average training loss: {0:.2f}'.format(avg_train_loss))
            print('  Training epcoh took: {:}'.format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on our validation set.
            print('')
            print('Running Validation...')
            t0 = time.time()
            # Put the model in evaluation mode--the dropout layers behave differently during evaluation.

            self.model.eval()

            # Tracking variables

            total_eval_accuracy = 0
            total_eval_loss = 0
            total_eval_f1 = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch

            for batch in self.val_dataloader:
                # Unpack this training batch from our dataloader.

                # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.

                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during the forward pass,
                # since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the 'segment ids', which differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the 'logits' output by the model. The 'logits' are the output values prior to applying an
                    # activation function like the softmax.

                    (loss, logits) = self.model(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels)

                # Accumulate the validation loss.

                total_eval_loss += loss.item()
                # Move logits and labels to CPU

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.

                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
                total_eval_f1 += self.flat_f1(logits, label_ids)

            # Report the final accuracy for this validation run.

            avg_val_accuracy = total_eval_accuracy / len(self.val_dataloader)
            print('  Accuracy: {0:.2f}'.format(avg_val_accuracy))

            # Report the final f1 score for this validation run.

            avg_val_f1 = total_eval_f1 / len(self.val_dataloader)
            print('  F1: {0:.2f}'.format(avg_val_f1))

            # Calculate the average loss over all of the batches.

            avg_val_loss = total_eval_loss / len(self.val_dataloader)
            # Measure how long the validation run took.

            validation_time = self.format_time(time.time() - t0)

            print('  Validation Loss: {0:.2f}'.format(avg_val_loss))
            print('  Validation took: {:}'.format(validation_time))

            # Record all statistics from this epoch.

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Val_F1': avg_val_f1,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        print('')
        print('Training complete!')

        print('Total training took {:} (h:mm:ss)'.format(self.format_time(time.time() - start)))

    def predict(self):
        print('Predicting labels for test sentences...')

        # Put model in evaluation mode
        self.model.eval()
        # Tracking variables

        predictions = []

        # Predict

        for batch in self.test_dataloader:
            # Add batch to device
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)
            logits = outputs[0]

            # Move logits and labels to device

            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(logits)

        print(' DONE.')

    def format_time(self, elapsed):
        """A function that takes a time in seconds and returns a string hh:mm:ss"""

        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def flat_accuracy(self, preds, labels):

        """A function for calculating accuracy scores"""

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat)

    def flat_f1(self, preds, labels):

        """A function for calculating f1 scores"""

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, pred_flat)

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
            # print(encoded_dict)
            inputs_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        inputs_ids = torch.cat(inputs_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        # We only want this for training labels.
        if labels:
            self.y_train = torch.tensor(self.y_train)

        return inputs_ids, attention_mask


    def fit_trainer(self):
        self.training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=0,
            weight_decay=0.01,
            logging_dir=None
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset
        )

        self.trainer.train()
        self.trainer.evaluate()

    def predict_trainer(self):
        self.trainer.predict(self.test_dataset)

    def TFfit_trainer(self):
        pass