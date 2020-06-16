import numpy as np
import pandas as pd
import tensorflow as tf
tf.gfile = tf.io.gfile
import tensorflow_hub as hub
import re
import unicodedata

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from bert import bert_tokenization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Global variables
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
LOAD_WEIGHTS = False
REDUCE_DATA = -1  # number of data
TRAIN, SAVE_CHECKPOINT = True, True
CHECKPOINT_FILEPATH = './checkpoints/'
MAX_SEQ_LENGTH = 28
EPOCHS = 3
BATCH_SIZE = 32

# List of GPUs avaiable:
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Loading files

# TODO: Regular expressions to clean the data

################## Preprecessing functions ##################

def remove_accented_chars(x):
    """
    Delete all the accents from the text.
    """
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

def remove_contractions(text):
    """
    Remove all the possible contractions in the text so we can tokenize it
    better and to delete more stopwords.
    """
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how does",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        " u ": " you ",
        " ur ": " your ",
        " n ": " and "}
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    return text



################### SPLIT TRAIN TEST ###################

X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=TEST_SIZE,
                                                    random_state=143233,
                                                    shuffle=True)


print('Train len: ', len(X_train))
print('Test len: ', len(X_test))


################### MODEL PREPARATION ##################


# Load the train model from TensorFlow Hub:
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)

# Now we prepare the input for BERT.
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

# Input for train-test
train_input = bert_encode(tokenizer, X_train, max_len=MAX_SEQ_LENGTH)
test_input = bert_encode(tokenizer, X_test, max_len=MAX_SEQ_LENGTH)

model = build_model(bert_layer, MAX_SEQ_LENGTH)
model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='max')

# Checkpoints to save the best weights

model_checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

if LOAD_WEIGHTS:
    model.load_weights(CHECKPOINT_FILEPATH)

if TRAIN:
    # n_time = time.time()
    if SAVE_CHECKPOINT:
        model.fit(
            train_input,
            y_train.values,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, model_checkpoint_callback]
        )
    else:
        model.fit(
            train_input,
            y_train.values,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping]
        )

# Save the best model to the path indicated
if SAVE_CHECKPOINT:
    model.load_weights(CHECKPOINT_FILEPATH)

################# Predict and results #################
# Make the prediction
y_pred = model.predict(test_input)
# Round the prediction
y_pred = np.round(y_pred).astype(np.int32)

# Results obtained:
print(classification_report(y_test, y_pred, digits=4))
# Roc Curve
print(roc_auc_score(y_test, y_pred))
