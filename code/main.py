"""
Main document
"""
import argparse
import time
import os

from cnnrnn_model import CNNRNNModel
from preprocessing import Preprocessing
import pandas as pd
from modelconfiguration import ModelConfig
from factory_embeddings import FactoryEmbeddings
from ast import literal_eval

def main():
    """
    Hay 3 modos de ejecución,

    Modo 1: Preprocesa los datos, calcula
    """
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='Preprocess or execute the data.', default=None)
    parser.add_argument('--embeddings', type=str, help='Embeddings to use.', default='glove')
    args = vars(parser.parse_args())  # Convert the arguments to a dict
    if args['mode'] == 1:
        train = pd.read_csv('../data/proppy_1.0.train.tsv', sep='\t', header=None)
        train_processed = Preprocessing.pipeline(train[train.columns[0]])
        train_processed_df = pd.DataFrame(columns=['text_stem', 'text', 'label'])
        train_processed_df['text_stem'], train_processed_df['text'] = train_processed
        train_processed_df['label'] = train[train.columns[len(train.columns) - 1]]
        # train_processed_df['embedding'] = train_embeddings
        train_processed_df.to_csv('../data/train_preprocessed.tsv', sep='\t', index=False,
                                  index_label=False)
        test = pd.read_csv('../data/proppy_1.0.test.tsv', sep='\t', header=None)
        test_processed = Preprocessing.pipeline(test[test.columns[0]])
        test_processed_df = pd.DataFrame(columns=['text_stem', 'text', 'label'])
        test_processed_df['text_stem'], test_processed_df['text'] = test_processed
        test_processed_df['label'] = test[test.columns[len(test.columns) - 1]]
        # train_processed_df['embedding'] = train_embeddings
        test_processed_df.to_csv('../data/test_preprocessed.tsv', sep='\t', index=False,
                                  index_label=False)
        dev = pd.read_csv('../data/proppy_1.0.dev.tsv', sep='\t', header=None)
        dev_processed = Preprocessing.pipeline(dev[dev.columns[0]])
        dev_processed_df = pd.DataFrame(columns=['text_stem', 'text', 'label'])
        dev_processed_df['text_stem'], dev_processed_df['text']= dev_processed
        dev_processed_df['label'] = dev[dev.columns[len(dev.columns) - 1]]
        # train_processed_df['embedding'] = train_embeddings
        dev_processed_df.to_csv('../data/dev_preprocessed.tsv', sep='\t', index=False,
                                 index_label=False)
    elif args['mode'] == 2:
        # Creación del modelo con embeddings de fasttext o glove
        config = ModelConfig.TrainEmbeddings.value
        model = CNNRNNModel(batch_size=config['batch_size'], epochs=config['epochs'], vocab_size=config['vocab_size'],
                               max_len=config['max_len'], filters=config['filters'], kernel_size=config['kernel_size'],
                               optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                               max_sequence_len=config['max_sequence_len'], lstm_units=config['lstm_units'],
                               embedding_size=config['embedding_size'], load_embeddings=config['load_embeddings'],
                               pool_size=config['pool_size'], path_train=config['path_train'], path_test=config['path_test'],
                               path_dev=config['path_dev'], emb_type=config['emb_type'], buffer_size = config['buffer_size'])
        model.prepare_data()
        print('Building the model.')
        model.call(300)
        print('Previo a fit')
        model.fit(with_validation=False)
        print('Previo a predict')
        model.predict()
    elif args['mode'] == 3:
        # TODO: CAMBIAR MODO 3 A LA NUEVA MODIFICACIÓN
        # Creación del modelo con embeddings aleatorios, no se pasa conjunto de validación ni de test.
        config = ModelConfig.TrainEmbeddings.value
        print('Configuration', config)
        model = CNNRNNModel(batch_size=config['batch_size'], epochs=config['epochs'], vocab_size=config['vocab_size'],
                               max_len=config['max_len'], filters=config['filters'], kernel_size=config['kernel_size'],
                               optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                               max_sentence_len=config['max_sentence_len'], lstm_units=config['lstm_units'],
                               embedding_size=config['embedding_size'], load_embeddings=config['load_embeddings'],
                               pool_size=config['pool_size'])
        model.call()
        model.fit(with_validation=False)
    else:
        print('No other mode implemented yed.')

    elapsed_time = time.time() - start_time

    print('The execution took: ' + str(elapsed_time) + ' seconds.')
    print('End of execution.')

if __name__ == '__main__':
    main()
