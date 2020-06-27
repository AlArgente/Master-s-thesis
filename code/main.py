"""
Main document
"""
import argparse
import time

from my_attention_model import CNNRNNModel
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='Preprocess or execute the data.', default=None)
    parser.add_argument('--embeddings', type=str, help='Embeddings to use.', default='glove')
    args = vars(parser.parse_args())  # Convert the arguments to a dict
    if args['mode'] == 1:
        train = pd.read_csv('../data/proppy_1.0.train.tsv', sep='\t', header=None)
        train_processed = Preprocessing.pipeline(train[train.columns[0]])
        train_processed_df = pd.DataFrame(columns=['text_stem', 'text_join', 'label'])
        train_processed_df['text_stem'], train_processed_df['text_join'] = train_processed
        train_processed_df['label'] = train[train.columns[len(train.columns) - 1]]
        # train_processed_df['embedding'] = train_embeddings
        train_processed_df.to_csv('../data/train_preprocessed.tsv', sep='\t', index=False,
                                  index_label=False)
        test = pd.read_csv('../data/proppy_1.0.test.tsv', sep='\t', header=None)
        test_processed = Preprocessing.pipeline(test[test.columns[0]])
        test_processed_df = pd.DataFrame(columns=['text_stem', 'text_join', 'label'])
        test_processed_df['text_stem'], test_processed_df['text_join'] = test_processed
        test_processed_df['label'] = test[test.columns[len(test.columns) - 1]]
        # train_processed_df['embedding'] = train_embeddings
        test_processed_df.to_csv('../data/test_preprocessed.tsv', sep='\t', index=False,
                                  index_label=False)
        dev = pd.read_csv('../data/proppy_1.0.dev.tsv', sep='\t', header=None)
        dev_processed = Preprocessing.pipeline(dev[dev.columns[0]])
        dev_processed_df = pd.DataFrame(columns=['text_stem', 'text_join', 'label'])
        dev_processed_df['text_stem'], dev_processed_df['text_join']= dev_processed
        dev_processed_df['label'] = dev[dev.columns[len(dev.columns) - 1]]
        # train_processed_df['embedding'] = train_embeddings
        dev_processed_df.to_csv('../data/dev_preprocessed.tsv', sep='\t', index=False,
                                 index_label=False)
    elif args['mode'] == 2:
        #TODO: CAMBIAR MODOS 2-3 A LA NUEVA MODIFICACIÓN
        # Creación del modelo con embeddings aleatorios, no se pasa conjunto de validación ni de test.
        train = pd.read_csv('../data/train_preprocessed.tsv', sep='\t')
        X_train, y_train = train['text'], train['label']
        config = ModelConfig.TrainEmbeddings.value
        print('Configuration', config)
        model = CNNRNNModel(batch_size=config['batch_size'], epochs=config['epochs'], vocab_size=config['vocab_size'],
                               max_len=config['max_len'], filters=config['filters'], kernel_size=config['kernel_size'],
                               optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                               max_sentence_len=config['max_sentence_len'], lstm_units=config['lstm_units'],
                               embedding_size=config['embedding_size'], embeddings_loaded=config['embeddings_loaded'],
                               pool_size=config['pool_size'])
        model.build(input_shape=X_train.shape[0])
        model.fit(X_train, y_train)
    elif args['mode'] == 3:
        # Creación del modelo con embeddings de fasttext o glove
        train = pd.read_csv('../data/test_preprocessed.tsv', sep='\t')
        X_train, y_train = train['text'], train['label']
        emb = FactoryEmbeddings()
        emb.load_embeddings(args['embeddings'])
        X_train = X_train.apply(lambda x: literal_eval(x))
        X_train_embeddings = emb.embeddings.calc_embeddings(text=X_train)
        print('Shape de los embeddings: ', len(X_train_embeddings[0]))
        print('Shape de los embeddings: ', len(X_train[0]))
        print('Shape de los embeddings: ', len(X_train_embeddings[1]))
        print('Shape de los embeddings: ', len(X_train[1]))
        print('Shape de los embeddings: ', len(X_train_embeddings[2]))
        print('Shape de los embeddings: ', len(X_train[2]))
        print('Shape de los embeddings: ', len(X_train_embeddings[3]))
        print('Shape de los embeddings: ', len(X_train[3]))
        time.sleep(10)
        config = ModelConfig.TrainEmbeddings.value
        print('Configuration', config)
        model = CNNRNNModel(batch_size=config['batch_size'], epochs=config['epochs'], vocab_size=config['vocab_size'],
                               max_len=config['max_len'], filters=config['filters'], kernel_size=config['kernel_size'],
                               optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                               max_sentence_len=config['max_sentence_len'], lstm_units=config['lstm_units'],
                               embedding_size=config['embedding_size'], load_embeddings=config['load_embeddings'],
                               pool_size=config['pool_size'])
        model.build( input_shape=X_train.shape[0])
        model.fit(X_train_embeddings, y_train)
    else:
        print('No other mode implemented yed.')

    elapsed_time = time.time() - start_time

    print('The execution took: ' + str(elapsed_time) + ' seconds.')
    print('End of execution.')

if __name__ == '__main__':
    main()
