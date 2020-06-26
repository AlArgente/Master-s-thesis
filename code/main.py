"""
Main document
"""
import argparse
import time

from my_attention_model import AttentionModel
from preprocessing import Preprocessing
import pandas as pd

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='Preprocess or execute the data.', default=None)
    parser.add_argument('--embeddings', type=str, help='Embeddings to use.', default='glove')
    args = vars(parser.parse_args())  # Convert the arguments to a dict

    if args['mode'] == 1:
        # train = pd.read_csv('../data/proppy_1.0.train.tsv', sep='\t', header=None)
        test = pd.read_csv('../data/proppy_1.0.test.tsv', sep='\t', header=None)
        # dev = pd.read_csv('../data/proppy_1.0.dev.tsv', sep='\t', header=None)
        train_processed = Preprocessing.pipeline1(test[test.columns[0]])
        train_processed_df = pd.DataFrame(columns=['text'])
        train_processed_df['text'] = train_processed
        # train_processed_df['embedding'] = train_embeddings
        train_processed_df.to_csv('../data/test_preprocessed.tsv', sep='\t', index=False, index_label=False)
    elif args['mode'] == 2:
        # train = pd.read_csv('../data/proppy_1.0.train.tsv', sep='\t', header=None)
        test = pd.read_csv('../data/proppy_1.0.test.tsv', sep='\t', header=None)
        # dev = pd.read_csv('../data/proppy_1.0.dev.tsv', sep='\t', header=None)
        train_processed, train_embeddings = Preprocessing.pipeline2(test[test.columns[0]], args['embeddings'])
        train_processed_df = pd.DataFrame(columns=['text', 'embedding'])
        train_processed_df['text'] = train_processed
        train_processed_df['embedding'] = train_embeddings
        train_processed_df.to_csv('../data/test_preprocessed_embeddings.tsv', sep='\t', index=False, index_label=False)
    else:
        print('No other mode implemented yed.')

    elapsed_time = time.time() - start_time

    print('The execution took: ' + str(elapsed_time) + ' seconds.')
    print('End of execution.')

if __name__ == '__main__':
    main()
