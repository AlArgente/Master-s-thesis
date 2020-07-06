from enum import Enum, unique

@unique
class ModelConfig(Enum):
    """All possible configurations for the attention model
    """
    TrainEmbeddings = {
        'batch_size' : 64,
        'epochs' : 5,
        'filters': 64,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 200,
        'lstm_units': 75,
        'path_train' : '../data/train_preprocessed.tsv',
        'path_test': '../data/test_preprocessed.tsv',
        'path_dev': '../data/dev_preprocessed.tsv',
        'vocab_size' : None,
        'learning_rate': 1e-5,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len' : 100000,
        'load_embeddings' : True,
        'buffer_size' : 3,
        'emb_type' : 'fasttext'
    }