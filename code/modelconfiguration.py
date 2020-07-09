from enum import Enum, unique

@unique
class ModelConfig(Enum):
    """All possible configurations for the attention model
    """
    TrainEmbeddings = {
        'batch_size' : 16,
        'epochs' : 5,
        'filters': 64,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 315, # Mejor resultado 300, segundo mejor 400
        'lstm_units': 25,
        'path_train' : '../data/train_preprocessed.tsv',
        'path_test': '../data/test_preprocessed.tsv',
        'path_dev': '../data/dev_preprocessed.tsv',
        'vocab_size' : None,
        'learning_rate': 5e-5,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len' : 100000,
        'load_embeddings' : True,
        'buffer_size' : 3,
        'emb_type' : 'glove',
        'rate': 0.2
    }

    FineTuning = {
        'batch_size' : 16,
        'epochs' : 6,
        'path_train': '../data/train_balanced.tsv',
        'path_test': '../data/test_balanced.tsv',
        'path_dev': '../data/dev_balanced.tsv',
        'max_sequence_len': 86,
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'eps' : 1e-8,
        'model_to_use' : 'Distilbert',
        'tr_size': 0.8,
    }