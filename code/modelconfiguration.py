from enum import Enum, unique

@unique
class ModelConfig(Enum):
    """All possible configurations for the attention model
    """
    TrainEmbeddings = {
        'batch_size' : 16,
        'epochs' : 25,
        'filters': 64,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 650,
        'lstm_units': 64,
        'path_train' : '../data/train_raw.tsv',
        'path_test': '../data/test_raw.tsv',
        'path_dev': '../data/dev_raw.tsv',
        'vocab_size' : None,
        'learning_rate': 1e-3,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len' : 100000,
        'load_embeddings' : True,
        'buffer_size' : 3,
        'emb_type' : 'fasttext',
        'rate': 0.05,
        'length_type': 'fixed',
        'dense_units': 128,
        'concat': False,
        'l2_rate':1e-5
    }

    SecondExperiment = {
        'batch_size' : 16,
        'epochs' : 25,
        'filters': 64,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 650,
        'lstm_units': 64,
        'path_train' : '../data/train_exp_2.tsv',
        'path_test': '../data/test_exp_2.tsv',
        'vocab_size' : None,
        'learning_rate': 1e-3,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len' : 100000,
        'load_embeddings' : True,
        'buffer_size' : 3,
        'emb_type' : 'fasttext',
        'rate': 0.15,
        'length_type': 'fixed',
        'dense_units': 128,
        'concat': False,
        'l2_rate': 1e-5
    }

    TransformerConfig = {
        'batch_size': 16,
        'epochs': 25,
        'filters': 64,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 650,
        'lstm_units': 64,
        'path_train': '../data/train_raw.tsv',
        'path_test': '../data/test_raw.tsv',
        'path_dev': '../data/dev_raw.tsv',
        'vocab_size': None,
        'learning_rate': 5e-5,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len': 100000,
        'load_embeddings': True,
        'buffer_size': 3,
        'emb_type': 'fasttext',
        'rate': 0.15,
        'length_type': 'fixed',
        'dense_units': 64,
        'attheads': 12,
        'att_layers': 2,
        'l2_rate': 1e-5
    }

    AttentionConfig = {
        'batch_size': 16,
        'epochs': 25,
        'filters': 64,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 650,
        'lstm_units': 64,
        'path_train': '../data/train_raw.tsv',
        'path_test': '../data/test_raw.tsv',
        'path_dev': '../data/dev_raw.tsv',
        'vocab_size': None,
        'learning_rate': 1e-3,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len': 100000,
        'load_embeddings': True,
        'buffer_size': 3,
        'emb_type': 'fasttext',
        'rate': 0.05,
        'length_type': 'fixed',
        'dense_units': 128,
        # 'both_embeddings': False,
        'att_units': 300,
        'l2_rate': 1e-5
    }

    FineTuning = {
        'batch_size' : 16,
        'epochs' : 6,
        'path_train': '../data/train_raw.tsv',
        'path_test': '../data/test_raw.tsv',
        'path_dev': '../data/dev_raw.tsv',
        'max_sequence_len': 463,
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'eps' : 1e-8,
        'model_to_use' : 'BERT',
        'tr_size': 0.8,
        'api': 'tf',
        'length_type': 'fixed'
    }

    BertConfig = {
        'batch_size': 16,
        'epochs': 3,
        'optimizer': 'adam',
        'max_sequence_len': 650,  # Mejor resultado 300, segundo mejor 400 (pruebas: 280,315,320,330,350,500)
        'path_train': '../data/train_raw.tsv',
        'path_test': '../data/test_raw.tsv',
        'path_dev': '../data/dev_raw.tsv',
        'vocab_size': None,
        'learning_rate': 5e-5,
        'pool_size': 2,
        'embedding_size': 463,
        'max_len': 100000,
        'load_embeddings': True,
        'buffer_size': 3,
        'emb_type': 'fasttext',
        'rate': 0.5,
        'trainable': True,
        'length_type': 'fixed'
    }

    BertConfigSecondExp = {
        'batch_size': 16,
        'epochs': 3,
        'optimizer': 'adam',
        'max_sequence_len': 650,  # Mejor resultado 300, segundo mejor 400 (pruebas: 280,315,320,330,350,500)
        'path_train' : '../data/train_exp_2.tsv',
        'path_test': '../data/test_exp_2.tsv',
        'vocab_size': None,
        'learning_rate': 5e-5,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len': 100000,
        'load_embeddings': True,
        'buffer_size': 3,
        'emb_type': 'fasttext',
        'rate': 0.5,
        'trainable': True,
        'length_type': 'fixed'
    }

    BertBiLSTMConfigSecondExp = {
        'batch_size': 16,
        'epochs': 3,
        'optimizer': 'adam',
        'max_sequence_len': 650,  # Mejor resultado 300, segundo mejor 400 (pruebas: 280,315,320,330,350,500)
        'path_train': '../data/train_exp_2.tsv',
        'path_test': '../data/test_exp_2.tsv',
        'vocab_size': None,
        'learning_rate': 5e-5,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len': 100000,
        'load_embeddings': True,
        'buffer_size': 3,
        'emb_type': 'fasttext',
        'rate': 0.5,
        'trainable': True,
        'length_type': 'fixed',
        'lstm_units': 64
    }

    MeanModelConfig = {
        'batch_size': 16,
        'epochs': 25,
        'filters': 128,
        'kernel_size': 5,
        'optimizer': 'adam',
        'max_sequence_len': 650,
        'lstm_units': 64,
        'path_train': '../data/train_raw.tsv',
        'path_test': '../data/test_raw.tsv',
        'path_dev': '../data/dev_raw.tsv',
        'vocab_size': None,
        'learning_rate': 1e-3,
        'pool_size': 2,
        'embedding_size': 300,
        'max_len': 100000,
        'load_embeddings': True,
        'buffer_size': 3,
        'emb_type': 'fasttext',
        'rate': 0.2,
        'length_type': 'fixed',
        'dense_units': 128,
        'l2_rate': 1e-5
    }