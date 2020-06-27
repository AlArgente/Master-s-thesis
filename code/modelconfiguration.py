from enum import Enum, unique

@unique
class ModelConfig(Enum):
    """All possible configurations for the attention model
    """
    TrainEmbeddings = {
        'batch_size' : 32,
        'epochs' : 40,
        'vocab_size' : 2000,
        'max_len' : 10,
        'filters' : 64,
        'kernel_size':5,
        'optimizer' : 'adam',
        'learning_rate' : 1e-3,
        'max_sentence_len' : 128,
        'lstm_units' : 25,
        'pool_size' : 2,
        'embedding_size' : 300,
        'load_embeddings' : True
    }