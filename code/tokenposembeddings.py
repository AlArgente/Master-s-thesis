import tensorflow as tf
from tensorflow.keras import layers

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, weights):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[weights], trainable=False)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1] # We took the maxlen from the x layer, 463 in my case if I choose the median as max_sequence_len
        # Now create the range of the positions for the sequence.
        positions = tf.range(start=0, limit=maxlen, delta=1) # Array 0..300 [0,1,2...,298,299,300]
        # Add the positions to the embeddings positions
        positions = self.pos_emb(positions)
        # Add the positions embeddings to the tokens embeddings.
        x = self.token_emb(x)
        return x + positions