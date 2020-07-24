import tensorflow.keras as keras
from tensorflow.keras import layers
from multihead_attlayer import MultiHeadSelfAttention


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """Initializer for the transformer block.
        Args:
            - embed_dim: Embeddings dimension
            - num_heads: num heads for the self attetion layer
            - ff_dim: Units for the dense layer
            - rate: rate for drop_out
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs) # (None, input_seq_len, embeddings_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # (None, input_seq_len, embeddings_dim)
        ffn_output = self.ffn(out1) # (None, input_seq_len, embeddings_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (None, input_seq_len, embeddings_dim)
        return out2