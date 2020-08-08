import tensorflow as tf

# Model definition
layers = tf.keras.layers
regularizers = tf.keras.regularizers


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


# Transformer from https://www.tensorflow.org/tutorials/text/transformer .
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    outputs: shape == (..., seq_len_q, depth_v)
    attention_weights: shape == (..., seq_len_q, seq_len_k)
  """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk.
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    outputs = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return outputs, attention_weights


class TransformerLayer(tf.keras.layers.Layer):
    """Implements a single transformer layer (https://arxiv.org/abs/1706.03762).
  """

    def __init__(self, d_model, num_heads, dff,
                 dropout_rate=0.1,
                 reorder_ln=False):
        super(TransformerLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.reorder_ln = reorder_ln

    def call(self, x):
        inp_x = x

        if self.reorder_ln:
            x = self.layernorm1(x)

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask=None)
        attn_output = self.dropout1(attn_output)

        if self.reorder_ln:
            out1 = inp_x + attn_output
            x = out1
        else:
            # (batch_size, input_seq_len, d_model)
            out1 = self.layernorm1(x + attn_output)
            x = out1

        if self.reorder_ln:
            x = self.layernorm2(x)

        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)

        if self.reorder_ln:
            out2 = out1 + ffn_output
        else:
            # (batch_size, input_seq_len, d_model)
            out2 = self.layernorm2(out1 + ffn_output)

        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(
            q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        outputs = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return outputs, attention_weights
