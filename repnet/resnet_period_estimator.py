import tensorflow as tf
from repnet import TransformerLayer
# Model definition
layers = tf.keras.layers
regularizers = tf.keras.regularizers


def flatten_sequential_feats(x, batch_size, seq_len):
    """Flattens sequential features with known batch size and seq_len."""
    x = tf.reshape(x, [batch_size, seq_len, -1])
    return x


def pairwise_l2_distance(a, b):
    """Computes pairwise distances between all rows of a and all rows of b."""
    norm_a = tf.reduce_sum(tf.square(a), 1)
    norm_a = tf.reshape(norm_a, [-1, 1])
    norm_b = tf.reduce_sum(tf.square(b), 1)
    norm_b = tf.reshape(norm_b, [1, -1])
    dist = tf.maximum(norm_a - 2.0 * tf.matmul(a, b, False, True) + norm_b, 0.0)
    return dist


def get_sims(embs, temperature):
    """Calculates self-similarity between batch of sequence of embeddings."""
    batch_size = tf.shape(embs)[0]
    seq_len = tf.shape(embs)[1]
    embs = tf.reshape(embs, [batch_size, seq_len, -1])

    def _get_sims(embs):
        """Calculates self-similarity between sequence of embeddings."""
        dist = pairwise_l2_distance(embs, embs)
        sims = -1.0 * dist
        return sims

    sims = tf.map_fn(_get_sims, embs)
    sims /= temperature
    sims = tf.nn.softmax(sims, axis=-1)
    sims = tf.expand_dims(sims, -1)
    return sims


class ResnetPeriodEstimator(tf.keras.models.Model):
    """RepNet model."""

    def __init__(
            self,
            num_frames=64,
            image_size=112,
            base_model_layer_name='conv4_block3_out',
            temperature=13.544,
            dropout_rate=0.25,
            l2_reg_weight=1e-6,
            temporal_conv_channels=512,
            temporal_conv_kernel_size=3,
            temporal_conv_dilation_rate=3,
            conv_channels=32,
            conv_kernel_size=3,
            transformer_layers_config=((512, 4, 512),),
            transformer_dropout_rate=0.0,
            transformer_reorder_ln=True,
            period_fc_channels=(512, 512),
            within_period_fc_channels=(512, 512)):
        super(ResnetPeriodEstimator, self).__init__()

        # Model params.
        self.num_frames = num_frames
        self.image_size = image_size

        self.base_model_layer_name = base_model_layer_name

        self.temperature = temperature

        self.dropout_rate = dropout_rate
        self.l2_reg_weight = l2_reg_weight

        self.temporal_conv_channels = temporal_conv_channels
        self.temporal_conv_kernel_size = temporal_conv_kernel_size
        self.temporal_conv_dilation_rate = temporal_conv_dilation_rate

        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        # Transformer config in form of (channels, heads, bottleneck channels).
        self.transformer_layers_config = transformer_layers_config
        self.transformer_dropout_rate = transformer_dropout_rate
        self.transformer_reorder_ln = transformer_reorder_ln

        self.period_fc_channels = period_fc_channels
        self.within_period_fc_channels = within_period_fc_channels

        # Base ResNet50 Model.
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, weights=None, pooling='max')
        self.base_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(self.base_model_layer_name).output)

        # 3D Conv on k Frames
        self.temporal_conv_layers = [
            layers.Conv3D(self.temporal_conv_channels,
                          self.temporal_conv_kernel_size,
                          padding='same',
                          dilation_rate=(self.temporal_conv_dilation_rate, 1, 1),
                          kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                          kernel_initializer='he_normal')]
        self.temporal_bn_layers = [layers.BatchNormalization()
                                   for _ in self.temporal_conv_layers]

        # Counting Module (Self-sim > Conv > Transformer > Classifier)
        self.conv_3x3_layer = layers.Conv2D(self.conv_channels,
                                            self.conv_kernel_size,
                                            padding='same',
                                            activation=tf.nn.relu)

        channels = self.transformer_layers_config[0][0]
        self.input_projection = layers.Dense(
            channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
            activation=None)
        self.input_projection2 = layers.Dense(
            channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
            activation=None)

        length = self.num_frames
        self.pos_encoding = tf.compat.v1.get_variable(
            name='resnet_period_estimator/pos_encoding',
            shape=[1, length, 1],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        self.pos_encoding2 = tf.compat.v1.get_variable(
            name='resnet_period_estimator/pos_encoding2',
            shape=[1, length, 1],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

        self.transformer_layers = []
        for d_model, num_heads, dff in self.transformer_layers_config:
            self.transformer_layers.append(
                TransformerLayer(d_model, num_heads, dff,
                                 self.transformer_dropout_rate,
                                 self.transformer_reorder_ln))

        self.transformer_layers2 = []
        for d_model, num_heads, dff in self.transformer_layers_config:
            self.transformer_layers2.append(
                TransformerLayer(d_model, num_heads, dff,
                                 self.transformer_dropout_rate,
                                 self.transformer_reorder_ln))

        # Period Prediction Module.
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        num_preds = self.num_frames // 2
        self.fc_layers = []
        for channels in self.period_fc_channels:
            self.fc_layers.append(layers.Dense(
                channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                activation=tf.nn.relu))
        self.fc_layers.append(layers.Dense(
            num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

        # Within Period Module
        num_preds = 1
        self.within_period_fc_layers = []
        for channels in self.within_period_fc_channels:
            self.within_period_fc_layers.append(layers.Dense(
                channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                activation=tf.nn.relu))
        self.within_period_fc_layers.append(layers.Dense(
            num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

    def call(self, x):
        # Ensures we are always using the right batch_size during train/eval.
        batch_size = tf.shape(x)[0]
        # Conv Feature Extractor.
        x = tf.reshape(x, [-1, self.image_size, self.image_size, 3])
        x = self.base_model(x)
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        x = tf.reshape(x, [batch_size, -1, h, w, c])

        # 3D Conv to give temporal context to per-frame embeddings.
        for bn_layer, conv_layer in zip(self.temporal_bn_layers,
                                        self.temporal_conv_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = tf.nn.relu(x)

        x = tf.reduce_max(x, [2, 3])

        # Reshape and prepare embs for output.
        final_embs = x

        # Get self-similarity matrix.
        x = get_sims(x, self.temperature)

        # 3x3 conv layer on self-similarity matrix.
        x = self.conv_3x3_layer(x)
        x = tf.reshape(x, [batch_size, self.num_frames, -1])
        within_period_x = x

        # Period prediction.
        x = self.input_projection(x)
        x += self.pos_encoding
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = flatten_sequential_feats(x, batch_size, self.num_frames)
        for fc_layer in self.fc_layers:
            x = self.dropout_layer(x)
            x = fc_layer(x)

        # Within period prediction.
        within_period_x = self.input_projection2(within_period_x)
        within_period_x += self.pos_encoding2
        for transformer_layer in self.transformer_layers2:
            within_period_x = transformer_layer(within_period_x)
        within_period_x = flatten_sequential_feats(within_period_x,
                                                   batch_size,
                                                   self.num_frames)
        for fc_layer in self.within_period_fc_layers:
            within_period_x = self.dropout_layer(within_period_x)
            within_period_x = fc_layer(within_period_x)

        return x, within_period_x, final_embs

    @tf.function
    def preprocess(self, imgs):
        imgs = tf.cast(imgs, tf.float32)
        imgs -= 127.5
        imgs /= 127.5
        imgs = tf.image.resize(imgs, (self.image_size, self.image_size))
        return imgs
