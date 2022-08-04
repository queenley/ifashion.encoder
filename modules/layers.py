import math

import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""

    def __init__(self,
                 num_classes,
                 margin=0.5,
                 logist_scale=64,
                 easy_margin=False,
                 **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.easy_margin = easy_margin

    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t**2, name='sin_t')

        cos_mt = tf.subtract(cos_t * tf.cast(self.cos_m, embds.dtype),
                             sin_t * tf.cast(self.sin_m, embds.dtype),
                             name='cos_mt')

        if self.easy_margin:
            cos_mt = tf.where(cos_t > tf.cast(self.th, embds.dtype), cos_mt,
                              cos_t)
        else:
            cos_mt = tf.where(cos_t > tf.cast(self.th, embds.dtype), cos_mt,
                              cos_t - tf.cast(self.mm, embds.dtype))

        # mask = tf.one_hot(tf.cast(labels, tf.int32),
        #                   depth=self.num_classes,
        #                   name='one_hot_mask')
        mask = tf.cast(labels, embds.dtype)

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')
        prob = tf.nn.softmax(logists, name='arcface_prob')

        return prob

    def get_config(self):
        config = super(ArcMarginPenaltyLogists, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'margin': self.margin,
            'logist_scale': self.logist_scale,
            'easy_margin': self.easy_margin,
        })
        return config
