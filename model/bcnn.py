#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from model.base import BaseModel


class BCNN(BaseModel):
    def __init__(self, config, mode):
        self.regularizer = tf.contrib.layers.l2_regularizer(config.l2_reg)
        super(BCNN, self).__init__(config, mode)
                    

    @staticmethod
    def _cal_similarity(x, y, method="cosine"):
        score = None
        if method.lower() == "cosine":
            score = tf.div(
                tf.reduce_sum(x * y, 1),
                tf.sqrt(tf.reduce_sum(x * x, 1)) * tf.sqrt(tf.reduce_sum(y * y, 1)) + 1e-6, 
                name="cosine_similarity"
            )
        elif method.lower() == "euclidean":
            score = tf.sqrt(
                tf.reduce_sum(tf.square(x - y), 1), 
                name="euclidean_similarity"
            )
        else:
            raise ValueError("invalid method, expected `cosine` or `euclidean`, found{}".format(method))
        return score

    @staticmethod
    def _wide_conv_pad(x, padding_size):
        # w: filter size
        return tf.pad(x, [[0, 0], [padding_size - 1, padding_size - 1], [0, 0], [0, 0]], name="wide_conv_pad")

    def _conv_layer(self, x, filter_size):
        # dynamic shape
        dim = x.get_shape().as_list()[2]

        conv = tf.layers.conv2d(
            inputs=x,
            filters=self.config.num_filters,
            kernel_size=(filter_size, dim),
            padding="VALID",
            activation=tf.nn.tanh,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            kernel_regularizer=self.regularizer,
            bias_initializer=tf.constant_initializer(0.01),
        ) 
        conv = tf.transpose(conv, [0, 1, 3, 2], name="conv_trans")
        return conv

    @staticmethod
    def _pooling_layer(x, pool_size, method="avg"):
        if method.lower() == "avg":
            pooling = tf.layers.average_pooling2d(inputs=x, pool_size=(pool_size, 1), strides=1, name='avg_pooling')
        elif method.lower() == "max":
            pooling = tf.layers.max_pooling2d(inputs=x, pool_size=(pool_size, 1), strides=1, name='max_pooling')
        else:
            raise ValueError("invalid pooling type, expected `avg` or `max`, found {}".format(method))
        return pooling

    # contain conv layer and pooling layer
    def _cnn_block(self, x1, x2, filter_size, num_layers):
        s1 = self.config.s1_max_len
        s2 = self.config.s2_max_len
        out1, out2 = [], []

        for i in range(num_layers):
            with tf.variable_scope("windows-{}/cnn-block-{}".format(filter_size, i + 1)):
                            
                # s + w - 1 + w - 1
                padded1 = self._wide_conv_pad(x=x1, padding_size=filter_size)
                padded2 = self._wide_conv_pad(x=x2, padding_size=filter_size)
                            
                # s + 2w - 2 - w + 1 = s + w - 1
                conv1 = self._conv_layer(x=padded1, filter_size=filter_size)
                conv2 = self._conv_layer(x=padded2, filter_size=filter_size)

                # s + w - 1 - w + 1 = s
                pool1 = self._pooling_layer(x=conv1, pool_size=filter_size, method='avg')
                pool2 = self._pooling_layer(x=conv2, pool_size=filter_size, method='avg')


                # s + w - 1 - (s + w - 1) + 1 = 1
                # b 1 d
                all_pool1 = self._pooling_layer(x=conv1, pool_size=s1 + filter_size - 1, method='avg')
                all_pool2 = self._pooling_layer(x=conv2, pool_size=s2 + filter_size - 1, method='avg')

                # b d
                out1.append(tf.squeeze(all_pool1, axis=[1, 3]))
                out2.append(tf.squeeze(all_pool2, axis=[1, 3]))

                x1 = pool1
                x2 = pool2

        return out1, out2

    def _build_logits(self):

        # here, output of all layers will be considered
        # there will be num_layers*num_filters scores
        scores = []

        for i, filter_size in enumerate(map(int, self.config.filter_sizes.split(','))):
            self.out1, self.out2 = self._cnn_block(self.x1, self.x2, filter_size, self.config.num_layers)

            # here, zip, for each item in the batch
            for r1, r2 in zip(self.out1, self.out2):
                score = self._cal_similarity(r1, r2, self.config.similar_method)
                scores.append(tf.expand_dims(score, 1))

        # scores: [(b, 1), (b, 1), (b, 1)]
        
        concat_scores = tf.concat(scores, axis=1)

        # concat_scores: [(b, num_layers*num_filters)]
        logits = tf.layers.dense(
            concat_scores, 2,
            bias_initializer=tf.constant_initializer(0.1),
            name="softmax",
        )

        return logits







