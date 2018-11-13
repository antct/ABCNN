#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from model.bcnn import BCNN


class ABCNN(BCNN):
    def __init__(self, config, mode):
        super(ABCNN, self).__init__(config, mode)

    def _attention_matrix(self, x1, x2):
        with tf.name_scope('attention_matrix'):
            x1 = tf.squeeze(x1, axis=3)
            x2 = tf.squeeze(x2, axis=3)
            # x1: (b, s1, d), x2: (b, s2, d)		
            # x1: (b, s1, d), x2T: (b, d, s2)
            # x1x2: (b, s2, s2)
            x1_x2 = tf.matmul(x1, x2, transpose_a=False, transpose_b=True)
            _, s1, s2 = x1_x2.get_shape().as_list()

            # x1_sum: (b, s1, 1), x2_sum(b, s2, 1)
            x1_x1 = tf.reduce_sum(x1*x1, axis=2, keep_dims=True)
            x2_x2 = tf.reduce_sum(x2*x2, axis=2, keep_dims=True)
    
            x1_x1 = tf.tile(x1_x1, [1, 1, s2])
            x2_x2 = tf.tile(x2_x2, [1, 1, s1])
            # x1_tile: (b, s2, s2), x2_tile: (b, s2, s2)

            d_d = x1_x1 + tf.matrix_transpose(x2_x2) - 2.0 * x1_x2		
            # here, sometimes the number will be nan, confirm all numbers are not negative
            d_d = tf.maximum(tf.zeros_like(d_d), d_d)
    
            d = tf.sqrt(d_d)

            a = tf.div(1.0, 1 + d)
        return a
    
    def _attention_feature_map(self, x1, x2):
        _, _, dim, _ = x1.get_shape().as_list()
        a = self._attention_matrix(x1, x2)
        w1 = tf.get_variable(
                name='W1',
                shape=(self.config.s2_max_len, dim),
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=self.regularizer
            )
        w2 = tf.get_variable(
                name='W2',
                shape=(self.config.s1_max_len, dim),
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=self.regularizer
            )

        w1 = tf.expand_dims(w1, axis=0)
        w1 = tf.tile(w1, [self.config.batch_size, 1, 1])

        w2 = tf.expand_dims(w2, axis=0)
        w2 = tf.tile(w2, [self.config.batch_size, 1, 1])

        f1 = tf.matmul(a, w1)
        f1 = tf.expand_dims(f1, axis=-1)

        f2 = tf.matmul(a, w2, transpose_a=True)
        f2 = tf.expand_dims(f2, axis=-1)

        return f1, f2


    def _attention_pooling_layer(self, x, w, filter_size, method='avg'):
        s = x.get_shape().as_list()[1] - filter_size + 1
            
        poolings = []

        # x1: (b, s2+w-1, d, ?), w1: (b, s2+w-1, 1, 1)
        for i in range(s):
            if method == 'avg':
                pooling = tf.reduce_mean(x[:, i:i+filter_size, :, :] * w[:, i:i+filter_size, :, :], axis=1, keep_dims=True)
                poolings.append(pooling)
            elif method == 'max':
                pooling = tf.reduce_max(x[:, i:i+filter_size, :, :] * w[:, i:i+filter_size, :, :], axis=1, keep_dims=True)
                poolings.append(pooling)
        attention_pooling = tf.concat(poolings, axis=1, name='attention_pooling')
        return attention_pooling
    


    def _attention_pooling_weights(self, x1, x2):
        # x1: (b, s+w-1, d, ?)
        a = self._attention_matrix(x1, x2)
        w1 = tf.reduce_sum(a, axis=2)
        w2 = tf.reduce_sum(a, axis=1)

        # attention_w1: (b, s2+w-1), attention_w2: (b, s2+w-1)
        w1 = tf.expand_dims(w1, axis=-1)
        w1 = tf.expand_dims(w1, axis=-1)
        w2 = tf.expand_dims(w2, axis=-1)
        w2 = tf.expand_dims(w2, axis=-1)
        return w1, w2


    # overwrite, add attention
    def _cnn_block(self, x1, x2, filter_size, num_layers):
        s1, s2 = self.config.s1_max_len, self.config.s2_max_len
        out1, out2 = [], []
        # ABCNN 1: add input feature map
        # ABCNN 2: add output attention pooling
        # ABCNN 3: 1 & 2
        
        # here, first feature map position
        # maybe once


        for i in range(1):
            with tf.variable_scope("window-{}/cnn-block-{}".format(filter_size, i+1)):
                if self.config.model_type == 1 or self.config.model_type == 3:
                    f1, f2 = self._attention_feature_map(x1, x2)
                    x1 = tf.concat([x1, f1], axis=3) 
                    x2 = tf.concat([x2, f2], axis=3)
                    

                padded1 = self._wide_conv_pad(x=x1, padding_size=filter_size)
                padded2 = self._wide_conv_pad(x=x2, padding_size=filter_size)

                conv1 = self._conv_layer(x=padded1, filter_size=filter_size)
                conv2 = self._conv_layer(x=padded2, filter_size=filter_size)
                
                # conv1: (b, s2+w-1, d, ?)
                if self.config.model_type == 2 or self.config.model_type == 3:
                    w1, w2 = self._attention_pooling_weights(conv1, conv2)
                    pool1 = self._attention_pooling_layer(conv1, w1, filter_size)
                    pool2 = self._attention_pooling_layer(conv2, w2, filter_size)
                else:
                    pool1 = self._pooling_layer(x=conv1, pool_size=filter_size)
                    pool2 = self._pooling_layer(x=conv2, pool_size=filter_size)

                all_pool1 = self._pooling_layer(x=conv1, pool_size=s1+filter_size-1)
                all_pool2 = self._pooling_layer(x=conv2, pool_size=s2+filter_size-1)

                out1.append(tf.squeeze(all_pool1, axis=[1, 3])) 
                out2.append(tf.squeeze(all_pool2, axis=[1, 3]))

                x1 = pool1
                x2 = pool2

        return out1, out2






