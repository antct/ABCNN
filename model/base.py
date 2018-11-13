#!/usr/bin/env python
# coding: utf-8
# author: Bryce
# date  : 2018/8/30

import abc
import tensorflow as tf
from functools import reduce
from model.utils import load_embedding, load_vocab
from model.dataset import get_inference_iterator, get_train_iterator

class BaseModel(object):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        # if model is bcnn, scope: bcnn, if model is abcnn, scope: abcnn
        self.scope = self.__class__.__name__ 

        self.vocab_table, _, self.vocab_size = load_vocab(self.config.vocab_file)
        self.word_embedding = load_embedding(self.config.vocab_file, self.config.embedding_file)

        self._build_graph()

    @abc.abstractmethod
    def _build_logits(self):
        pass


    def _build_graph(self):

        # for train, input format: [label\tquery1\tquery2]
        self.src = tf.placeholder(tf.string, None, name='src')
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # train
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                # form dataset accoring to the input tensor
                self.data = tf.data.Dataset.from_tensor_slices(self.src)
                self.iterator = get_train_iterator(
                                    data=self.data, 
                                    vocab_table=self.vocab_table, 
                                    batch_size=self.config.batch_size,
                                    s1_max_len=self.config.s1_max_len, 
                                    s2_max_len=self.config.s2_max_len
                                )

                labels, s1_ids, s2_ids = self.iterator.get_next()
                
                self.x1 = tf.nn.embedding_lookup(self.word_embedding, s1_ids)
                self.x2 = tf.nn.embedding_lookup(self.word_embedding, s2_ids)

                self.x1 = tf.expand_dims(self.x1, axis=-1)
                self.x2 = tf.expand_dims(self.x2, axis=-1)

                self.logits = self._build_logits()
                self.scores = tf.nn.softmax(self.logits, name="score")

                self.predicts = tf.argmax(self.scores, 1, output_type=tf.int32, name="predict")
                
                self.diff = tf.subtract(self.predicts, labels)
                self.diff = tf.reshape(self.diff, [self.config.batch_size])
                self.count = tf.constant(value=0, dtype=tf.int32)
                for item in tf.unstack(self.diff):
                    self.count = tf.cond(tf.equal(item, tf.constant(value=0, dtype=tf.int32)), lambda : tf.add(self.count, 1), lambda : self.count)


                with tf.name_scope("loss"):
                    self.loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits))
                    self.loss_2 = tf.losses.get_regularization_loss()
                    self.loss = self.loss_1 + self.loss_2
    
                    self.optimizer = None
                    if self.config.optimizer == "rmsprop":
                        self.optimizer = tf.train.RMSPropOptimizer(self.config.lr)
                    elif self.config.optimizer == "adam":
                        self.optimizer = tf.train.AdamOptimizer(self.config.lr)
                    elif self.config.optimizer == "sgd":
                        self.optimizer = tf.train.MomentumOptimizer(self.config.lr, 0.9)
                    else:
                        raise ValueError("unsupported optimizer {}".format(self.config.optimizer))

                    self.global_step = tf.Variable(0, trainable=False)
                    self.update = self.optimizer.minimize(self.loss, global_step=self.global_step)

            else:
                self.data = tf.data.Dataset.from_tensor_slices(self.src)
                self.iterator = get_inference_iterator(
                                    data=self.data, 
                                    vocab_table=self.vocab_table, 
                                    batch_size=self.config.batch_size,
                                    s1_max_len=self.config.s1_max_len, 
                                    s2_max_len=self.config.s2_max_len
                                )
    
                s1_ids, s2_ids = self.iterator.get_next()

                self.x1 = tf.nn.embedding_lookup(self.word_embedding, self.s1_ids)
                self.x2 = tf.nn.embedding_lookup(self.word_embedding, self.s2_ids)

                self.x1 = tf.expand_dims(self.x1, axis=-1)
                self.x2 = tf.expand_dims(self.x2, axis=-1)

                self.logits = self._build_logits()
                self.scores = tf.nn.softmax(self.logits, name="score")

                self.predicts = tf.argmax(self.scores, 1, output_type=tf.int32, name="predict")
                self.evidences = tf.reduce_max(self.scores, reduction_indices=[1], name='evidence')

