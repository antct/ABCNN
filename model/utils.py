#!/usr/bin/env python
# coding: utf-8
# Author: Bryce
# Date  : 2018/11/03

import codecs
import numpy as np
import tensorflow as tf
import math
import logging


# define logger
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

def load_data(fname):
	logger.info('begin to read from file {}'.format(fname))
	data = []
	with codecs.getreader('utf-8')(tf.gfile.GFile(fname, 'rb')) as f:
		for line in f:
			data.append(line.strip('\n'))
	return data

def load_embedding_matrix(embedding_file):
	logger.info("begin to read embedding matrix {}".format(embedding_file))
	embedding_matrix = dict()
	embedding_size = None
	count = 0 
	with codecs.getreader('utf-8')(tf.gfile.GFile(embedding_file, 'rb')) as f:
		for line in f:
			tokens = line.replace('\n', '').strip().split(' ')
			# here, using utf-8 encode for py2, but not for py3
			word = tokens[0]
			vector = [eval(i) for i in tokens[1:]]
			if not embedding_size:
				embedding_size = len(vector)
			embedding_matrix[word] = vector
			count += 1
			if not count % 10000:
				logger.info("process {} items".format(count))
	logger.info("finish reading embedding matrix {}".format(embedding_file))
	# TODO: read real embedding size
	return embedding_matrix, embedding_size

    
def load_vocab(vocab_file):
	logger.info("read vocabulary file {}".format(vocab_file))
	vocab_list = []
	vocab_size = 0
	with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
		for word in f:
			vocab_list.append(word.strip())
			vocab_size += 1
	vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_file, vocab_size=vocab_size, default_value=0)
	logger.info("done")
	return vocab_table, vocab_list, vocab_size

def load_pretrained_matrix(vocab_file, embedding_file, num_trainable_token=0, dtype=tf.float32):
	logger.info("begin to read pretrained matrix, vocab: {}, embedding: {}".format(vocab_file, embedding_file))
	logger.info("begin to read vocabulary file {}".format(vocab_file))
	vocab_table, vocab_list, vocab_size = load_vocab(vocab_file)

	logger.info("using to read pretrained word embedding: {}".format(embedding_file))

	embedding_dict, embedding_size = load_embedding_matrix(embedding_file)

	logger.info("fill blank of word embedding")
	embedding_keys = list(embedding_dict.keys())

	tmp_embedding = [embedding_dict[token] for token in vocab_list]

	logger.info("transform to numpy format")
	embedding_matrix = np.array(tmp_embedding, dtype=dtype.as_numpy_dtype())
	embedding_matrix = tf.constant(embedding_matrix)
	# embedding_matrix_const = tf.slice(embedding_matrix, [num_trainable_tokens, 0], [-1, -1])
	# with tf.variable_scope("pretrained_embedding", dtype=dtype):
	#    embedding_matrix_var = tf.get_variable("embedding_matrix_var", [num_trainable_tokens, embedding_size])
	#return tf.concat([embedding_matrix_var, embedding_matrix_const], 0)
	return embedding_matrix

def load_embedding(vocab_file, embedding_file=None, vocab_size=None, embedding_size=None):
    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        if vocab_file and embedding_file:
            embedding = load_pretrained_matrix(vocab_file, embedding_file)
        else:
            embedding = tf.get_variable("W", [vocab_size, embedding_size],
                                        initializer=tf.random_uniform_initializer(-1, 1))
            # tf.random_normal_initializer(0., embed_size ** -0.5)
            # or tf.keras.initializers.he_uniform() or
    return embedding

if __name__ == '__main__':
	# test vocab
	# vocab_table, vocab_list, vocab_size = load_vocab(vocab_file='/media/external/chentao/glove.840B.300d.vocab')
	# print(vocab_size)
	
	# test embedding	
	# embedding_matrix, embedding_size = load_embedding_matrix(embedding_file='/media/external/chentao/glove.6B.50d')
	# print(embedding_matrix['father'])
	# print(embedding_size)

	vocab_file = '/media/external/chentao/glove.6B.50d.vocab'
	embedding_file = '/media/external/chentao/glove.6B.50d'
	embedding = load_embedding(vocab_file=vocab_file, embedding_file=embedding_file)
