#!/usr/bin/env python
# coding: utf-8


import collections
import tensorflow as tf
import numpy as np

def get_train_iterator(data, vocab_table, batch_size, s1_max_len=None, s2_max_len=None, num_buckets=1, num_parallel_calls=4):


	# dataset: tf.data.Dataset
	# here, for english corpus, no need to cut the words
	dataset = data

	dataset = dataset.shuffle(batch_size * 1000)
     
	# train format: query1\tquery2\tlabel
	# query format: word word word word
	dataset = dataset.map(lambda line: tf.decode_csv(line, record_defaults=[[''], [''], [0]], field_delim='\t'))

	dataset = dataset.map(lambda s1, s2, label: (label, s1, s2))

	# split s1, s2 into words
	dataset = dataset.map(lambda label, s1, s2: (label, tf.string_split([s1]).values, tf.string_split([s2]).values), num_parallel_calls=num_parallel_calls)
    
	# if s1, s2 too long, cut
	dataset = dataset.map(lambda label, s1, s2: (label, s1[:s1_max_len], s2), num_parallel_calls=num_parallel_calls) if s1_max_len else dataset
	dataset = dataset.map(lambda label, s1, s2: (label, s1, s2[:s2_max_len]), num_parallel_calls=num_parallel_calls) if s2_max_len else dataset

	# transform words to ids
	# format: label, s1_ids, s2_ids
	dataset = dataset.map(
		lambda label, s1, s2: (
			label, 
			tf.cast(vocab_table.lookup(s1), tf.int32), 
			tf.cast(vocab_table.lookup(s2), tf.int32), 
		), num_parallel_calls=num_parallel_calls)

	if num_buckets > 1:
		bucket_unit = s1_max_len // num_buckets
		bucket_boundaries = [(i + 1) * bucket_unit for i in range(num_buckets)]
		bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

		batch_func = tf.contrib.data.bucket_by_sequence_length(
			element_length_func=lambda label, s1, s2: (tf.size(s1) + tf.size(s2)) // 2,
			bucket_boundaries=bucket_boundaries,
			bucket_batch_sizes=bucket_batch_sizes,
			padded_shapes=(
				tf.TensorShape([]),
				tf.TensorShape([s1_max_len]),
				tf.TensorShape([s2_max_len]),
			),
			padding_values=(0, 0, 0),
			pad_to_bucket_boundary=False
		)
		batch_dataset = dataset.apply(batch_func).prefetch(2 * batch_size)
	else:
		batch_dataset = dataset.padded_batch(
			batch_size=batch_size,
			padded_shapes=(
				tf.TensorShape([]),
				tf.TensorShape([s1_max_len]),
				tf.TensorShape([s2_max_len]),
			),
			padding_values=(0, 0, 0)
		)
		batch_dataset = batch_dataset.filter(lambda label, s1, s2: (tf.equal(tf.shape(label)[0], batch_size)))
		batch_dataset = batch_dataset.prefetch(2 * batch_size)

	batch_iterator = batch_dataset.make_initializable_iterator()

	return batch_iterator



def get_inference_iterator(data, vocab_table, batch_size, s1_max_len=None, s2_max_len=None):
    
	dataset = data

	dataset = dataset.map(lambda line: tf.decode_csv(line, record_defaults=[[''], [''], [0]], field_delim='\t'))
	
	dataset = dataset.map(lambda s1, s2, label: (label, s1, s2))

	dataset = dataset.map(lambda label, s1, s2: (tf.string_split(s1).values, tf.string_split(s2).values))
    
	dataset = dataset.map(lambda s1, s2: (s1[:s1_max_len], s2)) if s1_max_len else dataset
	dataset = dataset.map(lambda s1, s2: (s1, s2[:s2_max_len])) if s2_max_len else dataset

	dataset = dataset.map(
		lambda s1, s2: (
			tf.cast(vocab_table.lookup(s1), tf.int32),
			tf.cast(vocab_table.lookup(s2), tf.int32),
		)
	)

	batch_dataset = dataset.padded_batch(
		batch_size=batch_size,
		padded_shapes=(
			tf.TensorShape([s1_max_len]),
			tf.TensorShape([s2_max_len]),
		),
		padding_values=(0, 0)
	)
	batch_dataset = batch_dataset.prefetch(batch_size)

	batch_iterator = batch_dataset.make_initializable_iterator()
	return batch_iterator



