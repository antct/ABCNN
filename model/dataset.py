#!/usr/bin/env python
# coding: utf-8


import collections
import tensorflow as tf
import numpy as np

def get_train_iterator(data, vocab_table, batch_size, s0_max_len=None, s1_max_len=None, num_buckets=1, num_parallel_calls=4):


	# dataset: tf.data.Dataset
	# here, for english corpus, no need to cut the words
	dataset = data

	dataset = dataset.shuffle(batch_size * 1000)
     
	# train format: query1\tquery2\tlabel
	# query format: word word word word
	dataset = dataset.map(lambda line: tf.decode_csv(line, record_defaults=[[''], [''], [0]], field_delim='\t'))

	dataset = dataset.map(lambda s0, s1, label: (label, s0, s1))

	# split s0, s1 into words
	dataset = dataset.map(lambda label, s0, s1: (label, tf.string_split([s0]).values, tf.string_split([s1]).values), num_parallel_calls=num_parallel_calls)
    
	# if s0, s1 too long, cut
	dataset = dataset.map(lambda label, s0, s1: (label, s0[:s0_max_len], s1), num_parallel_calls=num_parallel_calls) if s0_max_len else dataset
	dataset = dataset.map(lambda label, s0, s1: (label, s0, s1[:s1_max_len]), num_parallel_calls=num_parallel_calls) if s1_max_len else dataset

	# transform words to ids
	# format: label, s0_ids, s1_ids
	dataset = dataset.map(
		lambda label, s0, s1: (
			label, 
			tf.cast(vocab_table.lookup(s0), tf.int32), 
			tf.cast(vocab_table.lookup(s1), tf.int32), 
		), num_parallel_calls=num_parallel_calls)

	if num_buckets > 1:
		bucket_unit = s0_max_len // num_buckets
		bucket_boundaries = [(i + 1) * bucket_unit for i in range(num_buckets)]
		bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

		batch_func = tf.contrib.data.bucket_by_sequence_length(
			element_length_func=lambda label, s0, s1: (tf.size(s0) + tf.size(s1)) // 2,
			bucket_boundaries=bucket_boundaries,
			bucket_batch_sizes=bucket_batch_sizes,
			padded_shapes=(
				tf.TensorShape([]),
				tf.TensorShape([s0_max_len]),
				tf.TensorShape([s1_max_len]),
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
				tf.TensorShape([s0_max_len]),
				tf.TensorShape([s1_max_len]),
			),
			padding_values=(0, 0, 0)
		)
		batch_dataset = batch_dataset.filter(lambda label, s0, s1: (tf.equal(tf.shape(label)[0], batch_size)))
		batch_dataset = batch_dataset.prefetch(2 * batch_size)

	batch_iterator = batch_dataset.make_initializable_iterator()

	return batch_iterator



def get_inference_iterator(data, vocab_table, batch_size, s0_max_len=None, s1_max_len=None):
    
	dataset = data

	dataset = dataset.map(lambda line: tf.decode_csv(line, record_defaults=[[''], [''], [0]], field_delim='\t'))
	
	dataset = dataset.map(lambda s0, s1, label: (label, s0, s1))

	dataset = dataset.map(lambda label, s0, s1: (tf.string_split(s0).values, tf.string_split(s1).values))
    
	dataset = dataset.map(lambda s0, s1: (s0[:s0_max_len], s1)) if s0_max_len else dataset
	dataset = dataset.map(lambda s0, s1: (s0, s1[:s1_max_len])) if s1_max_len else dataset

	dataset = dataset.map(
		lambda s0, s1: (
			tf.cast(vocab_table.lookup(s0), tf.int32),
			tf.cast(vocab_table.lookup(s1), tf.int32),
		)
	)

	batch_dataset = dataset.padded_batch(
		batch_size=batch_size,
		padded_shapes=(
			tf.TensorShape([s0_max_len]),
			tf.TensorShape([s1_max_len]),
		),
		padding_values=(0, 0)
	)
	batch_dataset = batch_dataset.prefetch(batch_size)

	batch_iterator = batch_dataset.make_initializable_iterator()
	return batch_iterator




