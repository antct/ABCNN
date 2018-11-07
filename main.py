import os
import tensorflow as tf
from model import *
from model.utils import logger, load_data


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

tf.app.flags.DEFINE_string("mode", 'train', 'run mode, train, test, eval, export')

tf.app.flags.DEFINE_string("train_file", '', "train file")
tf.app.flags.DEFINE_string("test_file", '', "test file")
tf.app.flags.DEFINE_string('dev_file', '', 'dev file')
tf.app.flags.DEFINE_string("vocab_file", '', "vocab file")
tf.app.flags.DEFINE_string("embedding_file", '', "embedding file")

tf.app.flags.DEFINE_integer("s0_max_len", 40, "valid sentence length, cut or pad")
tf.app.flags.DEFINE_integer("s1_max_len", 40, "valid sentence length, cut or pad")
tf.app.flags.DEFINE_integer("num_buckets", 1, "buckets of sequence length")
tf.app.flags.DEFINE_integer("shuffle_buffer_size", 10000, "Shuffle buffer size")

tf.app.flags.DEFINE_string("model_name", "abcnn", 'model name')
tf.app.flags.DEFINE_integer("model_type", 3, "model type, 1 for APCNN-1, 2 APCNN-2, 3 for APCNN-3")
tf.app.flags.DEFINE_string("model_dir", "ckpts/", "model path")

tf.app.flags.DEFINE_float("dropout", 0.8, "dropout keep prob [0.8]")
tf.app.flags.DEFINE_integer("num_layers", 2, "num of hidden layers [2]")
tf.app.flags.DEFINE_float("l2_reg", 0.004, "l2 regularization weight [0.004]")
tf.app.flags.DEFINE_string("pooling_method", "max", "pooling method")
tf.app.flags.DEFINE_string("similar_method", "cosine", "similar method")
tf.app.flags.DEFINE_integer("num_filters", 50, "num of conv filters [30]")
tf.app.flags.DEFINE_string("filter_sizes", '2,3,4', "filter sizes [2,3,4]")

tf.app.flags.DEFINE_integer("batch_size", 32, "train batch size [64]")
tf.app.flags.DEFINE_integer("max_epoch", 50, "max epoch")

tf.app.flags.DEFINE_float("lr", 0.002, "init learning rate [adam: 0.002, sgd: 1.1]")
tf.app.flags.DEFINE_integer("lr_decay_epoch", 3, "0 for not decay")
tf.app.flags.DEFINE_float("lr_decay_rate", 0.5, "learning rate decay rate [0.5]")

tf.app.flags.DEFINE_string("optimizer", "adam", "optimizer, `adam` | `rmsprop` | `sgd` [adam]")

tf.app.flags.DEFINE_boolean("use_grad_clip", True, "whether to clip grads [False]")
tf.app.flags.DEFINE_integer("grad_clip_norm", 5, "max grad norm if use grad clip [5]")
tf.app.flags.DEFINE_integer("random_seed", 123, "random seed [123]")


FLAGS = tf.app.flags.FLAGS

def export():
	with tf.Graph().as_default():
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			with sess.graph.as_default():
				init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
				sess.run(init_ops)
				ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
				logger.info("loading saved model: {}".format(ckpt_path))
				saver = tf.train.Saver()
				saver.restore(sess, ckpt_path)

			export_path = FLAGS.model_dir + 'export/version/'
			logger.info('export trained model to {}'.format(export_path))
			builder = tf.saved_model.builder.SavedModelBuilder(export_path)

			src = tf.saved_model.utils.build_tensor_info(model.src)
			classification_outputs_classes = tf.saved_model.utils.build_tensor_info(model.predicts)
			classification_outputs_scores = tf.saved_model.utils.build_tensor_info(model.evidences)
			classification_signature = ( 
				tf.saved_model.signature_def_utils.build_signature_def(
					inputs = {"src": src},
					outputs = {"predict_classification": classification_outputs_classes, "predict_scores": classification_outputs_scores},
					method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
			)
			
			legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            
			builder.add_meta_graph_and_variables(
				sess, [tf.saved_model.tag_constants.SERVING],
				signature_def_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature},
				legacy_init_op=legacy_init_op,
			)

			builder.save()
			logger.info('done')


def predict():
	data = load_data(FLAGS.test_file)
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		with sess.graph.as_default():
			init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
			sess.run(init_ops)
			ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
			logger.info("loading saved model: {}".format(ckpt_path))
			saver = tf.train.Saver()
			saver.restore(sess, ckpt_path)
        
		logger.info('start predicting')
		sess.run(model.iterator.initializer, feed_dict={model.src: data})

		while True:
			try: 
				predicts, evidences = sess.run([model.predicts, model.evidences])
			except Exception:
				break


def eval(sess):
	data = load_data(FLAGS.dev_file)
	sess.run(model.iterator.initializer, feed_dict={model.src: data})

	count, total = 0, 0
	while True:
		try: 
			count += sess.run(model.count)
			total += FLAGS.batch_size
		except Exception:
			break
	logger.info('acc {}'.format(count/total))
	

def train():
	data = load_data(FLAGS.train_file)
	tf.set_random_seed(FLAGS.random_seed)
	with tf.Session() as sess:

		init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
		sess.run(init_ops)
		
		lr = FLAGS.lr
		for epoch in range(FLAGS.max_epoch):
			if FLAGS.lr_decay_epoch != 0 and (epoch + 1) % FLAGS.lr_decay_epoch == 0:
				lr *= FLAGS.lr_decay_rate
			logger.info('start epoch {}, lr {}'.format(epoch+1, lr))
			sess.run(model.iterator.initializer, feed_dict={model.src: data})
			step = 0
			while True:
				try:
					scores, _, loss = sess.run([model.scores, model.update, model.loss])
					step += 1
					if step % 50 == 0:
						logger.info('epoch {}\tstep {:5d}\tloss={}'.format(epoch+1, step, loss))
				except tf.errors.OutOfRangeError:
					logger.info("finish epoch {}".format(epoch+1))
					break
			eval(sess)
		if not os.path.exists(FLAGS.model_dir):
			os.mkdir(FLAGS.model_dir)
		saver = tf.train.Saver(max_to_keep=5)
		saver.save(sess, FLAGS.model_dir, global_step=model.global_step.eval())                
		logger.info('save model to {}'.format(FLAGS.model_dir))

if __name__ == '__main__':
	model_name = FLAGS.model_name.lower()
	if FLAGS.mode == 'train':
		model_mode = tf.estimator.ModeKeys.TRAIN
	else:
		model_mode = tf.estimator.ModeKeys.PREDICT	

	if model_name == 'bcnn':
		model = BCNN(FLAGS, model_mode)
	elif model_name == 'abcnn':
		model = ABCNN(FLAGS, model_mode)
	else:
		raise ValueError('invalid model type: {}'.format(model_name))

	if FLAGS.mode == 'train':
		train()
	elif FLAGS.mode == 'predict':
		predict()
	elif FLAGS.mode == 'export':
		export()
	else:
		raise ValueError('invalid model mode: {}'.format(FLAGS.mode))
