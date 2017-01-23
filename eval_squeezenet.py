import tensorflow as tf
import math
from models.slim.datasets import dataset_factory
from models.slim.preprocessing import preprocessing_factory

import squeezenet
import arg_parsing

slim = tf.contrib.slim
args = arg_parsing.parse_args(training=False)

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default() as g:
    with g.device(args.eval_device):
        dataset = dataset_factory.get_dataset('cifar10', 'test', args.data_dir)

        tf_global_step = slim.get_or_create_global_step()

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * args.batch_size,
            common_queue_min=args.batch_size)

        [image, label] = provider.get(['image', 'label'])

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'cifarnet',
            is_training=False)

        image = image_preprocessing_fn(image, 32, 32)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=args.batch_size,
            num_threads=args.reader_threads,
            capacity=5 * args.batch_size)

        logits, end_points = squeezenet.inference(images, is_training=False)

        predictions = tf.argmax(logits, 1)

        accuracy, update_op = slim.metrics.streaming_accuracy(predictions,
                                                              labels)
        tf.scalar_summary('eval/accuracy', accuracy)
        summary_op = tf.merge_all_summaries()

        num_batches = math.ceil(dataset.num_samples / float(args.batch_size))

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        slim.evaluation.evaluation_loop(
            master='',
            checkpoint_dir=args.checkpoint_dir,
            logdir=args.output_eval_dir,
            num_evals=num_batches,
            eval_op=update_op,
            eval_interval_secs=args.eval_interval_secs,
            session_config=sess_config,
            variables_to_restore=slim.get_variables_to_restore())
