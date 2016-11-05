import tensorflow as tf

from models.slim.deployment import model_deploy
from models.slim.preprocessing import preprocessing_factory
from models.slim.datasets import dataset_factory
from tensorflow.python.ops import control_flow_ops

import squeezenet
import arg_parsing

slim = tf.contrib.slim
args = arg_parsing.parse_args(training=True)

tf.logging.set_verbosity(tf.logging.INFO)
deploy_config = model_deploy.DeploymentConfig(num_clones=args.num_gpus)


def clone_fn(batch_queue):
    images, labels = batch_queue.dequeue()
    logits, end_points = network_fn(images)
    slim.losses.softmax_cross_entropy(logits, labels)
    predictions = tf.argmax(logits, 1)
    labels = tf.argmax(labels, 1)
    accuracy, update_op = slim.metrics.streaming_accuracy(
       predictions,
       labels,
       metrics_collections=['accuracy'],
       updates_collections=tf.GraphKeys.UPDATE_OPS)
    return end_points

with tf.Graph().as_default() as g:
    with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

    dataset = dataset_factory.get_dataset('cifar10', 'train', args.data_dir)

    network_fn = squeezenet.inference

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        'cifarnet', is_training=True)

    with tf.device(deploy_config.inputs_device()):
        with tf.name_scope('inputs'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                  dataset,
                  num_readers=args.reader_threads,
                  common_queue_capacity=20 * args.batch_size,
                  common_queue_min=10 * args.batch_size)
            [image, label] = provider.get(['image', 'label'])

            image = image_preprocessing_fn(image, 32, 32)
            images, labels = tf.train.batch(
                  [image, label],
                  batch_size=args.batch_size,
                  num_threads=args.preprocessing_threads,
                  capacity=5 * args.batch_size)
            labels = slim.one_hot_encoding(labels, 10)

            batch_queue = slim.prefetch_queue.prefetch_queue(
                  [images, labels], capacity=2 * deploy_config.num_clones)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.name_scope('synchronization'):
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                global_step,
                args.learning_rate_decay_steps,
                args.learning_rate_decay,
                staircase=True,
                name='exponential_decay_learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate)
        variables_to_train = tf.trainable_variables()
        total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op],
                                                          total_loss,
                                                          name='train_op')
    with tf.name_scope('summaries'):
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.histogram_summary('activations/' + end_point, x))
            summaries.add(tf.scalar_summary('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))
        summaries.add(tf.scalar_summary('learning_rate', learning_rate,
                                        name='learning_rate'))
        summaries.add(tf.scalar_summary('eval/total_loss', total_loss,
                                        name='total_loss_summary'))
        accuracy = tf.get_collection('accuracy', first_clone_scope)[0]
        summaries.add(tf.scalar_summary('eval/accuracy', accuracy))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))
        summary_op = tf.merge_summary(list(summaries), name='summary_op')

    slim.learning.train(
        train_tensor,
        args.output_training_dir,
        summary_op=summary_op,
        number_of_steps=args.max_steps,
        log_every_n_steps=args.print_log_steps,
        save_summaries_secs=args.save_summaries_secs,
        save_interval_secs=args.save_checkpoint_secs)
