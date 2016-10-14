from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
slim = tf.contrib.slim


@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat(3, [e1x1, e3x3])


def build_model(inputs):
    with tf.variable_scope('squeezenet', values=[inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([fire_module, slim.conv2d,
                             slim.max_pool2d, slim.avg_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 96, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='maxpool1')
            net = fire_module(net, 16, 64, scope='fire2')
            net = fire_module(net, 16, 64, scope='fire3')
            net = fire_module(net, 32, 128, scope='fire4')
            net = slim.max_pool2d(net, [2, 2], scope='maxpool4')
            net = fire_module(net, 32, 128, scope='fire5')
            net = fire_module(net, 48, 192, scope='fire6')
            net = fire_module(net, 48, 192, scope='fire7')
            net = fire_module(net, 64, 256, scope='fire8')
            net = slim.max_pool2d(net, [2, 2], scope='maxpool8')
            net = fire_module(net, 64, 256, scope='fire9')
            net = slim.conv2d(net, 10, [1, 1], scope='conv10')
            net = slim.avg_pool2d(net, [4, 4], scope='avgpool10')
            net = tf.squeeze(net, [1, 2], name='unscaled_logits')
            net = utils.collect_named_outputs(end_points_collection,
                                              sc.name + '/unscaled_logits',
                                              net)
        end_points = utils.convert_collection_to_dict(end_points_collection)
    return net, end_points
