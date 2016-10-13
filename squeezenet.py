# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:31:28 2016

@author: dom
"""

import tensorflow as tf

slim = tf.contrib.slim

inputs = 0
net = inputs
end_points = []
end_point = 'fire2'
with tf.variable_scope(end_point):
    net = slim.conv2d(net, 16, [1, 1], scope='squeeze')
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(net, 64, [1, 1], scope='1x1')
        e3x3 = slim.conv2d(net, 64, [3, 3], scope='3x3')
    net = tf.concat(3, [e1x1, e3x3])
end_points[end_point] = net

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(net, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(net, num_outputs, [3, 3], scope='3x3')
    return tf.concat(3, [e1x1, e3x3])

def fire_module(inputs, squeeze_depth, expand_depth, scope):
    with tf.variable_scope(scope):
        net = squeeze(inputs, squeeze_depth)
        net = expand(net, expand_depth)

net = slim.conv2d(net, 96, [4, 4], 'conv1')
net = slim.max_pool2d(net, [3, 3], 'maxpool1')
net = fire_module(net, 16, 64, 'fire2')
net = fire_module(net, 16, 64, 'fire3')
net = fire_module(net, 32, 128, 'fire4')
net = slim.max_pool2d(net, [3, 3], 'maxpool4')
net = fire_module(net, 32, 128, 'fire5')
net = fire_module(net, 48, 192, 'fire6')
net = fire_module(net, 48, 192, 'fire7')
net = fire_module(net, 64, 256, 'fire8')
net = slim.max_pool2d(net, [3, 3], 'maxpool8')
net = fire_module(net, 64, 256, 'fire9')
net = slim.conv2d(net, 10, [1, 1], stride=1, 'conv10')
