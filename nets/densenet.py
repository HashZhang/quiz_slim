"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet_arg_scope(weight_decay=0.0004, is_training=True, data_format='NHWC'):
    """Defines the Densenet arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.
      is_training: for batch_norm
    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc


def composite_function(_input, out_features, training=True, dropout_keep_prob=0.8, kernel_size=[3, 3]):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = slim.batch_norm(_input, is_training=training)  # !!need op
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = slim.conv2d(output, out_features, kernel_size)
        # dropout(in case of training and in case it is not 1.0)
        if training:
            output = slim.dropout(output, keep_prob = dropout_keep_prob)
    return output


def bottleneck(_input, out_features, training=True, dropout_keep_prob=0.8):
    with tf.variable_scope("bottleneck"):
        inter_features = out_features * 4
        output = slim.batch_norm(_input, is_training=training)  # !!need op
        output = tf.nn.relu(output)
        output = slim.conv2d(_input, inter_features, [1, 1], padding='VALID')
        if training:
            output = slim.dropout(output, keep_prob=dropout_keep_prob)
    return output


def add_internal_layer(_input, growth_rate, training=True, bc_mode=False, dropout_keep_prob=1.0, scope="inner_layer"):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    with tf.variable_scope(scope):
        if not bc_mode:
            _output = composite_function(_input, growth_rate, training)
            if training:
                _output = slim.dropout(_output, dropout_keep_prob)

        elif bc_mode:
            bottleneck_out = bottleneck(_input, growth_rate, training)
            _output = composite_function(bottleneck_out, growth_rate, training)
            if training:
                _output = slim.dropout(_output,keep_prob = dropout_keep_prob)

        # concatenate _input with out from composite function
        # the only diffenence between resnet and densenet
        output = tf.concat(axis=3, values=(_input, _output))
        return output


def transition_layer(_input, num_filter, training=True, dropout_keep_prob=0.8, reduction=1.0):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    num_filter = int(num_filter * reduction)
    _output = composite_function(_input, num_filter, training, kernel_size=[1, 1])
    if training:
        _output = slim.dropout(_output, keep_prob = dropout_keep_prob)
    _output = slim.avg_pool2d(_output, [2, 2])
    return _output


def trainsition_layer_to_classes(_input, n_classes=10, training=True):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    _output = output = slim.batch_norm(_input, is_training=training)
    # L.scale in caffe
    _output = tf.nn.relu(_output)
    last_pool_kernel = int(_output.get_shape()[-2])
    _output = slim.avg_pool2d(_output, [last_pool_kernel, last_pool_kernel])
    logits = slim.fully_connected(_output, n_classes)
    return logits


def densenet(images, num_classes=1001, is_training=True,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5
    first_output_features = 12
    layers_per_block = 12
    #默认启用Bottleneck
    bc_mode=True
    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}
    nchannels = first_output_features
    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)):
            # first conv
            with tf.variable_scope("first_conv"):
                net = slim.conv2d(images, first_output_features, [3, 3])

            # block1
            with tf.variable_scope("block_1"):
                net = slim.repeat(net, layers_per_block, add_internal_layer,
                                  growth, is_training, bc_mode, dropout_keep_prob)
                nchannels += growth * layers_per_block
                with tf.variable_scope("transition_1"):
                    net = transition_layer(net, nchannels, is_training)

            # block2
            with tf.variable_scope("block_2"):
                net = slim.repeat(net, layers_per_block, add_internal_layer,
                                  growth, is_training, bc_mode, dropout_keep_prob)
                nchannels += growth * layers_per_block
                with tf.variable_scope("transition_2"):
                    net = transition_layer(net, nchannels, is_training)

            # block3
            with tf.variable_scope("block_3"):
                net = slim.repeat(net, layers_per_block, add_internal_layer,
                                  growth, is_training, bc_mode, dropout_keep_prob)
                nchannels += growth * layers_per_block
                assert (nchannels == net.shape[-1])
                with tf.variable_scope("trainsition_layer_to_classes"):
                    net = trainsition_layer_to_classes(net, num_classes, is_training)

            # (m,1,1,10) => (n,10)
            logits = tf.reshape(net, [-1, num_classes])

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return logits, end_points

def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
