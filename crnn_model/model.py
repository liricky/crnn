import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
import numpy as np

_BATCH_DECAY = 0.999


def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer
    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay
    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
            avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
            var = tf.reshape(var, [var.shape.as_list()[-1]])
            # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg = tf.assign(moving_avg, moving_avg * decay + avg * (1 - decay))
            # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


class CRNNCTCNetwork(object):
    def __init__(self, phase, hidden_num, layers_num, num_classes):
        self.__phase = phase.lower()
        self.__hidden_num = hidden_num
        self.__layers_num = layers_num
        self.__num_classes = num_classes
        return

    def __feature_sequence_extraction(self, input_tensor):
        is_training = True if self.__phase == 'train' else False
        # is_training = True
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            net = slim.repeat(input_tensor, 2, slim.conv2d, 64, kernel_size=3, stride=1,
                              scope='conv1')  # input_tensor  shape(32,64,?,3)  to_shape(32,1,?,x)
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool3')
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv4')
            # net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn4')
            bn_layer(x=net, scope='bn4', is_training=is_training, decay=_BATCH_DECAY)
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv5')
            # net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn5')
            bn_layer(x=net, scope='bn5', is_training=is_training, decay=_BATCH_DECAY)
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
            net = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv6')

            # net = slim.repeat(input_tensor, 2, slim.conv2d, 64, kernel_size=4, stride=1,
            #                   scope='conv1')  # input_tensor  shape(32,64,?,3)  to_shape(32,1,?,x)
            # net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
            # net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=4, stride=1, scope='conv2')
            # net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
            # net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=4, stride=1, scope='conv3')
            # net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool3')
            # net = slim.conv2d(net, 512, kernel_size=4, stride=1, scope='conv4')
            # net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn4')
            # net = slim.conv2d(net, 512, kernel_size=4, stride=1, scope='conv5')
            # net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn5')
            # net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
            # net = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv6')
        return net

    def __map_to_sequence(self, input_tensor):
        shape = input_tensor.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(input_tensor, axis=1)

    def __sequence_label(self, input_tensor, input_sequence_length):
        with tf.variable_scope('LSTM_Layers'):
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num] * self.__layers_num]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num] * self.__layers_num]
            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, input_tensor, sequence_length=input_sequence_length, dtype=tf.float32)

            [batch_size, _, hidden_num] = input_tensor.get_shape().as_list()
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_num])

            # Doing the affine projection
            w = tf.Variable(tf.truncated_normal([hidden_num, self.__num_classes], stddev=0.01), name="w")
            logits = tf.matmul(rnn_reshaped, w)

            logits = tf.reshape(logits, [batch_size, -1, self.__num_classes])
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        return rnn_out, raw_pred

    def build_network(self, images, sequence_length=None):
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(images)
        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(input_tensor=cnn_out)
        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(input_tensor=sequence, input_sequence_length=sequence_length)
        return net_out
