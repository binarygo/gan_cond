import numpy as np
import tensorflow as tf


def maybe_expand_list(a_list, expected_len):
    if len(a_list) == expected_len:
        return a_list
    elif len(a_list) == 1:
        return [a_list[0]] * expected_len


def set_first_dim(tensor, dim):
    return tf.reshape(tensor, [dim] + tensor.get_shape().as_list()[1:])


def get_flatten_dim(tensor):
    return np.prod(tensor.get_shape().as_list()[1:])


def linear(x, num_outputs):
    return tf.contrib.layers.fully_connected(
        x, num_outputs, activation_fn=None, normalizer_fn=None)


def square_error(labels, predictions):
    return tf.reduce_mean(tf.reduce_sum(
        tf.square(tf.contrib.layers.flatten(predictions) -
                  tf.contrib.layers.flatten(labels)), axis=1))


def leaky_relu(x, alpha=0.01):
    pos_x = tf.nn.relu(x)
    neg_x = tf.nn.relu(-x)
    return pos_x - neg_x * alpha


def make_leaky_relu(alpha=0.01):
    return lambda x: leaky_relu(x, alpha)


class TensorflowQueues(object):

    def __init__(self, sess):
        self._sess = sess

    def __enter__(self):
        self._coord = tf.train.Coordinator()
        self._queue_threads = tf.train.start_queue_runners(
            self._sess, self._coord)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._coord.request_stop()
        self._coord.join(self._queue_threads)
