import numpy as np
import tensorflow as tf


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
