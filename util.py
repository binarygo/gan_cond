import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def maybe_expand_list(a_list, expected_len):
    if len(a_list) == 1:
        return [a_list[0]] * expected_len
    assert len(a_list) == expected_len
    return a_list


def set_first_dim(tensor, dim):
    return tf.reshape(tensor, [dim] + tensor.get_shape().as_list()[1:])


def add_first_dim(tensor):
    return tf.expand_dims(tensor, 0)


def get_flatten_dim(tensor):
    return np.prod(tensor.get_shape().as_list()[1:])


def linear(x, num_outputs):
    return tf.contrib.layers.fully_connected(
        x, num_outputs, activation_fn=None, normalizer_fn=None)


def square_errors(labels, predictions):
    return tf.reduce_sum(
        tf.square(tf.contrib.layers.flatten(predictions) -
                  tf.contrib.layers.flatten(labels)), axis=1)


def square_error(labels, predictions):
    # mean across batches
    return tf.reduce_mean(square_errors(labels, predictions))


def l2_normalize(tensor):
    return tensor / tf.norm(tensor, axis=1, keep_dims=True)


def leaky_relu(x, alpha=0.01):
    pos_x = tf.nn.relu(x)
    neg_x = tf.nn.relu(-x)
    return pos_x - neg_x * alpha


def make_leaky_relu(alpha=0.01):
    return lambda x: leaky_relu(x, alpha)


def get_trainable_variables_in_scope(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)


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


def plot_gray_images(images, figsize=(20, 20)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(
        np.concatenate([x[:,:,0] for x in images], axis=1), 
        cmap='gray')
    plt.show()
    

def plot_rgb_images(images, figsize=(20, 20)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(
        np.concatenate([x for x in images], axis=1))
    plt.show()
