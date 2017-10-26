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
    pos_x = tf.maximum(0.0, x)
    neg_x =  tf.minimum(0.0, x)
    return pos_x - neg_x * alpha


def make_leaky_relu(alpha=0.01):
    return lambda x: leaky_relu(x, alpha)


def parametric_relu(x):
    alpha = tf.Variable(tf.constant(0.0, shape=x.get_shape()[-1:]))
    pos_x = tf.maximum(0.0, x)
    neg_x =  tf.minimum(0.0, x)
    return pos_x + neg_x * alpha    


def get_trainable_variables_in_scope(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)


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


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, perm=(0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, axis=1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, b, a*r, r
    X = tf.split(X, b, axis=1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


@tf.contrib.framework.add_arg_scope
def pixel_shuffle_2d(x, r, depth, **kwargs):
    # x must be (B, H, W, C)
    # output (B, r*H, r*W, depth)
    assert len(x.get_shape()) == 4
    x = tf.contrib.layers.conv2d(
        inputs=x, num_outputs=depth*r*r, **kwargs)
    xs = tf.split(x, depth, axis=3)
    return tf.concat([_phase_shift(y, r) for y in xs], axis=3)


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


class InferenceBase(object):
    
    def __init__(self, train_log):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            ckpt = tf.train.latest_checkpoint(train_log)
            saver = tf.train.import_meta_graph(ckpt + '.meta')
            saver.restore(self._sess, ckpt)

    def cleanup(self):
        self._sess.close()
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cleanup()
