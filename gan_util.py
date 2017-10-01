import numpy as np
import tensorflow as tf

import util


class CatVector(object):
    
    def __init__(self, sparse, num_classes):
        assert num_classes >= 2
        self._sparse = tf.reshape(sparse, (-1,))
        self._num_classes = num_classes

    @property
    def one_hot(self):
        return tf.one_hot(self._sparse, self._num_classes)
    
    @property
    def sparse(self):
        return self._sparse
    
    @property
    def num_classes(self):
        return self._num_classes

    def get_embeddings(self, embedding_dim, name):
        return tf.get_variable(
            name, [self._num_classes, embedding_dim])

    def embedding_lookup(self, embeddings):
        return tf.nn.embedding_lookup(
            embeddings, self._sparse)

    def linear_output(self, input):
        if self._num_classes == 2:
            return util.linear(input, 1)
        return util.linear(input, self._num_classes)

    def get_cross_entropy_loss(self, logit):
        if self._num_classes == 2:
            return tf.losses.sigmoid_cross_entropy(
                tf.reshape(self._sparse, [-1, 1]), logit)
        return tf.losses.softmax_cross_entropy(
            self.one_hot, logit)

    def make_placeholder(self, collection=None):
        sparse_placeholder = tf.placeholder(
            shape=(None,), dtype=self._sparse.dtype)
        if collection is not None:
            tf.add_to_collection(collection, sparse_placeholder)
        return CatVector(sparse_placeholder, self._num_classes)

    def feed_dict(self, sparse_array):
        return {self._sparse : sparse_array}


def embed_cont_tensors(cont_tensors, embedding_dims):
    if np.isscalar(embedding_dims):
        embedding_dims = [embedding_dims] * len(cont_tensors)    
    result = []
    for i, (cont_tensor, embedding_dim) in (
            enumerate(zip(cont_tensors, embedding_dims))):
        result.append(util.linear(cont_tensor, embedding_dim))
    return result
    

def embed_cat_vectors(cat_vectors, embedding_dims, name):
    if np.isscalar(embedding_dims):
        embedding_dims = [embedding_dims] * len(cat_vectors)
    result = []
    for i, (cat_vector, embedding_dim) in (
            enumerate(zip(cat_vectors, embedding_dims))):
        embeddings = cat_vector.get_embeddings(
            embedding_dim,
            '{}_embeddings_{}'.format(name, i))
        result.append(cat_vector.embedding_lookup(embeddings))
    return result


class Signal(object):
    
    def __init__(self, cont_tensors, cat_vectors):
        self.cont_tensors = [
            tf.contrib.layers.flatten(cont_tensor)
            for cont_tensor in cont_tensors
        ]
        self.cat_vectors = cat_vectors


    def embed(self, cont_dims, cat_dims, name):
        return (embed_cont_tensors(self.cont_tensors, cont_dims) +
                embed_cat_vectors(self.cat_vectors, cat_dims, name))

    def linear_output(self, cont_inputs, cat_inputs):
        cont_means = [
            util.linear(tf.contrib.layers.flatten(cont_input),
                        util.get_flatten_dim(cont_tensor))
            for cont_tensor, cont_input in zip(self.cont_tensors, cont_inputs)
        ]
        cat_logits = [
            cat_vector.linear_output(tf.contrib.layers.flatten(cat_input))
            for cat_vector, cat_input in zip(self.cat_vectors, cat_inputs)
        ]
        return cont_means, cat_logits

    def get_losses(self, cont_means, cat_logits):
        return [
            util.square_error(cont_tensor, cont_means)
            for cont_tensor in self.cont_tensors
        ] + [
            cat_vector.get_cross_entropy_loss(logit)
            for logit, cat_vector in zip(cat_logits, self.cat_vectors)
        ]

    def make_placeholder(self, collection=None):
        cont_tensor_placeholders = []
        for cont_tensor in self.cont_tensors:
            pl = tf.placeholder(
                shape=[None] + cont_tensor.get_shape().as_list()[1:],
                dtype=cont_tensor.dtype)
            cont_tensor_placeholders.append(pl)
            if collection is not None:
                tf.add_to_collection(collection, pl)            
        return Signal(
            cont_tensor_placeholders,
            [
                cat_vector.make_placeholder(collection=collection)
                for cat_vector in self.cat_vectors
            ])

    def feed_dict(self, cont_arrays, sparse_arrays):
        result = dict([
            (pl, arr)
            for pl, arr in zip(self.cont_tensors, cont_arrays)
        ])
        for pl, arr in zip(self.cat_vectors, sparse_arrays):
            result.update(pl.feed_dict(arr))
        return result
