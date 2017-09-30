import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

import dataset_util


class DatasetFactory(object):
    
    def __init__(self, images_path, labels_path):
        self._images_path = images_path
        self._labels_path = labels_path
        
        with open(self._images_path) as f:
            self._images = mnist.extract_images(f)
        with open(self._labels_path) as f:
            self._labels = mnist.extract_labels(f)
        
        self._images_placeholder = tf.placeholder(
            tf.uint8, shape=self._images.shape)
        self._labels_placeholder = tf.placeholder(
            tf.uint8, shape=self._labels.shape)

        self._feed_dict = {
            self._images_placeholder: self._images,
            self._labels_placeholder: self._labels
        }
            
    def make_dataset(self):
        images_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            self._images_placeholder)
        labels_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            self._labels_placeholder)
        return tf.contrib.data.Dataset.zip((images_dataset, labels_dataset))
    
    @property
    def feed_dict(self):
        return self._feed_dict
