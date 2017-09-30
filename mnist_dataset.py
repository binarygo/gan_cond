import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

import dataset_util


IMAGE_KEY = 'image'
LABEL_KEY = 'label'


class Dataset(dataset_util.DatasetBase):
    
    def __init__(self, images_path, labels_path):
        with open(images_path) as f:
            images = mnist.extract_images(f).astype(np.float32) / 255.0
        with open(labels_path) as f:
            labels = mnist.extract_labels(f).astype(np.int32)
        
        images_placeholder = tf.placeholder(
            images.dtype, shape=images.shape)
        labels_placeholder = tf.placeholder(
            labels.dtype, shape=labels.shape)

        images_raw_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            images_placeholder)
        labels_raw_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            labels_placeholder)

        feed_dict = {
            images_placeholder: images,
            labels_placeholder: labels
        }

        raw = tf.contrib.data.Dataset.zip(
            {
                IMAGE_KEY: images_raw_dataset,
                LABEL_KEY: labels_raw_dataset
            })
        super(Dataset, self).__init__(raw, feed_dict)

    def vocab_size_dict(self):
        return { LABEL_KEY: 10 }
