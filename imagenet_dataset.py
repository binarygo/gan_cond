import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

import dataset_util


IMAGE_KEY = 'image'
IMAGE_LABEL_KEY = 'label'


def get_all_image_labels(imagenet_dir):
    return sorted([
        d.split('/')[-1]
        for d in glob.glob(os.path.join(imagenet_dir, '*'))
    ])


class Dataset(dataset_util.DatasetBase):

    def __init__(self, imagenet_dir, image_size=None,
                 selected_image_labels=None):
        if selected_image_labels is None:
            image_files = glob.glob(os.path.join(imagenet_dir, '*/*.JPEG'))
        else:
            image_files = []
            selected_image_labels = set(selected_image_labels)
            for image_label in selected_image_labels:
                image_files += glob.glob(os.path.join(
                    imagenet_dir, image_label, '*.JPEG'))
            
        image_labels = [ f.split('/')[-2] for f in image_files ]

        image_file_dataset = dataset_util.NumDataset(image_files)
        image_label_dataset = dataset_util.NumDataset(image_labels)
        
        image_file_label_raw_dataset = tf.contrib.data.Dataset.zip(
            (image_file_dataset.raw, image_label_dataset.raw)).shuffle(
                buffer_size=len(image_files))
        
        def map_fn(image_file, image_label):
            image = tf.read_file(image_file)
            image = tf.image.decode_image(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.slice(tf.tile(image, [1,1,3]), [0,0,0], [-1,-1,3])
            s = tf.shape(image)
            h, w = s[0], s[1]
            a = tf.minimum(h, w)
            image = tf.image.resize_image_with_crop_or_pad(image, a, a)
            image = tf.image.resize_images(
                image, size=[image_size, image_size])
            return {
                IMAGE_KEY: image,
                IMAGE_LABEL_KEY: image_label
            }

        raw = image_file_label_raw_dataset.map(map_fn)
        feed_dict = dataset_util.combine_feed_dicts([
            image_file_dataset, image_label_dataset])

        super(Dataset, self).__init__(raw, feed_dict)


    def make_iterator(self, batch_size, buffer_size=None):
        ds = dataset_util.repeat_shuffle_batch(
            self.raw, batch_size=batch_size, buffer_size=buffer_size)
        return ds.make_initializable_iterator()
