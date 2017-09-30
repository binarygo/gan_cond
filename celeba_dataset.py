import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

import dataset_util


IMAGE_ID_KEY = 'image_id'
IMAGE_KEY = 'image'
ATTR_KEY_PREFIX = 'attr.'


class Dataset(dataset_util.DatasetBase):

    def __init__(self, celeba_dir, image_size):
        celeba_annot_dir = os.path.join(celeba_dir, 'Anno')
        celeba_image_dir = os.path.join(celeba_dir, 'Img/img_celeba')

        bbox_df = pd.read_csv(
            os.path.join(celeba_annot_dir, 'list_bbox_celeba.txt'),
            header=0, skiprows=1, delim_whitespace=True)
        bbox_df.set_index('image_id', drop=True, inplace=True)

        bbox_df = bbox_df[(bbox_df['height']>0) & (bbox_df['width']>0)]
        bbox_ratio = np.asarray(bbox_df['height'] * 1.0 / bbox_df['width'])
        median_bbox_ratio = np.median(bbox_ratio)
        bbox_df = bbox_df[(bbox_ratio > median_bbox_ratio * 0.9) &
                          (bbox_ratio < median_bbox_ratio * 1.1)]
        bbox_df = bbox_df.astype(np.int32)

        attr_df = pd.read_csv(
            os.path.join(celeba_annot_dir, 'list_attr_celeba.txt'),
            header=0, skiprows=1, delim_whitespace=True)
        attr_df.index.name = 'image_id'
        attr_df.replace(-1, 0, inplace=True)
        attr_df = attr_df.astype(np.int32)

        all_image_files = sorted(list(set([
            os.path.basename(f)
            for f in glob.glob(
                os.path.join(celeba_image_dir, '*.jpg'))
        ]) & set(bbox_df.index) & set(attr_df.index)))

        bbox_df = bbox_df.loc[all_image_files]
        attr_df = attr_df.loc[all_image_files]

        attr_dataset_dict = dict([
            (col, dataset_util.NumDataset(attr_df[col]))
            for col in attr_df.columns
        ])

        image_id_dataset = dataset_util.CatDataset(attr_df.index)
    
        y_dataset = dataset_util.NumDataset(bbox_df['y_1'])
        x_dataset = dataset_util.NumDataset(bbox_df['x_1'])
        h_dataset = dataset_util.NumDataset(bbox_df['height'])
        w_dataset = dataset_util.NumDataset(bbox_df['width'])

        image_dataset = dataset_util.crop_and_resize_image(
            dataset_util.read_image(celeba_image_dir, image_id_dataset),
            y_dataset, x_dataset, h_dataset, w_dataset, image_size)

        feed_dict = dataset_util.combine_feed_dicts([
            image_id_dataset, image_dataset
        ] + attr_dataset_dict.values());

        raw_dataset_dict = {
            IMAGE_ID_KEY: image_id_dataset.raw,
            IMAGE_KEY: image_dataset.raw
        }
        vocab_size_dict = {}
        for col, attr_dataset in attr_dataset_dict.iteritems():
            key = ATTR_KEY_PREFIX + col
            raw_dataset_dict[key] = attr_dataset.raw
            vocab_size_dict[key] = 2
        raw = tf.contrib.data.Dataset.zip(raw_dataset_dict)

        super(Dataset, self).__init__(raw, feed_dict)

        self.vocab_size_dict = vocab_size_dict
