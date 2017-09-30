import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

import dataset_util


def read_attr_df(attr_txt_path):
    attr_df = pd.read_csv(attr_txt_path, sep='\t', header=1)
    col_names = attr_df.columns
    attr_df.drop(col_names[-1], axis=1, inplace=True)
    attr_df.columns = col_names[1:]
    return attr_df


def _get_image_path(image_dir, person, imagenum):
    file_name = (
        person.replace(' ', '_') + '_' +
        '{:04d}.jpg_0.png'.format(imagenum))
    return os.path.join(image_dir, file_name)


class DatasetFactory(object):
    
    def __init__(self, image_dir, attr_df):
        all_image_path_set = set(glob.glob(os.path.join(image_dir, '*.png')))
        image_paths = attr_df.apply(
            lambda r: _get_image_path(
                image_dir, r['person'], r['imagenum']), axis=1)
        attr_mask = image_paths.apply(
            lambda path: path in all_image_path_set)
        
        self._image_dir = image_dir
        self._attr_df = attr_df[attr_mask]

        self._image_paths = image_paths[attr_mask]
        self._image_paths_placeholder = tf.placeholder(
            tf.string, shape=(len(self._image_paths),))
        self._image_path_to_attr_index_table = (
            tf.contrib.lookup.index_table_from_tensor(
                self._image_paths_placeholder))

        self._cat_cols = sorted(['person'])
        self._cont_cols = sorted(list(attr_df.columns[2:]))
        
        self._feature_dict = {}
        for col in self._cat_cols:
            f = dataset_util.CatFeature(attr_df[col])
            self._feature_dict[col] = f
        for col in self._cont_cols:
            f = dataset_util.ContFeature(attr_df[col])
            self._feature_dict[col] = f
        
        self._feed_dict = {
            self._image_paths_placeholder: self._image_paths
        }
        for _, f in self._feature_dict.iteritems():
            self._feed_dict.update(f.feed_dict)
    
    def _read_image(self, image_path):
        image_data = tf.read_file(image_path)
        image = tf.image.decode_image(image_data)
        attr_index = self._image_path_to_attr_index_table.lookup(image_path)
        
        image_attr_dict = {}
        for col, f in self._feature_dict.iteritems():
            image_attr_dict[col] = tf.squeeze(
                tf.slice(f.placeholder, [attr_index], [1]))

        return image_path, attr_index, image, image_attr_dict
    
    def make_dataset(self):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(
            self._image_paths_placeholder)
        dataset = dataset.map(lambda image_path: self._read_image(image_path))
        return dataset

    @property
    def feature_dict(self):
        return self._feature_dict
    
    @property
    def feed_dict(self):
        return self._feed_dict
