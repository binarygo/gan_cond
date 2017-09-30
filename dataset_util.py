import numpy as np
import tensorflow as tf


class DatasetBase(object):
    
    def __init__(self, raw_dataset, feed_dict={}):
        self.raw = raw_dataset
        self.feed_dict = feed_dict
        

class NumDataset(DatasetBase):
    
    def __init__(self, data):
        data = np.asarray(data)
        data_placeholder = tf.placeholder(
            dtype=data.dtype, shape=len(data))
        raw = tf.contrib.data.Dataset.from_tensor_slices(
            data_placeholder)
        feed_dict = { data_placeholder: data }
        super(NumDataset, self).__init__(raw, feed_dict)
        

class BucketizedDataset(DatasetBase):
    
    def __init__(self, data, num_buckets):
        assert num_buckets > 1
        data = np.asarray(data)
        bins = np.percentile(
            data, np.arange(1, num_buckets) * 100.0 / num_buckets)
        data = np.digitize(data, bins, right=True).astype(np.int32)
        data_placeholder = tf.placeholder(
            dtype=data.dtype, shape=len(data))
        raw = tf.contrib.data.Dataset.from_tensor_slices(
            data_placeholder)
        feed_dict = { data_placeholder: data }
        super(BucketizedDataset, self).__init__(raw, feed_dict)
        self.num_buckets = num_buckets
        self.bins = bins

        
class CatDataset(DatasetBase):
    
    def __init__(self, data):
        data = np.asarray(data)
        data_placeholder = tf.placeholder(
            dtype=data.dtype, shape=len(data))
        raw = tf.contrib.data.Dataset.from_tensor_slices(
            data_placeholder)
        feed_dict = {
            data_placeholder: data
        }
        super(CatDataset, self).__init__(raw, feed_dict)
        self.vocab_size = len(set(data))
        
        
class CombinedDataset(DatasetBase):
    
    def __init__(self, datasets):
        raw_datasets = []
        feed_dict = {}
        for dataset in datasets:
            raw_datasets.append(dataset.raw)
            feed_dict.update(dataset.feed_dict)
        raw = tf.contrib.data.Dataset.zip(tuple(raw_datasets))
        super(CombinedDataset, self).__init__(raw, feed_dict)
        self.datasets = datasets
        

def read_image(image_dir, image_file_dataset):
    def map_fn(image_file):
        image = tf.read_file(
            tf.string_join([image_dir, '/', image_file]))
        image = tf.image.decode_image(image)
        return tf.cast(image, tf.float32) / 255.0
    return DatasetBase(image_file_dataset.raw.map(map_fn),
                       image_file_dataset.feed_dict)


def crop_and_resize_image(image_dataset,
                          y_dataset, x_dataset,
                          h_dataset, w_dataset,
                          target_a):
    def map_fn(image, y, x, h, w):
        image = tf.image.crop_to_bounding_box(
            image, y, x, h, w)
        a = tf.minimum(h, w)
        image = tf.image.resize_image_with_crop_or_pad(
            image, a, a) 
        return tf.image.resize_images(
            image, size=[target_a, target_a])
    dataset = CombinedDataset([
        image_dataset, y_dataset, x_dataset,
        h_dataset, w_dataset])
    return DatasetBase(dataset.raw.map(map_fn), dataset.feed_dict)


def repeat_shuffle_batch(dataset, batch_size, buffer_size=None):
    if buffer_size is None:
        buffer_size = 10 * batch_size
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset
