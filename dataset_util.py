import numpy as np
import tensorflow as tf


class CatFeature(object):

    def __init__(self, int_or_string_array, mapping=None):
        if mapping is None:
            self.mapping = sorted(list(set(list(int_or_string_array))))
        else:
            self.mapping = mapping
        m = dict(zip(self.mapping, range(len(self.mapping))))
        self.data = np.asarray([m[x] for x in int_or_string_array])
        self.placeholder = tf.placeholder(tf.int64, shape=(len(self.data),))
        self.feed_dict = {
            self.placeholder: self.data
        }

        
class BucketizedFeature(object):
    
    def __init__(self, num_array, num_buckets):
        bins = np.percentile(num_array, np.arange(1, num_buckets) * 100.0 / num_buckets)
        self.mapping = [(-np.inf, bins[0])] + [
            (bins[i-1], bins[i])
            for i in range(1, len(bins))
        ] + [(bins[-1], np.inf)]
        self.data = np.digitize(num_array, bins, right=True)
        self.placeholder = tf.placeholder(tf.int64, shape=(len(self.data),))
        self.feed_dict = {
            self.placeholder: self.data
        }
            

class ContFeature(object):

    def __init__(self, num_array):
        self.data = np.asarray(num_array)
        self.placeholder = tf.placeholder(tf.float32, shape=(len(self.data),))
        self.feed_dict = {
            self.placeholder: self.data
        }


def shuffle_repeat_batch(dataset, batch_size, buffer_size=None):
    if buffer_size is None:
        buffer_size = 10 * batch_size
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset
