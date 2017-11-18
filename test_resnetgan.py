import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import slim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import util
import dataset_util
import celeba_dataset


# image_size: 256 x 256

def _make_block(top, depth):
    x = top
    top = layers.conv2d(top, depth, stride=1)
    top = layers.conv2d(top, depth, stride=1, activation_fn=None)
    return x + top


def _make_blocks(top, depth, num_blocks):
    for _ in range(num_blocks):
        top = _make_block(top, depth)
    return top


def _make_up_sampling_block(top, r, depth):
    x = util.pixel_shuffle_2d(top, r, depth, kernel_size=(1,1), stride=1,
                              activation_fn=None, normalizer_fn=None)
    top = util.pixel_shuffle_2d(top, r, depth, stride=1)
    top = layers.conv2d(top, depth, stride=1, activation_fn=None)
    return x + top


def _make_down_sampling_block(top, r, depth):
    x = layers.conv2d(top, depth, kernel_size=(1,1), stride=r,
                      activation_fn=None, normalizer_fn=None)
    top = layers.conv2d(top, depth, stride=r)
    top = layers.conv2d(top, depth, stride=1, activation_fn=None)
    return x + top

    
def make_generator(noise, is_training):
    num_blocks = 1
    with slim.arg_scope(
            [layers.conv2d, util.pixel_shuffle_2d],
            kernel_size=(3, 3), padding='SAME',
            activation_fn=util.parametric_relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params={'is_training': is_training,
                               'updates_collections': None}):
        top = layers.flatten(noise)
        top = layers.fully_connected(
            top, 8 * 8 * 512,
            activation_fn=util.parametric_relu, normalizer_fn=None)
        top = tf.reshape(top, [-1, 8, 8, 512])
        top = _make_blocks(top, depth=top.get_shape().as_list()[-1],
                           num_blocks=num_blocks)
        
        for depth in [256, 128, 64, 32, 16]:
            top = _make_up_sampling_block(top, r=2, depth=depth)
            top = _make_blocks(top, depth=depth, num_blocks=num_blocks)
        
        return layers.conv2d(
            top, 3, kernel_size=(9,9), stride=1,
            activation_fn=tf.sigmoid, normalizer_fn=None)


def make_discriminator(data, is_training):
    num_blocks = 1
    with slim.arg_scope(
            [layers.conv2d],
            kernel_size=(3, 3), padding='SAME',
            activation_fn=util.parametric_relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params={'is_training': is_training,
                              'updates_collections': None}):
        top = data
        top = layers.conv2d(top, 16, kernel_size=(9, 9), stride=2,
                            normalizer_fn=None)
        top = _make_blocks(top, depth=16, num_blocks=num_blocks)
        
        for depth in [32, 64, 128, 256, 512]:
            top = _make_down_sampling_block(top, r=2, depth=depth)
            top = _make_blocks(top, depth=depth, num_blocks=num_blocks)
            
        top = layers.flatten(top)
        data_logit = util.linear(top, 1)
        return data_logit
    

train_dir = 'celeba_resnetgan_logs'
batch_size = 16
image_size = 256
noise_dim = 200
with tf.Graph().as_default():
    dataset = celeba_dataset.Dataset('../CelebA', image_size)
    iterator = dataset.make_iterator(batch_size=batch_size)
    next_elem = iterator.get_next()
    
    real_data = next_elem.pop(celeba_dataset.IMAGE_KEY)
    real_data.set_shape((batch_size, image_size, image_size, 3))
    noise = tf.random_normal(shape=(batch_size, noise_dim))
    
    # Build model
    with tf.variable_scope('gen') as gen_scope:
        gen_data = make_generator(noise, is_training=True)
    
    with tf.variable_scope('dis') as dis_scope:
        dis_gen_logit = make_discriminator(
            gen_data, is_training=True)
        
    with tf.variable_scope(dis_scope, reuse=True):
        dis_real_logit = make_discriminator(
            real_data, is_training=True)
    
    with tf.variable_scope(gen_scope, reuse=True):
        test_noise = tf.placeholder(dtype=tf.float32, shape=(None, noise_dim))
        test_data = make_generator(test_noise, is_training=False)
    
    # Add test variables to collections
    tf.add_to_collection('celeba/noise', test_noise)
    tf.add_to_collection('celeba/data', test_data)

    # Loss
    gen_loss = -tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(dis_gen_logit), dis_gen_logit)
    
    dis_loss = (
        tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(dis_gen_logit), dis_gen_logit) +
        tf.losses.sigmoid_cross_entropy(
            tf.ones_like(dis_real_logit), dis_real_logit)
    )
    
    # Train ops
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    gen_vars = util.get_trainable_variables_in_scope(gen_scope)
    dis_vars = util.get_trainable_variables_in_scope(dis_scope)
    
    optim = tf.train.AdamOptimizer(1.0e-6, beta1=0.5, beta2=0.99)
    gen_train_op = optim.minimize(
        loss=gen_loss, var_list=gen_vars, global_step=global_step)
    
    optim = tf.train.AdamOptimizer(1.0e-6, beta1=0.5, beta2=0.99)
    dis_train_op = optim.minimize(
        loss=dis_loss, var_list=dis_vars, global_step=global_step)
        
    tf.summary.scalar('gen_loss', gen_loss)
    tf.summary.scalar('dis_loss', dis_loss)
    summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer(), feed_dict=dataset.feed_dict)
        sess.run(iterator.initializer, feed_dict=dataset.feed_dict)
        
        summary_writer = tf.summary.FileWriter(
            logdir=train_dir, graph=sess.graph)
        
        with util.TensorflowQueues(sess):
            step = 0
            gen_step = 0
            dis_step = 0
            dis_loss_value = 0.0
            gen_loss_value = 0.0
            while step < 5000000:
                if gen_loss_value < -np.log(2.0):
                    _, dis_loss_value = sess.run([dis_train_op, dis_loss])
                    dis_step += 1
                if dis_loss_value < 2.0 * np.log(2.0):
                    _, gen_loss_value = sess.run([gen_train_op, gen_loss])
                    gen_step += 1
            
                step = sess.run(global_step)
                if step % 100 == 0:
                    print ('Global step {}: gen_loss = {}, dis_loss = {}, '
                           'gen_step = {}, dis_step = {}').format(
                        step, gen_loss_value, dis_loss_value, gen_step, dis_step)
                    
                    print 'Writing summaries'
                    summary_proto = sess.run(summary_op)
                    summary_writer.add_summary(summary_proto, global_step=step)
                if step % 5000 == 0:
                    print 'Writing checkpoint'
                    saver.save(sess, os.path.join(train_dir, 'model'), global_step=step)
                    
                    feed_dict = {
                        test_noise: np.random.normal(size=(5, noise_dim))
                    }
                    test_data_value = sess.run(test_data, feed_dict=feed_dict)
                    util.plot_rgb_images(test_data_value)
                    plt.savefig(os.path.join(train_dir, 'test_data_%d.png'%step))

                    print '='*50
                sys.stdout.flush()
