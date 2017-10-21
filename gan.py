import tensorflow as tf

import util


class _Model(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)


    def train_ops(self, generator_optimizer, discriminator_optimizer,
                  global_step=None):
        return (
            generator_optimizer.minimize(
                loss=self.generator_loss,
                var_list=self.generator_variables),
            discriminator_optimizer.minimize(
                loss=self.discriminator_loss,
                var_list=self.discriminator_variables,
                global_step=global_step))        


# make_generator_fn
#   (noise, label, is_training) -> data
#
# make_discriminator_fn
#   (data, noise, label, is_training) -> data_logit, label_logits
#
# where noise, label must be gan_util.Signal.
def make_gan_model(make_generator_fn,
                   make_discriminator_fn,
                   fake_noise,
                   fake_label,
                   real_data,
                   real_label):
    num_params = len(tf.trainable_variables())
    with tf.variable_scope('generator') as generator_scope:
        fake_data = make_generator_fn(
            fake_noise, fake_label, is_training=True)
    generator_variables = tf.trainable_variables()[num_params:]
    
    num_params = len(tf.trainable_variables())
    with tf.variable_scope('discriminator') as discriminator_scope:
        (fake_data_logit,
         fake_label_output) = make_discriminator_fn(
            fake_data, fake_noise, fake_label, is_training=True)
    discriminator_variables = tf.trainable_variables()[num_params:]
    
    with tf.variable_scope(discriminator_scope, reuse=True):
        (real_data_logit,
         real_label_output) = make_discriminator_fn(
            real_data, fake_noise, real_label, is_training=True)
    
    # test
    with tf.variable_scope(generator_scope, reuse=True):
        test_noise = fake_noise.make_placeholder()
        test_label = fake_label.make_placeholder()
        test_data = make_generator_fn(
            test_noise, test_label, is_training=False)

    data_loss_fn = tf.losses.sigmoid_cross_entropy
    # data_loss_fn = util.square_error
    
    generator_losses = [
        data_loss_fn(tf.ones_like(fake_data_logit),
                     fake_data_logit)
    ] + fake_label.get_losses(fake_label_output)
    generator_loss = sum(generator_losses)
    
    discriminator_losses = [
        (data_loss_fn(tf.ones_like(real_data_logit),
                      real_data_logit) +
         data_loss_fn(tf.zeros_like(fake_data_logit),
                      fake_data_logit))
    ] + real_label.get_losses(real_label_output)
    discriminator_loss = sum(discriminator_losses)
    
    return _Model(
        make_generator_fn=make_generator_fn,
        generator_losses=generator_losses,
        generator_loss=generator_loss,
        generator_scope=generator_scope,
        generator_variables=generator_variables,
        #
        make_discriminator_fn=make_discriminator_fn,
        discriminator_losses=discriminator_losses,
        discriminator_loss=discriminator_loss,
        discriminator_scope=discriminator_scope,
        discriminator_variables=discriminator_variables,
        #
        test_noise=test_noise,
        test_data=test_data,
        test_label=test_label)
