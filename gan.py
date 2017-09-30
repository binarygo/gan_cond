import tensorflow as tf

import util


TEST_NOISE_KEY = 'gan_test_noise'
TEST_LABEL_KEY = 'gan_test_label'
TEST_DATA_KEY = 'gan_test_data'

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
    num_params = len(tf.trainable_variables())
    with tf.variable_scope(generator_scope, reuse=True):
        test_noise = fake_noise.make_placeholder(
            collection=TEST_NOISE_KEY)
        test_label = fake_label.make_placeholder(
            collection=TEST_LABEL_KEY)
        test_data = make_generator_fn(
            test_noise, test_label, is_training=False)
        tf.add_to_collection(TEST_DATA_KEY, test_data)
    
    # data_loss_fn = tf.losses.sigmoid_cross_entropy
    data_loss_fn = util.square_error
    
    generator_losses = [
        data_loss_fn(tf.ones_like(fake_data_logit),
                     fake_data_logit)
    ] + fake_label.get_losses(*fake_label_output)
    generator_loss = sum(generator_losses)
    
    discriminator_losses = [
        (data_loss_fn(tf.ones_like(real_data_logit),
                      real_data_logit) +
         data_loss_fn(tf.zeros_like(fake_data_logit),
                      fake_data_logit))
    ] + real_label.get_losses(*real_label_output)
    discriminator_loss = sum(discriminator_losses)
    
    class Model(object):
        
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

    model = Model()
    model.make_generator_fn = make_generator_fn
    model.generator_losses = generator_losses
    model.generator_loss = generator_loss
    model.generator_scope = generator_scope
    model.generator_variables = generator_variables

    model.make_discriminator_fn = make_discriminator_fn
    model.discriminator_losses = discriminator_losses
    model.discriminator_loss = discriminator_loss
    model.discriminator_scope = discriminator_scope
    model.discriminator_variables = discriminator_variables

    model.test_noise = test_noise
    model.test_data = test_data
    model.test_label = test_label

    return model
