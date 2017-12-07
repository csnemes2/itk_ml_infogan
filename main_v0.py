import tensorflow as tf
import sys
from tf_wrappers import *
from train import *
import numpy as np


class RecognizedDistribution():
    def __init__(self, dist_type, cat_num=1):
        self.dist_type = dist_type
        self.cat_num = cat_num

#regularized_latent_variable_distribution = RecognizedDistribution("categorical",cat_num=10)
regularized_latent_variable_distribution = RecognizedDistribution("uniform")


# @note: regularized vars are calculated with one hot vector!!!
# otherwise poor convergence
def tf_noise_generator(batch_size, noise_dim):
    global regularized_latent_variable_distribution
    cat_num = regularized_latent_variable_distribution.cat_num
    dist_type = regularized_latent_variable_distribution.dist_type

    ret = [
        tf.cast(tf.random_uniform([batch_size, noise_dim - cat_num], minval=-1., maxval=1.),
                tf.float32)]

    if dist_type == "uniform":
        ret.append(tf.cast(tf.random_uniform([batch_size, 1], minval=-1., maxval=1.),
                           tf.float32))
    elif dist_type == "categorical":
        x = tf.contrib.distributions.Categorical(np.ones(cat_num), dtype=tf.int32).sample(batch_size)
        x = tf.one_hot(x,cat_num)
        ret.append(x)

    return tf.concat(ret, 1)


def cpu_noise_generator(batch_size, noise_dim):
    global regularized_latent_variable_distribution
    cat_num = regularized_latent_variable_distribution.cat_num
    dist_type = regularized_latent_variable_distribution.dist_type

    ret = [
        np.random.uniform(-1, 1, size=[batch_size, noise_dim-cat_num])]

    if dist_type == "uniform":
        ret.append(np.random.uniform(-1, 1, size=[batch_size, 1]))
    elif dist_type == "categorical":
        x = np.random.random_integers(cat_num, size=batch_size)- 1
        x = np.eye(cat_num)[x]
        ret.append(x)
    return np.concatenate(ret,axis=1)

def cpu_noise_generator_tricky(batch_size, noise_dim):
    global regularized_latent_variable_distribution
    cat_num = regularized_latent_variable_distribution.cat_num
    dist_type = regularized_latent_variable_distribution.dist_type

    vis_size = 10
    if dist_type == "uniform":
        repeat_noise = int(10)
    elif dist_type == "categorical":
        repeat_noise = cat_num

    noise_mtx = np.tile(
        np.random.uniform(-1, 1, size=[vis_size, noise_dim - cat_num]),
        [repeat_noise, 1])

    if dist_type == "uniform":
        x = np.arange(-1, 1, 2.0 / repeat_noise)
        latent_mtx = np.array([e for e in x for _ in range(vis_size)]).reshape([vis_size*repeat_noise,1])

    elif dist_type == "categorical":
        x = np.arange(0, repeat_noise, 1)
        latent_mtx = [e for e in x for _ in range(vis_size)]
        latent_mtx = np.eye(cat_num)[latent_mtx]

    tricky_z_batch = np.concatenate([noise_mtx, latent_mtx], axis=1)
    tricky_z_batch = np.concatenate([
        tricky_z_batch,
        np.random.uniform(-1, 1,
                          size=[batch_size - (repeat_noise * vis_size), noise_dim])
    ], axis=0)

    print (tricky_z_batch[:10,-10:])

    return tricky_z_batch

def generator(z, batch_size, reuse=False, img_size=28):
    with tf.variable_scope("g_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("fully1"):
            a0 = fully_batch_relu(z, [100, 1024], [1024])

        s1 = int(img_size / 4)
        with tf.variable_scope("fully2"):
            a1 = fully_batch_relu(a0, [1024, s1 * s1 * 128], [s1 * s1 * 128])

        a1r = tf.reshape(a1, [batch_size, s1, s1, 128])

        s2 = int(img_size / 2)
        with tf.variable_scope("deconv2"):
            a2 = deconv_batch_relu(a1r, [batch_size, s2, s2, 64], 'SAME')

        with tf.variable_scope("deconv3"):
            a3 = deconv_sigmoid(a2, [batch_size, img_size, img_size, 1], 'SAME')

    return a3


def discriminator(x_image, reuse=False):
    global regularized_latent_variable_distribution
    cat_num = regularized_latent_variable_distribution.cat_num
    dist_type = regularized_latent_variable_distribution.dist_type

    with tf.variable_scope("d_"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("conv1"):
            ret = conv_stride2_leaky(x_image, [4, 4, 1, 64], [64])

        with tf.variable_scope("conv2"):
            ret2 = conv_stride2_norm_leaky(ret, [4, 4, 64, 128], [128])

        ret3 = tf.reshape(ret2, [-1, 7 * 7 * 128])

        with tf.variable_scope("fully1"):
            ret4 = fully_batch_leaky(ret3, [7 * 7 * 128, 1024], [1024])

        with tf.variable_scope("fully2"):
            ret5 = fully(ret4, [1024, 1], [1])

        with tf.variable_scope("q1"):
            retq1 = fully(ret4, [1024, 128], [128])

        with tf.variable_scope("q2"):
            if dist_type == "uniform":
                retq2 = tf.tanh(fully(retq1, [128, 1], [1]))
            elif dist_type == "categorical":
                retq2 = tf.nn.softmax(fully(retq1, [128, cat_num], [cat_num]))

        return ret5, retq2

def q_loss_computation(noise,q_out):
    global regularized_latent_variable_distribution
    cat_num = regularized_latent_variable_distribution.cat_num
    dist_type = regularized_latent_variable_distribution.dist_type

    if dist_type == "uniform":
        loss = tf.reduce_mean(10*tf.square(noise[:,99] - q_out))

    elif dist_type == "categorical":
        noise_oh = noise[:, -cat_num:]
        q_relevant = tf.reduce_max(noise_oh * q_out,axis=1)
        loss = tf.reduce_mean(-tf.log(q_relevant))

    return loss

train(generator,
      discriminator,
      tf_noise_generator,
      cpu_noise_generator,
      cpu_noise_generator_tricky,
      q_loss_computation)