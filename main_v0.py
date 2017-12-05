import tensorflow as tf
import sys
from tf_wrappers import *
from train import *
import numpy as np

class RecognizedDistribution():
    def __init__(self,type_string,cat_num=10):
        self.type_string = type_string
        self.cat_num = cat_num

regularized_latent_variable_distribution = RecognizedDistribution("categorical",cat_num=3)
#regularized_latent_variable_distribution = RecognizedDistribution("uniform")


def tf_noise_generator(batch_size, noise_dim):
    global regularized_latent_variable_distribution
    ret = []

    ret.append(tf.cast(tf.random_uniform([batch_size, noise_dim-1], minval=-1., maxval=1.),
                       tf.float32))

    if regularized_latent_variable_distribution.type_string == "uniform":
        ret.append(tf.cast(tf.random_uniform([batch_size, 1], minval=-1., maxval=1.),
                           tf.float32))
    else:
        cats= regularized_latent_variable_distribution.cat_num
        probs=np.ones([cats],dtype=np.float32)/cats
        x = tf.contrib.distributions.Categorical(probs,dtype=tf.float32).sample(batch_size)/cats
        x = tf.reshape(x,[batch_size,1])
        ret.append(x)

    return tf.concat(ret, 1)

def cpu_noise_generator(batch_size, noise_dim):
    global regularized_latent_variable_distribution
    ret=[]
    ret.append(np.random.uniform(-1, 1, size=[batch_size, noise_dim-1]))

    if regularized_latent_variable_distribution.type_string == "uniform":
        ret.append(np.random.uniform(-1, 1, size=[batch_size, 1]))
    else:
        cats = regularized_latent_variable_distribution.cat_num
        ret.append((np.random.random_integers(cats, size=batch_size).reshape(batch_size,1)-1)/cats)

    return np.concatenate(ret,axis=1)

def cpu_noise_generator_tricky(batch_size, noise_dim):
    global regularized_latent_variable_distribution
    cats = regularized_latent_variable_distribution.cat_num

    vis_size = 10
    if regularized_latent_variable_distribution.type_string == "uniform":
        repeat_noise = int(10)
    else:
        repeat_noise = cats

    noise_mtx = np.tile(
        np.random.uniform(-1, 1, size=[vis_size, noise_dim - 1]),
        [repeat_noise, 1])

    if regularized_latent_variable_distribution.type_string == "uniform":
        latent_mtx = np.arange(-1, 1, 2.0 / repeat_noise)[:repeat_noise]
    else:
        latent_mtx = np.arange(0, repeat_noise, 1)[:repeat_noise]/cats

    # well, not very elegant
    latent_mtx = np.tile(latent_mtx.reshape(repeat_noise, 1), (1, vis_size)).reshape(
        repeat_noise * vis_size, 1)

    tricky_z_batch = np.concatenate([latent_mtx, noise_mtx], axis=1)
    tricky_z_batch = np.concatenate([
        tricky_z_batch,
        np.random.uniform(-1, 1,
                          size=[batch_size - (repeat_noise * vis_size), noise_dim])
    ], axis=0)
    return tricky_z_batch

def generator(z, batch_size, reuse=False, img_size=28):
    """
    A Generator inspired by InfoGAN's generator: https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
    """
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
    """
    A Generator inspired by InfoGAN's generator: https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
    """
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

        return ret5

train(generator, discriminator, tf_noise_generator, cpu_noise_generator, cpu_noise_generator_tricky)