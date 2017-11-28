import matplotlib.pyplot as plt
from model import BEGAN
import tensorflow as tf
from glob import glob
import numpy as np
import helper
import math
import os


def train(model, epoch_count, batch_size, z_dim, star_learning_rate, beta1, beta2, get_batches, data_shape,
          image_mode):
    input_real, input_z, lrate, k_t = model.model_inputs(
        *(data_shape[1:]), z_dim)

    d_loss, g_loss, d_real, d_fake = model.model_loss(
        input_real, input_z, data_shape[3], z_dim, k_t)

    d_opt, g_opt = model.model_opt(d_loss, g_loss, lrate, beta1, beta2)

    losses = []
    learning_rate = 0
    iter = 0

    epoch_drop = 3

    lam = 1e-3
    gamma = 0.5
    k_curr = 0.0

    test_z = np.random.uniform(-1, 1, size=(16, z_dim))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):

            learning_rate = star_learning_rate * \
                math.pow(0.2, math.floor((epoch_i + 1) / epoch_drop))

            for batch_images in get_batches(batch_size):
                iter += 1

                batch_images *= 2

                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                _, D_real_curr = sess.run([d_opt, d_real], feed_dict={
                                          input_z: batch_z, input_real: batch_images, lrate: learning_rate, k_t: k_curr})

                _, D_fake_curr = sess.run([g_opt, d_fake], feed_dict={
                                          input_z: batch_z, input_real: batch_images, lrate: learning_rate, k_t: k_curr})

                k_curr = k_curr + lam * (gamma * D_real_curr - D_fake_curr)

                if iter % 100 == 0:
                    measure = D_real_curr + \
                        np.abs(gamma * D_real_curr - D_fake_curr)
                    losses.append(measure)

                    print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                          'Convergence measure: {:.4}'.format(measure))

                if iter % 100 == 0:
                    # helper.show_generator_output(
                    #     sess, model.generator, 4, input_z, batch_z, data_shape[3], image_mode, iter)

                    helper.show_generator_output(
                        sess, model.generator, input_z, test_z, data_shape[3], image_mode, iter)

        print('Training steps: ', iter)

        losses = np.array(losses)
        # fig = plt.figure()
        # plt.plot(losses)
        # plt.plot(helper.smooth(losses))
        # fig.savefig('convergence_measure.png')
        # plt.close(fig)

        helper.save_plot([losses, helper.smooth(losses)],
                         'convergence_measure.png')


if __name__ == '__main__':
    batch_size = 16
    z_dim = 64  # aka embedding
    learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    epochs = 1

    data_dir = './data'

    # Download dataset
    helper.download_extract('celeba', data_dir)

    model = BEGAN()

    celeba_dataset = helper.Dataset('celeba', glob(
        os.path.join(data_dir, 'img_align_celeba/*.jpg')))

    with tf.Graph().as_default():
        train(model, epochs, batch_size, z_dim, learning_rate, beta1, beta2, celeba_dataset.get_batches,
              celeba_dataset.shape, celeba_dataset.image_mode)
