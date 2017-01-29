
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import os
import sys
import tensorflow as tf

path = os.path.join(os.path.dirname(__file__), os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import vae.vae
import vae.dataset
import vae.data_utils

FLAGS = tf.app.flags.FLAGS

# training constants
tf.app.flags.DEFINE_integer('batch_size', 
                            32,
                            """Number of samples in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 
                            1000,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_string('snapshot_dir', 
                           '../data/snapshots/test/',
                           """Path to directory where to save weights.""")
tf.app.flags.DEFINE_string('summary_dir', 
                           '../data/summaries/test',
                           """Path to directory where to save summaries.""")
tf.app.flags.DEFINE_bool('verbose', 
                            True,
                            """Wether or not to print out progress.""")
tf.app.flags.DEFINE_integer('debug_size', 
                            None,
                            """Debug size to use.""")
tf.app.flags.DEFINE_integer('random_seed', 
                            1,
                            """Random seed value to use.""")
tf.app.flags.DEFINE_bool('load_network', 
                            False,
                            """Wether or not to load from a saved network.""")
tf.app.flags.DEFINE_float('action_lambda', 
                            1.,
                            """Relative weight of the action loss to state loss.""")

# network constants
tf.app.flags.DEFINE_integer('hidden_dim', 
                            32,
                            """Hidden units in each hidden layer.""")
tf.app.flags.DEFINE_integer('latent_dim', 
                            4,
                            """Dimension of latent space.""")
tf.app.flags.DEFINE_integer('timesteps', 
                            5,
                            """Number of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', 
                            0.0005,
                            """Initial learning rate to use.""")
tf.app.flags.DEFINE_integer('decrease_lr_threshold', 
                            .001,
                            """Percent decrease in validation loss below 
                            which the learning rate will be decayed.""")
tf.app.flags.DEFINE_float('dropout_keep_prob', 
                            1.,
                            """Probability to keep a unit in dropout.""")
tf.app.flags.DEFINE_float('l2_reg', 
                            0.0,
                            """Probability to keep a unit in dropout.""")

# dataset constants
tf.app.flags.DEFINE_string('dataset_filepath',
                            '../data/datasets/maze.npz',
                            'Filepath of dataset.')

def evaluate(model, dataset):
    # sample train batch and reconstruct
    s, a, m = next(dataset.next_batch())
    s_hat, a_hat = model.reconstruct(s, a, m)
    
    a_hat = [[np.argmax(probs) for probs in seq] for seq in a_hat]
    s = s * dataset.data['stds'] + dataset.data['means']
    s_hat = s_hat * dataset.data['stds'] + dataset.data['means']
    print('\ntrain')
    print('s: {} s_hat: {}'.format(s, s_hat))
    print('a: {} a_hat: {}'.format(a, a_hat))

    # sample val batch and reconstruct
    s, a, m = next(dataset.next_batch(validation=True))
    s_hat, a_hat = model.reconstruct(s,a,m)
    s = s * dataset.data['stds'] + dataset.data['means']
    s_hat = s_hat * dataset.data['stds'] + dataset.data['means']
    a_hat = [[np.argmax(probs) for probs in seq] for seq in a_hat]
    print('\nval')
    print('s: {} s_hat: {}'.format(s, s_hat))
    print('a: {} a_hat: {}'.format(a, a_hat))

def main(argv=None):
    data, FLAGS.state_dim, FLAGS.action_dim = vae.data_utils.load_run_data(
        FLAGS.dataset_filepath, timestep=FLAGS.timesteps, normalize=True, 
        debug_size=FLAGS.debug_size, shuffle=True)
    d = vae.dataset.Dataset(data, FLAGS)

    with tf.Session() as session:
        model = vae.vae.VAE(session, FLAGS)
        model.fit(d)
        evaluate(model, d)

if __name__ == '__main__':
    # # to convert a run dataset to an actual dataset, run this
    # vae.data_utils.convert_run_data_to_training_format(
    #     '../data/runs/maze.npz', '../data/datasets/maze.npz', timestep=5)
    tf.app.run()