"""
problems:
1. you cannot run with the sequences set to the same length and then use a mask because then the rnn is forward proping into zeros
2. batch size issue - need to change sizes to placeholders for rnns
"""


import numpy as np
import os
import tensorflow as tf
from tensorflow.python.ops import rnn
import time

class VAE(object):

    def __init__(self, session, flags):
        self.session = session
        self.flags = flags
        self._build_model()

        self.saver = tf.train.Saver(
            max_to_keep=100, keep_checkpoint_every_n_hours=.5)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.flags.summary_dir, 'train'), 
            self.session.graph)
        self.test_writer = tf.summary.FileWriter(
            os.path.join(self.flags.summary_dir, 'val'), 
            self.session.graph)
        self.start_time = time.time()

    def fit(self, dataset):

        # optionally load
        if self.flags.load_network:
            self.load()

        # fit the model to the dataset over a number of epochs
        for epoch in range(self.flags.num_epochs):
            train_loss, val_loss = 0, 0

            # train epoch
            for x, a, m in dataset.next_batch():
                # need to switch to placeholders for the cell states etc
                # but until then
                if len(x) != self.flags.batch_size:
                    continue
                noise = np.random.randn(
                    self.flags.batch_size, self.flags.latent_dim)
                outputs_list = [self._summary_op, self._loss, self._train_op]
                summary, loss, _ = self.session.run(outputs_list,
                    feed_dict={self._states_ph: x, self._actions_ph: a,
                    self._masks_ph: m, 
                    self._dropout_ph: self.flags.dropout_keep_prob,
                    self._noise_ph: noise})
                self.train_writer.add_summary(summary, epoch)
                train_loss += loss

            # validation epoch
            for x, a, m in dataset.next_batch(validation=True):
                # need to switch to placeholders for the cell states etc
                # but until then
                if len(x) != self.flags.batch_size:
                    continue
                noise = np.random.randn(
                    self.flags.batch_size, self.flags.latent_dim)
                summary, loss = self.session.run([self._summary_op, self._loss],
                    feed_dict={self._states_ph: x, self._actions_ph: a, 
                    self._masks_ph: m, self._dropout_ph: 1.,
                    self._noise_ph: noise})
                self.test_writer.add_summary(summary, epoch)
                val_loss += loss

            # snapshot network
            self.save(epoch)

            # print out progress
            self.log(epoch, dataset, train_loss, val_loss)

    def predict(self, inputs):
        return self.session.run(
            self._probs, feed_dict={self._input_ph: inputs, 
            self._dropout_ph: 1.})

    def reconstruct(self, states, actions, masks):
        outputs_list = [self._states_pred, self._action_probs]
        noise = np.random.randn(len(states), self.flags.latent_dim)
        feed_dict = {
            self._states_ph: states, 
            self._actions_ph: actions,
            self._masks_ph: masks,
            self._dropout_ph: 1.,
            self._noise_ph: noise
        }
        states_hat, action_probs_hat = self.session.run(
            outputs_list, feed_dict=feed_dict)
        return states_hat, action_probs_hat

    def save(self, epoch):
        """
        Description:
            - Save the session and network parameters to checkpoint file.

        Args:
            - epoch: epoch of save
        """
        if not os.path.exists(self.flags.snapshot_dir):
            os.mkdir(self.flags.snapshot_dir)
        filepath = os.path.join(self.flags.snapshot_dir, 'weights')
        self.saver.save(self.session, filepath, global_step=epoch)

    def load(self):
        """
        Description:
            - Load the lastest checkpoint file if it exists.
        """
        filepath = tf.train.latest_checkpoint(self.flags.snapshot_dir)
        if filepath is not None:
            self.saver.restore(self.session, filepath)

    def log(self, epoch, dataset, train_loss, val_loss):
        """
        Description:
            - Log training information to console

        Args:
            - epoch: training epoch
            - dataset: dataset used for training
            - train_loss: total training loss of the epoch
            - val_loss: total validation loss of the epoch
        """
        train_loss /= dataset.num_train
        val_loss /= dataset.num_val
        print('epoch: {}\ttrain loss: {:.6f}\tval loss: {:.6f}\ttime: {:.4f}'.format(
            epoch, train_loss, val_loss, time.time() - self.start_time))

    def _build_model(self):
        # placeholders
        (self._states_ph, self._actions_ph, self._masks_ph, self._dropout_ph, 
            self._noise_ph) = self._build_placeholders()

        # network
        self._mu, self._sigma = self._encode(
            self._states_ph, self._actions_ph, self._dropout_ph)
        self._states_pred, self._action_scores_pred = self._decode(
            self._mu, self._sigma, self._noise_ph, self._dropout_ph)

        # loss
        self._loss, self._action_probs = self._build_loss(
            self._states_ph, self._actions_ph, self._states_pred, 
            self._action_scores_pred, self._masks_ph)

        # train operation
        self._train_op = self._build_train_op(self._loss)

        # summaries
        self._summary_op = tf.summary.merge_all()

        # intialize the model
        self.session.run(tf.global_variables_initializer())

    def _build_placeholders(self):
        """
        Description:
            - build placeholders for inputs to the tf graph.
        """
        states_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.timesteps, self.flags.state_dim),
                name="input_ph")
        actions_ph = tf.placeholder(tf.int32,
                shape=(None, self.flags.timesteps),
                name="target_ph")
        mask_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.timesteps),
                name="input_ph")
        dropout_ph = tf.placeholder(tf.float32,
                shape=(),
                name="dropout_ph")
        noise_ph = tf.placeholder(tf.float32,
                    shape=(None, self.flags.latent_dim),
                    name="noise_ph")

        # summaries
        tf.summary.scalar('dropout keep prob', dropout_ph)
        return (states_ph, actions_ph, mask_ph, dropout_ph, noise_ph)

    def _encode(self, states, actions, dropout):
        # unpack
        state_dim, action_dim, hidden_dim, latent_dim, batch_size, timesteps = (
            self.flags.state_dim, self.flags.action_dim, self.flags.hidden_dim, 
            self.flags.latent_dim, self.flags.batch_size, self.flags.timesteps)

        # concat one-hot actions to state for full input
        actions = tf.one_hot(actions, action_dim, axis=2)
        inputs = tf.concat(2, (states, actions))

        # lstm cell output
        lstm = rnn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(
            cell=lstm, dtype=tf.float32, inputs=inputs)

        # only use the final output to generate the parameters
        output = tf.squeeze(tf.slice(outputs, begin=(0,timesteps - 1,0), 
            size=(-1,-1,-1)), 1)

        # convert final output to mean and sigma
        w_mu = tf.get_variable("w_mu", (hidden_dim, latent_dim))
        b_mu = tf.get_variable("b_mu", (latent_dim, ), 
            initializer=tf.constant_initializer(0.0))
        w_log_var = tf.get_variable(
            "w_log_var", (hidden_dim, latent_dim))
        b_log_var = tf.get_variable(
            "b_log_var", (latent_dim, ), 
            initializer=tf.constant_initializer(0.0))
        mu = tf.matmul(output, w_mu) + b_mu
        log_var = tf.matmul(output, w_log_var) + b_log_var
        sigma = tf.sqrt(tf.exp(log_var))

        # summaries
        tf.summary.histogram('mu', mu)
        tf.summary.histogram('sigma', sigma)

        return mu, sigma

    def _decode(self, mu, sigma, noise, dropout):
        # unpack
        state_dim, action_dim, hidden_dim, latent_dim, batch_size, timesteps = (
            self.flags.state_dim, self.flags.action_dim, self.flags.hidden_dim, 
            self.flags.latent_dim, self.flags.batch_size, self.flags.timesteps)
        input_dim = state_dim + action_dim

        # use lstm cell
        lstm = rnn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0, 
            state_is_tuple=True)

        # scale latent state to lstm hidden state dim
        w_z = tf.get_variable('w_z', (latent_dim, hidden_dim))
        b_z = tf.get_variable('b_z', (hidden_dim,))
        
        # compute latent state, and from that the initial cell state
        z = noise * sigma + mu

        # should the cell_state be z or should the hidden state or both?
        hidden_state = tf.matmul(z, w_z) + b_z
        cell_state = tf.zeros((batch_size, hidden_dim), dtype=tf.float32)
        state = (cell_state, hidden_state)

        # iterate sequence_length time generating and collecting
        # outputs in rnn_outputs
        states, action_scores = [], []
        prev_output = tf.zeros((batch_size, input_dim))
        for t in range(timesteps):
            with tf.variable_scope("decode") as scope:
                if t != 0:
                    scope.reuse_variables()

                w_i = tf.get_variable("w_i", (input_dim, hidden_dim))
                b_i = tf.get_variable("b_i", (hidden_dim, ), 
                    initializer=tf.constant_initializer(0.0))
                w_o = tf.get_variable("w_o", (hidden_dim, input_dim))
                b_o = tf.get_variable("b_o", (input_dim, ), 
                    initializer=tf.constant_initializer(0.0))

                next_input = tf.matmul(prev_output, w_i) + b_i
                hidden, state = lstm(next_input, state)
                prev_output = tf.matmul(hidden, w_o) + b_o

            # track state and action predictions
            states.append(tf.expand_dims(tf.slice(
                prev_output, begin=(0,0), 
                size=(-1, state_dim)),1 ))
            action_scores.append(tf.expand_dims(tf.slice(
                prev_output, begin=(0, state_dim), 
                size=(-1, action_dim)), 1))

        # concat the lists into tensors
        states = tf.concat(values=states, concat_dim=1)
        action_scores = tf.concat(values=action_scores, concat_dim=1)
        return states, action_scores

    def _build_loss(self, x, a, x_hat, a_scores, mask):
        """
        Description:
            - Build a loss function to optimize using the 
                scores of the network (unnormalized) and 
                the target values

        Args:
            - scores: unnormalized scores output from the network
                shape = (batch_size, output_dim)
            - targets: the target values
                shape = (batch_size, output_dim)

        Returns:
            - symbolic loss value
        """
        # create op for probability to use in 'predict'
        action_probs = tf.nn.softmax(a_scores)

        # action loss, yields tensor of shape (batch, timesteps)
        action_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            a_scores, a)

        # state loss, reduce sum over the state dimension yielding
        # tensor of shape (batch, timesteps)
        state_loss = tf.reduce_sum((x - x_hat) ** 2, reduction_indices=(2))

        # add the two, and then apply a mask over the invalid timesteps
        total_loss = tf.reduce_sum((
            self.flags.action_lambda * action_loss + state_loss) * mask)

        # collect regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss += reg_loss

        # summaries
        tf.summary.scalar('action_loss', tf.reduce_sum(action_loss * mask))
        tf.summary.scalar('state_loss', tf.reduce_sum(state_loss * mask))
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('l2_reg_loss', reg_loss)

        return total_loss, action_probs

    def _build_train_op(self, loss):
        """
        Description:
            - Build a training operation minimizing the loss

        Args:
            - loss: symbolic loss

        Returns:
            - training operation
        """
        opt = tf.train.AdamOptimizer(self.flags.learning_rate)
        train_op = opt.minimize(loss)
        return train_op
