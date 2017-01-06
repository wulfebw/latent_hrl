"""
A feed-forward neural network class
"""
import tensorflow as tf

from . import initializers

class VAE(object):
    def __init__(self, session, flags):
        self.session = session
        self.flags = flags
        self._build_model()

    def fit(self, dataset):
        # build saver
        saver = tf.train.Saver()

        # build summary writers
        train_writer = tf.train.SummaryWriter(
            self.flags.summary_dir + '/train', self.session.graph)
        test_writer = tf.train.SummaryWriter(
            self.flags.summary_dir + '/val')

        # fit the model to the dataset over a number of epochs
        for epoch in range(self.flags.num_epochs):
            train_loss, val_loss = 0, 0

            # train epoch
            for x, y in dataset.next_batch():
                summary, loss, _ = self.session.run(
                    [self._summary_op, self._loss, self._train_op],
                    feed_dict={self._input_ph: x, self._target_ph: y,
                    self._dropout_ph: self.flags.dropout_keep_prob})
                train_writer.add_summary(summary, epoch)
                train_loss += loss

            # validation epoch
            for x, y in dataset.next_batch(validation=True):
                summary, loss = self.session.run([self._summary_op, self._loss],
                    feed_dict={self._input_ph: x, self._target_ph: y,
                    self._dropout_ph: 1.})
                test_writer.add_summary(summary, epoch)
                val_loss += loss

            # snapshot network
            if (epoch + 1) % self.flags.save_every == 0:
                saver.save(self.session, self.flags.snapshot_filepath)

            # print out progress if verbose
            if self.flags.verbose:
                train_loss /= dataset.num_train_batches
                val_loss /= dataset.num_val_batches
                print('epoch: {}\ttrain loss: {:.8f}\tval loss: {:.8f}'.format(
                    epoch, train_loss, val_loss))

        # save again after training
        saver.save(self.session, self.flags.snapshot_filepath)

    def predict(self, inputs):
        return self.session.run(
            self._probs, feed_dict={self._input_ph: inputs, 
            self._dropout_ph: 1.})

    def _build_model(self):
        # placeholders
        self._input_ph, self._target_ph = self._build_placeholders()

        # network
        self._encoding, self._decoding = self._build_network(self._input_ph)

        # loss
        self._loss, self._probs = self._build_loss(
            self._scores, self._target_ph)

        # train operation
        self._train_op = self._build_train_op(self._loss)

        # summaries
        self._summary_op = tf.merge_all_summaries()

        # intialize the model
        self.session.run(tf.initialize_all_variables())

    def _build_placeholders(self):
        """
        Description:
            - build placeholders for inputs to the tf graph.

        Returns:
            - input_ph: placeholder for a input batch
            - target_ph: placeholder for a target batch
        """
        input_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.input_dim),
                name="input_ph")
        target_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.output_dim),
                name="target_ph")
        # summaries
        tf.scalar_summary('dropout keep prob', dropout_ph)

        return input_ph, target_ph, dropout_ph, 

    def _build_network(self, input_ph, dropout_ph):
        """
        Description:
            - Builds a feed forward network with relu units.

        Args:
            - input_ph: placeholder for the inputs
                shape = (batch_size, input_dim)
            - dropout_ph: placeholder for dropout value

        Returns:
            - scores: the scores for the target values
        """

        # build initializers specific to relu
        weights_initializer = initializers.get_weight_initializer(
            'relu')
        bias_initializer = initializers.get_bias_initializer(
            'relu')

        # build regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(
            self.flags.l2_reg)

        # build hidden layers
        hidden = input_ph
        for lidx in range(self.flags.num_hidden_layers):
            hidden = tf.contrib.layers.fully_connected(hidden, 
                self.flags.hidden_dim, 
                activation_fn=tf.nn.relu,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=bias_initializer)
            hidden = tf.nn.dropout(hidden, dropout_ph)

        # build output layer
        scores = tf.contrib.layers.fully_connected(hidden, 
                self.flags.output_dim, 
                activation_fn=None,
                weights_regularizer=weights_regularizer)

        # summaries
        tf.histogram_summary('scores', scores)

        return scores

    def _build_loss(self, scores, targets):
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
        probs = tf.sigmoid(scores)

        # create loss separately
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(scores, targets))

        # collect regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_loss

        # summaries
        tf.histogram_summary('probs', probs)
        tf.scalar_summary('loss', loss)
        tf.scalar_summary('l2 reg loss', loss)

        return loss, probs

    def _build_train_op(self, loss):
        """
        Description:
            - Build a training operation minimizing the loss

        Args:
            - loss: symbolic loss

        Returns:
            - training operation
        """
        # exponential decaying learning rate
        global_step = tf.Variable(0, trainable=False)
        init_learning_rate = self.flags.learning_rate
        decay_every = self.flags.decay_lr_every 
        decay_ratio = self.flags.decay_lr_ratio
        learning_rate = tf.train.exponential_decay(
            init_learning_rate, global_step,
            decay_every, decay_ratio, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)   

        # clip gradients by norm
        grads_params = opt.compute_gradients(loss) 
        clipped_grads_params = [(tf.clip_by_norm(
            g, self.flags.max_norm), p) 
            for (g, p) in grads_params]
        train_op = opt.apply_gradients(
            clipped_grads_params, global_step=global_step)  

        # summaries
        tf.scalar_summary('learning rate', learning_rate)
        for (g, p) in clipped_grads_params:
            tf.histogram_summary('grads for {}'.format(p.name), g)

        return train_op
