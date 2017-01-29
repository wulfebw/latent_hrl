"""
Class for iterating through a dataset.
"""

import numpy as np

class Dataset(object):

    def __init__(self, data, flags):
        self.flags = flags
        self.data = data
        
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        keys = ['states_train', 'actions_train', 'masks_train', 'states_val', 
            'actions_val', 'masks_val']
        for k in keys:
            if k not in data:
                raise ValueError('data must contain key: {}'.format(k))

        # set data
        self._data = data

        # compute batch information
        for split in ['train', 'val']:
            num_samples = len(data['states_{}'.format(split)])
            num_batches = int(num_samples / self.flags.batch_size)

            # if num_samples not divisible by batch_size, then 
            # simply add an additional batch, which will be addressed
            # in next_batch using python indexing past the end of a container
            if num_samples % self.flags.batch_size != 0:
                num_batches += 1

            if split == 'train':
                self.num_train_batches = num_batches
                self.num_train = num_samples
            else:
                self.num_val_batches = num_batches
                self.num_val = num_samples

    def next_batch(self, validation=False):
        # retrieve training or validation set
        suffix = 'val' if validation else 'train'
        states = self.data['states_{}'.format(suffix)]
        actions = self.data['actions_{}'.format(suffix)]
        masks = self.data['masks_{}'.format(suffix)]
        num_batches = self.num_val_batches if validation else self.num_train_batches

        # suffle data for this epoch
        idxs = np.random.permutation(len(states))
        states = states[idxs]
        actions = actions[idxs]
        masks = masks[idxs]

        # yield data in batches
        for bidx in range(num_batches):
            # compute start and end indices
            start = bidx * self.flags.batch_size
            end = (bidx + 1) * self.flags.batch_size

            # retrieve the data
            b_states = states[start:end]
            b_actions = actions[start:end]
            b_masks = masks[start:end]

            yield b_states, b_actions, b_masks
        