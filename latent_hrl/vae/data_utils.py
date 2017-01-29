
import numpy as np
from sklearn.datasets import fetch_mldata

NUM_TRAIN = 60000
NUM_DEBUG = 512

def load_mnist(debug=False):
    mnist = fetch_mldata('MNIST original')
    data = {}
    data['train_x'] = mnist['data'][:NUM_TRAIN]
    data['test_x'] = mnist['data'][NUM_TRAIN:]
    data['train_y'] = mnist['target'][:NUM_TRAIN]
    data['test_y'] = mnist['target'][NUM_TRAIN:]

    if debug:
        idxs = np.random.permutation(len(data['train_x']))
        data['train_x'] = data['train_x'][idxs][:NUM_DEBUG]
        data['train_y'] = data['train_y'][idxs][:NUM_DEBUG]

    return data

def convert_run_data_to_training_format(input_filepath, output_filepath, timestep):
    # load and unpack states and actions
    # states and actions are stored as an array of lists, where the lists 
    # may be of different lengths
    infile = np.load(input_filepath)
    states = infile['states']
    actions = infile['actions']

    # extract shape info
    num_samples = len(states)
    state_dim = len(states[0][0])
    action_dim = max(max(a) for a in actions) + 1

    # put the data in an array with fixed column width
    final_num_seq = int((sum(len(s) + (timestep - len(s) % timestep) 
        for s in actions)) / 5)
    dataset_states = np.empty((final_num_seq, timestep, state_dim))
    dataset_actions = np.empty((final_num_seq, timestep))
    dataset_masks = np.empty((final_num_seq, timestep))
    next_idx = 0
    for state_seq, action_seq in zip(states, actions):
        print('next_idx: {}'.format(next_idx))
        # for now just add a zero to action_seq for the terminal state
        action_seq += [0]

        # add zeros to make it even with timesteps then reshape
        remainder = len(state_seq) % timestep
        zero_timesteps = timestep - remainder
        if remainder > 0:
            state_seq = np.vstack((state_seq, np.zeros((zero_timesteps, state_dim))))
            action_seq = np.hstack((action_seq, np.zeros(zero_timesteps)))
        else:
            state_seq = np.asarray(state_seq)
            action_seq = np.asarray(action_seq)
        mask = np.ones(len(state_seq))
        if remainder != 0:
            mask[-zero_timesteps:] = 0

        # reshape to even samples
        state_seq = state_seq.reshape(-1, timestep, state_dim)
        action_seq = action_seq.reshape(-1, timestep)
        mask = mask.reshape(-1, timestep)

        new = len(state_seq)
        dataset_states[next_idx:next_idx + new, :, :] = state_seq
        dataset_actions[next_idx:next_idx + new, :] = action_seq
        dataset_masks[next_idx:next_idx + new, :] = mask
        next_idx += new

    np.savez(output_filepath, states=dataset_states, actions=dataset_actions, 
        masks=dataset_masks)

def normalize_states(data, threshold=1e-8):
    # compute stats
    states = data['states_train']
    masks = data['masks_train']
    means = states[masks == 1].mean(axis=0)
    data['states_train'][masks == 1] -= means
    stds = data['states_train'][masks == 1].std(axis=0)
    data['states_train'][masks == 1] /= stds

    # normalize validation
    data['states_val'] -= means
    data['states_val'] /= stds

    # store means and standard deviations as well
    data['means'] = means
    data['stds'] = stds

    return data

def load_run_data(input_filepath, timestep=5, normalize=True, 
        debug_size=None, train_split=.8, shuffle=False):
    # load and unpack states and actions
    # states and actions are stored as an array of lists, where the lists 
    # may be of different lengths
    infile = np.load(input_filepath)
    states = infile['states']
    actions = infile['actions'].astype(int)
    masks = infile['masks'].astype(int)

    # extract state and action dim from dataset
    state_dim, action_dim = states.shape[-1], int(np.max(actions)) + 1

    # if debugging, use fewer samples
    if debug_size is not None:
        states = states[:debug_size]
        actions = actions[:debug_size]
        masks = masks[:debug_size]

    # if shuffle then randomly permute order
    if shuffle:
        idxs = np.random.permutation(len(states))
        states = states[idxs]
        actions = actions[idxs]
        masks = masks[idxs]

    # create data dictionary
    num_samples = len(states)
    num_train = int(num_samples * train_split)
    data = {
        'states_train': states[:num_train],
        'actions_train': actions[:num_train],
        'masks_train': masks[:num_train],
        'states_val': states[num_train:],
        'actions_val': actions[num_train:],
        'masks_val': masks[num_train:],
        }

    # normalize using train statistics
    if normalize:
        data = normalize_states(data)
    
    return data, state_dim, action_dim

if __name__ == '__main__':
    load_mnist()