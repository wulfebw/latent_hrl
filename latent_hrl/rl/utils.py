import numpy as np

def score_sample(values, temp):
    try:
        values =  np.float64(values)
        exp = np.exp(values - np.max(values))
        softmax = exp / np.sum(exp)
        a = np.log(softmax) / temp
        a = np.exp(a) / np.sum(np.exp(a))
        idx = np.argmax(np.random.multinomial(1, a, 1))
        return idx
    except Exception as e:
        print 'score_sample raised exception'
        print 'values: {}'.format(values)
        # if exception just return first idx
        return 0

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))