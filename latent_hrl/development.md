



# multi-sample hmm
- I need to alter the hmm so that it takes multiple samples (in order to handle terminal states)
    + see http://stats.stackexchange.com/questions/95144/training-a-hidden-markov-model-multiple-training-instances
    + the gist is that, you run the e-step on all the samples, and then run the m step a single time
        * I'm not sure if I can do the e step in a loop and then aggregate or what
        * but I don't think it should be too bad