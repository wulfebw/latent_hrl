
# hmm

## questions
- how do you model the option as a latent variable?
    + there's like a distribution over the primitive actions that is determined by the option and dependent on the state
    + but the output variables are all discrete, their order doesn't matter at all. Like 0,1,2,0,1,2 could be the same thing as 3,4,5,3,4,5. It just depends how the actions have been assigned. There no "distribution" over these is there?
        * you have a set of possible values that the actions could take on.
        * their actual value is arbitrary, it's just that it is consistenly that value.
        * how do you model that?
        * the relative values mean nothing
        * well then use multinomial
            - that establishes probabilities over a set of discrete values without inherent order
            - so this would be a multinomial (output) hidden markov model?
                + yep seems like that's a thing 
                + ok so, implement that?
    + you of course have to account for the state in addition to the action 
        * otherwise how could the learned option condition on the state to choose the action?
        * well, if your goal is the option, then sure. But what if your goal is not the option?
            - like say you only model the actions and ignore the state
            - what you'll end up with is still a set of classes and probability distributions over primitive actions for each of those classes
                + so I can see that as a proof of concept
                + but if you want to use it I think you have to condition on the state
                    * maybe, probably
- so after implementing, 
    + it works, but it
        * a. doesn't capture that much it seems like, all it says is that the two latent class situation is the most likely
        * and b. that each class corresponds to an action
        * and c. that each class is more likely (60% to 40%) to transition to the other class rather than to itself
    + so what can I do with this?
        * let's say I start executing the agent
        * then pass the actions to the hmm
        * the hmm will be able to predict the latent class behind the actions
        * and I could like color the agent the color of the latent state for example
    + the problem is that the hmm models this like way to simply
        * I want like options over extended actions, not just single step transitions
        * does a hierarchical hmm get me that?
    + how could I use this to improve the agent?
        * what simple task could I try it on?
            - train agent to explore maze
            - derive latent classes and their action distributions
            - then put agent in different maze and use the latent classes as the actions
                + how long to run these "options" for?
                    * some set amount of time
                + how to improve the options?
    * how to perform inter-option learning?
        - use q-learning for the options I guess
    * how to account for the state?
        - pass it to the option?
- ok ok here we go:
    + run some sort of latent variable analysis to derive a set of likely latent classes
    + then create that many options, assign each to a latent class, and then in the optimization of each of those options, include a term that penalizes the kl divergence between the option and the latent variable
        * ok sounds super convoluted, but might work
- why do this as latent variable problem at all?
    + because they are latent variables so it makes sense to learn them that way
    + what I would really like to do though is to learn them using a single network
        * clearly that network needs to make multiple passes over the input / actions and it needs to perform some sort of meta analysis / self analysis though so how to do that?
        
