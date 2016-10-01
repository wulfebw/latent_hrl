
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