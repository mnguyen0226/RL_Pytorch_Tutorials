"""
    Dec 28
    RL Tutorial
    1/ MDP: Structuring a RL Problem
    2/ Expected Return
    3/ QLearning
    4/ Exploration vs Exploitation: Epsilong greedy strategy
"""

"""
    MDP: 
        Agent, Environment, States, Actions, Rewards
        Environment => new state
        Agent: reward vs punishment
    Expected Return: 
        Maximize the expected discounted reward instead of the expected reward
        Discount rate: allow agent to care more about immediate reward over future reward since future reward is heavily discounted
        => Return is finite
    Policy = funciton that maps a given state to probabilities of selecting each possible action from that state
        pi(a|s) = an agent follow policy pi at time t = at time t, under policy pi, probability of taking action a in state s i pi(a|s)
    Value Function = function of state/state-action pair = estimate how good it is for an agent to be in a given state
        => Return the value of the action under policy pi    
    Optimal Policy = A policy is optimal when its value is greater than other value
"""
