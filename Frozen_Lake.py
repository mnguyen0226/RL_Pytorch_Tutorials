"""
    Frozen Lake Game:
    Game stop when the agent fall in the hole or reach the goal
    Reach Goal => +1
    Hole: No point

    Understanding the result: THe agent play 10,000 episodes. At each time step within an episode, the agents receive reward of 1 if it reached the frisbee, else ,it receive reward or 0
        This mean: The total reward receive bythe agent for entire episode is either 1 or 0.
        For a first 1000 episode, 5% of the time, the agent win the game
"""
import numpy as np
import  gym
import random
import time
from IPython.display import clear_output

# Create environment: #############################################
env = gym.make("FrozenLake-v0")

# Generate a qtable keeping track of the value ####################
# Row = state_space, Col = action_space
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

# Q Learning Algorithms Implementation ############################
num_episodes = 10000 # agent play the game 1000 times
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

# Exploration vs exploitation trade off
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001 # 0.01

# store game score over time
rewards_all_episodes = []

# Q Learning Algorithms: ####################################################
# Play 10000 episode, each episode got max of 100 steps
for episode in range (num_episodes): # Set up environment for training episode
    state = env.reset() # reset the environment

    done = False
    rewards_current_episodes = 0

    for step in range(max_steps_per_episode):
        # (1) Exploration-Explotation trade off - choose action base on trade off
        exploration_rate_threshold = random.uniform(0,1) #
        if exploration_rate_threshold > exploration_rate: # Exploitation and choose action that has highest Q value in the q_table for the current state
            action = np.argmax(q_table[state,:])
        else: # Exploration and sample an action randomly
            action = env.action_space.sample()

        # (2) Take new action: return a tuple of new state, reward for action we took and info for debugging
        new_state, reward, done, info = env.step(action)

        # (3) Update Q-Table for Q(s,a) pair
        q_table[state, action] = q_table[state, action] * (1-learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        # (4) Set new state
        state = new_state

        # (5) Add new Rewards if reach the place
        rewards_current_episodes += reward

        if done == True:
            break

    # (6): Exploration rate decay:
    # When an episode is finish we need to update the exploration rate using exploration decay
    # Meaning = the exploration rate decrease at the rate proportional to the current value
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # (7): Add current episode reward to total reward list
    rewards_all_episodes.append(rewards_current_episodes)

# Calculate and print out the average reward per thousand episodes
rewards_per_thousand_episode = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("************** Average reward per thousand episodes **************\n")
for r in rewards_per_thousand_episode:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n ***** Q-table *****\n")
print(q_table)

# Visualization - Testing Process #########################################
for episode in range (3): # Set up environment for training episode
    state = env.reset() # reset the environment

    done = False

    print("*********** Episode: ", episode + 1, "************\n\n\n\n")
    time.sleep(1) # Sleep for read the output before move on

    for step in range(max_steps_per_episode):
        clear_output(wait=True) # Wait until a new display is output
        env.render() # See where the agent is in the grid
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])

        # (2) Take new action: return a tuple of new state, reward for action we took and info for debugging
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1: # If the agent reach the goal in 1 episode
                print("********* You reach the goal **********")
                time.sleep(3)
            else:
                print("********* You fall to the hole **********")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state

env.close()