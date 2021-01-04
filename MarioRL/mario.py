"""
    Pytorch.org
    Train a Maio Playing RL agent using DDQN
    https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""
# !pip install gym-super-mario-bros==7.3.0

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

import time, datetime
import matplotlib.pyplot as plt


"""
    RL Definition
    - Environment = where agent interact and learn
    - Action = how Agent responds to environment. Possible action = action-space
    - State = Current characteristic of environment. Possible state of environment = state-space
    - Reward = Key feed back fron envinonment to Agent.
    - Optimal Action-Value function. Action in a state that gave the maximize return
    - 
"""

"""
    1/ Set up Environment:
    - Tubes, mushroom and other componenets.
    - WHen mario makes action, environment responds with the changed (next) state, reward, and other info
"""
# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

"""
    Preprocess Environment:
    - env data is return to agent in next_state
    - each state = [3, 240, 256] size array => more info for agent (color of the pips or sky)
    - use Wrappers to preprocess env before sending it to agents
"""
############################################################################################# ENVIRONMENT
# PREPROCESS ENVIRONMENT before sending it to the agent use Wrapper
"""
    custom wrapper that iniherit from gym.Wrapper and implements the step(0 function.
"""
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action) # for each execute action, the return is obs, reward, done, info
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

"""
    Common wrapper to transform an RGB image to grayscale. We reduce the size of the state representation without lose useful info
    Now the size of each state is: [1, 240, 256]
"""
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

# Downsample each observation into square image. new size if [1,84,84]
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

#############################################################################
# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

"""
    FrameStack = wrapper that allow us t squash consecutive frames of environemnt into single observation point to feed to the leaning model
    This way we can identify if Mario was landing or jumpying based on the direction of his movement in the previous several fram
"""

###=> After applying this, the stucture of the environment is 3D array [3,84,84]

############################################################################################# AGENT
"""
    Act = according to the optimal action policy based on the current state of the env
    Remember experience.Experience = (current state, current action, reward, next state). Mario caches and later recalls his experiences to update his action policy.
    Learn = a better action policy over time
"""
class Mario:
    # For a given state, agent can choose to do exploit or explore
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        # Mario randomly explores with the chance of self.exploration_rate
        # When Mario choose exploit, he relies on MarioNet (Learn section) to provide the most optimal action
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net


    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    """
        Cache and Recall = Mario's memory process
        
    """
    class Mario(Mario):  # subclassing for continuity
        def __init__(self, state_dim, action_dim, save_dir):
            super().__init__(state_dim, action_dim, save_dir)
            self.memory = deque(maxlen=100000)
            self.batch_size = 32 # 32 exp per run

        """
            Each time Mario perform an action, he stores the experience to his memory. 
            His experience include the current state, action performed, reward from action nexstate, and whether the game is done
        """
        def cache(self, state, next_state, action, reward, done):
            """
            Store the experience to self.memory (replay buffer)

            Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done(bool))
            """
            state = state.__array__()
            next_state = next_state.__array__()

            if self.use_cuda:
                state = torch.tensor(state).cuda()
                next_state = torch.tensor(next_state).cuda()
                action = torch.tensor([action]).cuda()
                reward = torch.tensor([reward]).cuda()
                done = torch.tensor([done]).cuda()
            else:
                state = torch.tensor(state)
                next_state = torch.tensor(next_state)
                action = torch.tensor([action])
                reward = torch.tensor([reward])
                done = torch.tensor([done])

            self.memory.append((state, next_state, action, reward, done,))

        def recall(self):
            """
            Retrieve a batch of experiences from memory
            Mario randomly sample the batch of experiences from his memory  and use it to learn the game
            """
            batch = random.sample(self.memory, self.batch_size)
            state, next_state, action, reward, done = map(torch.stack,
                                                          zip(*batch))
            return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    """
        Mario uses DDQN algo. DDQN use two ConvNets Q online and Q target that independentally approximate the optimal action-value function
        In the implementation, we share feature generator "features" acorss Qonline and Qtarget but maintain the separate DC classifier for each. Theta tarrget is frozen to prevent backprop. It is preiodically synced with theta online
        
    """
    class MarioNet(nn.Module):
        """mini cnn structure
      input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
      """

        def __init__(self, input_dim, output_dim):
            super().__init__()
            c, h, w = input_dim

            if h != 84:
                raise ValueError(f"Expecting input height: 84, got: {h}")
            if w != 84:
                raise ValueError(f"Expecting input width: 84, got: {w}")

            self.online = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8,
                          stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                          stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                          stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )

            self.target = copy.deepcopy(self.online)

            # Q_target parameters are frozen.
            for p in self.target.parameters():
                p.requires_grad = False

        def forward(self, input, model):
            if model == "online":
                return self.online(input)
            elif model == "target":
                return self.target(input)

########################################################################## TD Estimate nad TD Target
"""
    Two values involve learning:
        TD Estimate = predict the optimal Q value for given state s
        TD Target = aggregation of current reward and the estimated Q in the next state s
"""

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    # no grad to disable gradient calculation since we don't need to backprop the theta target
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

############################################################################ Update model
"""
    As Mario samples inputs from his replay bufer, we compute TD t and TD e and back prothe loss down Q online to update its param theta online
    Theta target does not update thru back prop, we periodically copy theta online to theta target    
"""
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

# SAVE CHECK POINT:
class Mario(Mario):
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

################################### ENTIRE PROCESSS ############################3
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

######################################### Run Training loop for 10 episode, we need to run it 40000 episodes!
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


