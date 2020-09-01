import sys
sys.path.append("../python")
from unityagents import UnityEnvironment
import time
import os
import argparse
import torch
import numpy as np
from navigation import QNetwork

DEVICE = "cpu"    # no GPU required for demo play
STATE_SIZE = 37   # state size, fixed
ACTION_SIZE = 4   # action size, fixed
DELAY = 0.03      # delay between actions, generates a reasonable speed for a screencast.
UNITY_ENV_PATH = "Banana_Linux/Banana.x86_64"


class DQNAgent(object):
    """A very simple agent. Takes a state, gives it to the trained neural network 
    (state-value function, deterministically return an action).
    """

    def __init__(self, state_dict_path):
        """Initialise (non-training) Agent"""
        self.q = QNetwork(STATE_SIZE, ACTION_SIZE, 0)
        state_dict = torch.load(state_dict_path)
        self.q.load_state_dict(state_dict)
        self.q.eval()
    
    def act(self, state):
        """Return action based on given state and policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_values = self.q(state)
        return action_values.argmax().item()


class RandomAgent(object):
    """A random agent"""

    def act(self, state):
        """Return random action"""
        return np.random.randint(ACTION_SIZE) 



# get command line arguments
parser = argparse.ArgumentParser("Let an agent play an episode, and watch him on screen.")
parser.add_argument('--w', type=str, default="", 
                    help="Path to agent Q-Function weights. Use random agent if empty.")
args = parser.parse_args()


# Initialise agent. 
if (args.w == "") or (not os.path.exists(args.w)):
    agent = RandomAgent()
else: 
    agent = DQNAgent(args.w)


# start environment
env = UnityEnvironment(file_name=UNITY_ENV_PATH)
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=False)[brain_name]


# initialise score 
score = 0                                      


# run
state = env_info.vector_observations[0]            # get the current state
while True:
    action = agent.act(state)                      # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    time.sleep(DELAY)                               # wait a bit (for smooth visualisation)
    if done:                                       # exit loop if episode finished
        break
    
# done
env.close()
print("\nScore: {}".format(score))
