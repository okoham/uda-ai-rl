import sys
sys.path.append("../python")
from unityagents import UnityEnvironment
import numpy as np
from model import Agent, dqn
import argparse 

# get command line arguments
parser = argparse.ArgumentParser("Train new agent and save weights.")
parser.add_argument('--w', type=str, default="checkpoint.pth", 
                    help="Where to save the weights.")
args = parser.parse_args()

# Settings
DEVICE = "cpu"    
STATE_SIZE = 37   # state size, fixed
ACTION_SIZE = 4   # action size, fixed
DELAY = 0.03      # delay between actions, generates a reasonable speed for a screencast.
UNITY_ENV_PATH = "Banana_Linux_NoVis/Banana.x86_64"

# Initialise environment
env = UnityEnvironment(file_name=UNITY_ENV_PATH)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# train
agent = Agent(state_size=brain.vector_observation_space_size,
              action_size=brain.vector_action_space_size,
              seed=0)
scores = dqn(env, agent)

# close environment, save weights
env.close()
torch.save(agent.qnetwork_local.state_dict(), args.w)

