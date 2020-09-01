import sys
sys.path.append("../python")
from unityagents import UnityEnvironment
import numpy as np
from model import Agent, dqn


DEVICE = "cpu"    # no GPU required for demo play
STATE_SIZE = 37   # state size, fixed
ACTION_SIZE = 4   # action size, fixed
DELAY = 0.03      # delay between actions, generates a reasonable speed for a screencast.
UNITY_ENV_PATH = "Banana_Linux/Banana.x86_64"

env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size=brain.vector_observation_space_size,
              action_size=brain.vector_action_space_size,
              seed=0)

scores = dqn(env, agent)
env.close()