import random
from collections import namedtuple, deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ENV_PATH = "Banana_Linux_NoVis/Banana.x86_64"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0   

# Hyperparameters
GAMMA = 0.99            # discount factor
EPS_INITIAL = 1.0       # start value for epsilon greedy policy
EPS_MIN = 0.01          # minimum value for epsilon
EPS_DECAY = 0.995       # epsilon decay, per episode. 

BUFFER_SIZE = int(1e5)  # replay buffer size
UPDATE_EVERY = 4        # how often to update the network
TAU = 1e-3              # for soft update of target parameters

LR = 2.5e-4             # learning rate 
BATCH_SIZE = 64         # minibatch size
FC1_UNITS = 64          # width of first hidden layer 
FC2_UNITS = 64          # width of second hidden layer

SOLVED_SCORE = 13       # if average score (over 100 episodes) is higher, environment is solved.
SCORE_WINDOW_SIZE = 100 # average score over how many episodes


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, 
                                       fc1_units=FC1_UNITS, 
                                       fc2_units=FC2_UNITS).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, 
                                       fc1_units=FC1_UNITS, 
                                       fc2_units=FC2_UNITS).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state. shape (state_size)
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval() # set to eval mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # set to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) arrays!!
            gamma (float): discount factor
        """
        
        # states: float, shape (batch_size, state_size)
        # actions: (batch_size, 1)
        # rewards: (batch_size, 1)
        # next_states: (batch_size, state_size)
        # dones: (batch_size, 1)
        states, actions, rewards, next_states, dones = experiences

        ## compute and minimize the loss
        
        # Get max predicted Q values (for next states) from target model
        # --------------------------------------------------------------
        # this is the orange circled part on page 18 of nebook Aug/2020:
        # max_a \hat q (s', a, w^-)
        #
        # 1. call to qnetwork_target: return 2d float tensor (batch_size, action_size)
        #
        # 2. detach: make sure that no gradients are computed on Q_targets_next
        #
        # 3. max(1): maximum value over actions dimension, and index
        #    > torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
        #    > Returns a namedtuple (values, indices) where values is the maximum value 
        #    > of each row of the input tensor in the given dimension dim. 
        #    > And indices is the index location of each maximum value found (argmax).
        #
        # 4. [0]: get the max values only. float tensor, shape (action_size,)
        #
        # 5. unsqueeze(1): Returns a new tensor with a dimension of size one inserted 
        #    at the specified position. result: shape (action_size, 1)
        #
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        # ------------------------------------
        # this is the green __TD target__ term on page 18 of notebook Aug-2020:
        # R + gamma * max_a \hat q (s', a, w^-)
        #
        # dones: 1 if done, 0 otherwise. If done, then q_target = reward
        #
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # --------------------------------------
        # blue term on p. 18 of notebook Aug-2020: old value \hat q (S, A, w)
        # it's the prediction of the (local) network.
        # This is the supervised learning thing: regression problem, 
        # optimise weights of local network so that error to target is minimised.
        # 
        # 1. qnetwork_local(states) -> return q-values for all actions, 2d float tensor, 
        #    shape (batch_size, action_size)
        #
        # 2. gather: for each row (sample), get the ith value, where i is the action id
        #    return float tensor, shape (batch_size, 1) -> TBC!
        #    > torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        #    > Gathers values along an axis specified by dim.
        #    > If input is an n-dimensional tensor with size (x0,x1...,xi−1,xi,xi+1,...,xn−1) and dim = i, 
        #    > then index must be an n-dimensional tensor with size (x0,x1,...,xi−1,y,xi+1,...,xn−1)
        #    > where y≥1 and out will have the same size as index.
        #
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)        
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def dqn(env, agent, n_episodes=1000, max_t=1000, 
        eps_initial=EPS_INITIAL, eps_min=EPS_MIN, eps_decay=EPS_DECAY):
    """Deep Q-Learning.
    
    Params
    ======
        env ... environment
        agent ... the agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        epsilon: a callable, returning epsilon(episode) for eps-greedy policy
    """

    # start environment
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]

    scores = []                                                   # list containing scores from each episode
    scores_window = deque(maxlen=SCORE_WINDOW_SIZE)               # last 100 scores
    solved = False

    eps = eps_initial                                             # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]         # reset environment
        state = env_info.vector_observations[0]                   # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                        # evaluate policy -> action
            env_info = env.step(action)[brain_name]               # send the action to the environment
            next_state = env_info.vector_observations[0]          # get the next state
            reward = env_info.rewards[0]                          # get the reward
            done = env_info.local_done[0]                         # see if episode has finished            
            agent.step(state, action, reward, next_state, done)   # memorize experience, learn id applicable
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)                               # save most recent score
        scores.append(score)
        eps = max(eps*eps_decay, eps_min)                         # decay epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if (i_episode % SCORE_WINDOW_SIZE == 0):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if (np.mean(scores_window) >= SOLVED_SCORE) and (not solved):
            # print for the first time only
            solved = True
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-SCORE_WINDOW_SIZE, np.mean(scores_window)))

    return scores



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Train a DQN-like agent for banana navigation.")  
    parser.add_argument("--weights", default=None,
                        help="Where to store trained network weights")
    parser.add_argument("--episodes", default=1000,
                        help="number of training episodes")  
    args = parser.parse_args()                      

    import sys
    sys.path.append("../python")
    from unityagents import UnityEnvironment

    # set up environment
    print("Setting up environment")
    env = UnityEnvironment(file_name=ENV_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]               

    # train agent
    print("Start training")
    agent = Agent(state_size=brain.vector_observation_space_size,
                 action_size=brain.vector_action_space_size,
                 seed=SEED)
    scores = dqn(env, agent, n_episodes=args.episodes)
    env.close()  

    # save weights
    if args.weights is not None:
        print("saving weights to {}".format(args.weights))
        torch.save(agent.qnetwork_local.state_dict(), args.weights)