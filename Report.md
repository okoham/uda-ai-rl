# Report: Navigation

by Oliver Koch

## Learning Algorithm

[DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

epsilon-greedy policy
Q-Learning: 
state values are continuous
nonlinear function approximation with a neural network. after x time steps, update 
unstable
NN: is a regression NN

improvement 1: experience replay. this makes it a supervised laerning problem

improvement 2: fixed targets. dqn paper: change target after xx episodes. here: weighted average of current and previous target
soft update is used for target weights \theta_{target} <- \tau \theta_{local} + (1 - \tau) \theta_{target}
that is a linear interpolation between local and target weits

- algorithm

#### Chosen hyperparameters

| Parameter | Value | Remark |
| --- | --- | --- |
| Minibatch size |
| Replay memory size |
| agent history length |
| target network update frequency |
| tau |
| discount factor |
| action repeat |
| update frequency | 
| optimizer | Adam |
| learning rate | 0.00025 | The learning rate used by the optimizer. Same value as in DQN paper. |
| adam parameter 1 |
| adam parameter 2 |
| exploration decay | exponential
| initial exploration | 
| final exploration |
| exploration decay rate |
| replay start size |
| no-op max |
| max length of episode |




#### NN architecture

The neural network has three layers, two hidden layers and one output layer.

    ReLu(FC(37, 64)) -> ReLu(FC(64, 64)) -> FC(64, 4)

I tried various modifications to the network architecture, which did not improve the result:

- making hidden layers wider (128, 256 neurons)
- making the network deeper by adding a third hidden layer
- using SeLu instead of ReLu as a nonlinear activation function

## Plot of Rewards


After 496 episodes, an average score (over 100 episodes) of 13 was achieved. 
Training continued until 1000 episodes were completed. 
The final average score is 16.80.

![plot of scores](scores_1.png)

There are two YouTube videos, one of the [trained agent](https://youtu.be/w-poO3H8ICg), 
one of a [random agent](https://youtu.be/MQixDCK0A18).
The difference in performance is obvious.

## Ideas for future work

- concrete ideas for improving the agent's performance
- double dqn, dueling dqn, proritized experience replay
- sequence of states

- give him a sequence of actions, as in the dqn paper