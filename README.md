a README that describes how someone not familiar with this project should use your repository. The README should be designed for a general audience that may not be familiar with the Nanodegree program; you should describe the environment that you solved, along with how to install the requirements before running the code in your repository.

# Project: Navigation

## Introduction

This project is about training a reinforcement learning get that navigates in a square world and collects bananas.
The given environment is modeled with Unity ML Agents. 
The agent is similar to a Deep-Q Network as described in "Human level control through deep reinforcement learning".

## Problem Description

The task consists in training an agent that navigates in a 2D world populated with bananas.
The goal of the agent is to collect as many yellow bananas as possible, and to avoid blue
bananas.

The environment has two modes, one providing an image of the scene (pixels), and another 
providing ray-based perceptions of objects in front of the agent. In this project, only the 
second environment mode iis used.

The state space has 37 dimensions (floating point numbers) and contains the agent's velocity, 
along with ray-based perception of objects around the agent's forward direction.

Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana. 

The task is episodic, and in order to solve the environment, the agent must get an average 
score of +13 over 100 consecutive episodes.


## Getting Started

- what to install and download

Step 1: Clone the DRLND Repository

If you haven't already, please follow the instructions in the DRLND GitHub repository to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels. 

https://github.com/udacity/deep-reinforcement-learning#dependencies

Step 2: Download the Unity Environment

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

    Linux: click here
    Mac OSX: click here
    Windows (32-bit): click here
    Windows (64-bit): click here

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)


## Instructions

- how to run code for training
- how to run simulation

