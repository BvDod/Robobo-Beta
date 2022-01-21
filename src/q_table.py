#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np
import math
import time
import random

import robobo
import cv2
import sys
import signal
import prey
import vrep

from betacode.BetaRobot import BetaRobot
from betacode.BetaRobotEnv import BetaRobotEnv



def terminate_program(signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)


    env = BetaRobotEnv()

    q_table = np.zeros((6,3))
    # Hyperparameters
    alpha = 0.02
    gamma = 0.90
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

 
    total_rewards = []


    epochs, penalties, reward, = 0, 0, 0
    done = False
    state = env.reset()
    rewards = []
    reward_steps = []
    for i in range(1, 100001):
        if i % 1000 == 0:
            np.savetxt(f"q_table_{i}.txt", q_table)
            np.savetxt(f"total_rewards.txt", total_rewards)
            np.savetxt(f"total_rewards_steps.txt", reward_steps)
            print(q_table)
            print(total_rewards)

        if i % 100 == 0:
            print(total_rewards)
            


        if random.uniform(0, 1) < epsilon:
            action = random.randint(0,2) # Explore action space
        else:
            action = np.argmax(q_table[state,:]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 

        rewards.append(reward)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

        if done:
            total_rewards.append(np.sum(rewards))
            reward_steps.append(i)
            rewards = []
            state = env.reset()

  


if __name__ == "__main__":
    main()
