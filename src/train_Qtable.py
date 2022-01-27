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

    screen_segments = 5
    epsiode_length = 500
    add_bottom_segment = True
    env = BetaRobotEnv(max_iterations=epsiode_length, screen_segments=5, add_bottom_segment=add_bottom_segment)
    moves = screen_segments + 1 + int(add_bottom_segment)

    # (states, actions)
    q_table = np.loadtxt("q_table_59000.txt")
    
    # Hyperparameters
    alpha = 0.02
    gamma = 0.90

    start_epsilon = 0.5
    end_epsilon = 0.05
    end_epsilon_iteration = 20000



    total_rewards = []
    rewards = []
    reward_steps = []
    steps_per_iteration = []


    done = False
    state = env.reset()
    epsilon = 0.05
    for i in range(59000, 100001):
        print(i)
        print(q_table)
        print(q_table[state,:])
        if i % 1000 == 0:
            np.savetxt(f"q_table_{i}.txt", q_table)
            np.savetxt(f"total_rewards2.txt", total_rewards)
            np.savetxt(f"total_rewards_steps2.txt", reward_steps)
            np.savetxt(f"steps_per_iteration2.txt", steps_per_iteration)
            print(q_table)
            print(total_rewards)

        # e-greedy selection of action
        if i < end_epsilon_iteration:
            epsilon = start_epsilon - ((start_epsilon - end_epsilon) * (i/end_epsilon_iteration))

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0,2) # Explore action space
        else:
            action = np.argmax(q_table[state,:]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        print(next_state)
        if reward >= 100:
            rewards.append(1)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state


        if done:
            total_rewards.append(np.sum(rewards))
            reward_steps.append(i)
            if steps_per_iteration:
                steps_per_iteration.append(i-reward_steps[-2])
            else:
                steps_per_iteration.append(i)
            rewards = []
            state = env.reset()

  


if __name__ == "__main__":
    main()
