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
    epsiode_length = 150
    generalist = True
    env = BetaRobotEnv(max_iterations=epsiode_length, screen_segments=5, generalist=generalist)
    moves = (screen_segments + 1) + (not generalist * (screen_segments + 1))

    # (states, actions)
    q_table = np.ones((moves,3))
    # q_table = np.loadtxt("training_results/Task3/First_Experiment/q_table_95000.txt")

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

    steps_to_secured = []
    steps_secured_to_win = []
    times_lost_food = []


    done = False
    state = env.reset()
    epsilon = 0.05
    for i in range(1, 100001):
        if i % 1000 == 0:
            print(i)
            np.savetxt(f"q_table_{i}.txt", q_table)
            np.savetxt(f"total_rewards.txt", total_rewards)
            np.savetxt(f"total_rewards_steps.txt", reward_steps)
            np.savetxt(f"steps_per_iteration.txt", steps_per_iteration)
            np.savetxt(f"steps_to_secured.txt", steps_to_secured)
            np.savetxt(f"steps_secured_to_win.txt", steps_secured_to_win)
            np.savetxt(f"times_lost_food.txt", times_lost_food)
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

        if reward >= 100:
            rewards.append(1)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state


        if done:
            print(f"times lost food: {env.timesLostFood}")
            total_rewards.append(np.sum(rewards))
            reward_steps.append(i)
            if steps_per_iteration:
                steps_per_iteration.append(i-reward_steps[-2])
            else:
                steps_per_iteration.append(i)

            if env.firstSecuredFoodIteration == 0:
                steps_to_secured.append(epsiode_length)
                steps_secured_to_win.append(epsiode_length)
                times_lost_food.append(0)
            elif env.iteration >= env.max_iterations:
                steps_to_secured.append(env.firstSecuredFoodIteration)
                steps_secured_to_win.append(epsiode_length)
                times_lost_food.append(env.timesLostFood)
            else:
                steps_to_secured.append(env.firstSecuredFoodIteration)
                steps_secured_to_win.append(env.iteration - env.firstSecuredFoodIteration)
                times_lost_food.append(env.timesLostFood)
            rewards = []
            env.reset()            

  


if __name__ == "__main__":
    main()
