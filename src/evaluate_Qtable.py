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

    "training_results/Task2/5+1"
    q_table = np.loadtxt("training_results/Task2/5+1/q_table_87000.txt")
    bottom_segment = False
    env = BetaRobotEnv(screen_segments=q_table.shape[0] - 1 + int(bottom_segment))
    


    rewards = 0
    total_rewards = []
    total_moves = []
    done = False
    for i in range(10):
        rewards = 0
        moves = 0
        state = env.reset()
        while True:
            moves += 1
            action = np.argmax(q_table[state,:]) # Exploit learned values
            #action =  random.randint(0,q_table.shape[1]-1)
            next_state, reward, done, info = env.step(action) 
            if reward >= 100:
                rewards += 1
            state = next_state

            if done:
                total_rewards.append(rewards)
                total_moves.append(moves)
                state = env.reset()
                break
        print(total_rewards)
        print(total_moves)
    print(total_rewards)
  


if __name__ == "__main__":
    main()
