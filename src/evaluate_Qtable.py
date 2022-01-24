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
    q_table = np.loadtxt("q_table_17000.txt")


    total_rewards = []
    rewards = 0

    done = False
    for i in range(10):
        while True:
            action = np.argmax(q_table[state,:]) # Exploit learned values
            if action == 1:
                rewards += 1
            next_state, reward, done, info = env.step(action) 
            state = next_state

            if done:
                total_rewards.append(np.sum(rewards))
                rewards = []
                break
    print(total_rewards)
  


if __name__ == "__main__":
    main()
