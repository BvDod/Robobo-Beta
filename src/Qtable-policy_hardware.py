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


    # Use this code instead of above code to test the robot
    q_table = np.loadtxt("q_table_17000.txt")
    env = BetaRobotEnv(physical=True)
    i = 0
    state = env.reset()
    while True:
        action = np.argmax(q_table[state,:]) # Exploit learned values
        if action == 1:
            i += 1
        next_state, reward, done, info = env.step(action) 
        state = next_state
        print(i)


        #nrobot.readIR()
        
        # robot.getFitness()
        # robot.checkIfStuck()
        # robot.updateEvalStats()


    robot.pauseSimulation()
    



if __name__ == "__main__":
    main()
