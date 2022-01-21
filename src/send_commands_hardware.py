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



def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    """
    env = BetaRobotEnv()
    policy_kwargs = dict(net_arch=[5])
    
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, 
                verbose=1, 
                learning_rate=0.001, 
                target_update_interval=200,
                learning_starts=1000,
                batch_size=128,
                exploration_fraction=0.2,
                exploration_final_eps=0.1,
                ).learn(20000)
    """
    


    # Use this code instead of above code to test the robot
    
    robot = BetaRobot(physical=True)

    robot.makeMove("stop")
    i = 0
    while True:

        i += 1
        #nrobot.readIR()
        robot.executeBaseline()
        # robot.getFitness()
        # robot.checkIfStuck()
        # robot.updateEvalStats()


    robot.pauseSimulation()
    



if __name__ == "__main__":
    main()
