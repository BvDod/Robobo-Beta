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

    robot = BetaRobot()

    robot.resetRobot()
    robot.makeMove("stop")
    i = 0
    start_time = time.time()
    while True:
        i += 1
        robot.executeBaseline()
        robot.getFitness()
        robot.checkIfStuck()
        robot.updateEvalStats()

    robot.pauseSimulation()



if __name__ == "__main__":
    main()
