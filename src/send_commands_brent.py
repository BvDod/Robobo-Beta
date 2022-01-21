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

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

from stable_baselines.common.callbacks import CheckpointCallback


def terminate_program(signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)


    BetaRobotEnv()
    policy_kwargs = dict(layers=[3])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')

    model = DQN(MlpPolicy, env, verbose=1)
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, 
                verbose=1, 
                learning_rate=0.001, 
                prioritized_replay=True,
                target_network_update_freq=500,
                learning_starts=1000,
                batch_size=64,
                exploration_fraction=0.05,
                exploration_final_eps=0.05,
                tensorboard_log="./tensorboard/",
                gamma=0.975
                ).learn(100000, callback=checkpoint_callback)
    


    # Use this code instead of above code to test the robot
    
<<<<<<< HEAD
    robot = BetaRobot(physical=True)
=======
    robot = BetaRobot()
>>>>>>> a43297f2717d5524ff4d25d430ec02bfb9589c04

    robot.resetRobot()
    robot.makeMove("stop")
    i = 0
    while True:

        i += 1
        se
        robot.executeBaseline()
        # robot.getFitness()
        # robot.checkIfStuck()
        # robot.updateEvalStats()


    robot.pauseSimulation()
    



if __name__ == "__main__":
    main()
