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

    env = DummyVecEnv([BetaRobotEnv,])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    model = DQN.load("src/logs[5]/best(5000).zip")

    forward_steps_list = []
    forward_steps = 0
    for i in range(10):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            
            # To test random baseline
            # action = random.randint(0,2)
            
            obs, reward, done, info = env.step([action,])
            if action == 1:
                forward_steps += 1
            
            if done:
                forward_steps_list.append(forward_steps)
                forward_steps = 0
                break
    print(forward_steps_list)

if __name__ == "__main__":
    main()