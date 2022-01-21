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


    env = DummyVecEnv([BetaRobotEnv,])
    policy_kwargs = dict(layers=[5])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    model = DQN.load("src/logs[5]/best(5000).zip")

    total_rewards = []
    rewards = []
    for i in range(10):
        obs = env.reset()
        while True:
            """
            action, _states = model.predict(obs)
            """
            action = random.randint(0,2)
            obs, reward, done, info = env.step([action,])

            if action == 1:
                rewards.append(1)
            
            if done:
                total_rewards.append(np.sum(rewards))
                rewards = []
                break
        print(i)
    print(total_rewards)

if __name__ == "__main__":
    main()