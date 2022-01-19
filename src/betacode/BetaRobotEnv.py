from asyncio import base_tasks
from betacode.BetaRobot import BetaRobot

import sys
import signal

import numpy as np
import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


class BetaRobotEnv(gym.Env):
    def __init__(self):
        self.robot = BetaRobot()

        self.robot.resetRobot()
    
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(5,), dtype=np.float32)
    
    def reset(self):
        self.robot.resetRobot()
        observation =  self.robot.readIR()
        return observation
    
    def step(self, action):

        self.robot.makeMove(action)
        reward = self.robot.getFitness()
        isStuck = self.robot.checkIfStuck()
        observation =  self.robot.readIR()
        self.robot.updateEvalStats()

        if isStuck:
            reward -= 100

        return observation, reward, isStuck, {}
    
    def check_env(self):
        check_env(self, warn=True)