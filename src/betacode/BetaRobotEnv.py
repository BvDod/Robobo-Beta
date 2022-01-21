from asyncio import base_tasks
from betacode.BetaRobot import BetaRobot

import sys
import signal

import numpy as np
import gym
from gym import spaces



def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


class BetaRobotEnv(gym.Env):
    def __init__(self):
        self.robot = BetaRobot()

        self.robot.resetRobot()
    
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(6)
    
    def reset(self):
        self.robot.resetRobot()
        observation =  self.robot.readIR()

        if np.all(observation == 0):
            observation = 5
        else:
            observation = np.argmax(observation)

        return observation
    
    def step(self, action):
        self.robot.makeMove(action)
        reward = self.robot.getFitness()
        isStuck = self.robot.checkIfStuck()
        observation =  self.robot.readIR()

        # Disctresize observation
        if np.all(observation == 0):
            observation = 5
        else:
            observation = np.argmax(observation)
        self.robot.updateEvalStats()

        if isStuck:
            reward -= 100

        return observation, reward, isStuck, {}
    
    def check_env(self):
        check_env(self, warn=True)