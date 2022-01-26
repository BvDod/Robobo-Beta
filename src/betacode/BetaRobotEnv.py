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
    def __init__(self, physical=False, max_iterations=500, screen_segments=3):
        self.physical = physical
        self.robot = BetaRobot(physical=physical, screen_segments=screen_segments)
        self.iteration = 0
        self.max_iterations = 500
        self.screen_segments = screen_segments

        if not physical:
            self.robot.resetRobot()
    
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(6)
    
    def reset(self):
        if not self.physical:
            self.robot.resetRobot()
        observation =  self.robot.readCamera()
        self.iteration = 0

        

        return observation
    
    def step(self, action):
        self.iteration += 1
        self.robot.makeMove(action)
        observation =  self.robot.readCamera()
        
        if not self.physical:
            reward = self.robot.getFitness()
        
        if not self.physical:
            self.robot.updateEvalStats()


        return observation, reward, self.iteration == self.max_iterations or self.robot.foodCollected == 11, {}
    