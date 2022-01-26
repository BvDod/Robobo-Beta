from asyncio import base_tasks
from betacode.BetaRobot import BetaRobot

import sys
import signal

import numpy as np



def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


class BetaRobotEnv():
    def __init__(self, physical=False, max_iterations=500, screen_segments=3):
        self.physical = physical
        self.robot = BetaRobot(physical=physical, screen_segments=screen_segments)
        self.iteration = 0
        self.max_iterations = 500
        self.screen_segments = screen_segments

        if not physical:
            self.robot.resetRobot()
    
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

        if self.physical:
            return observation, 0, False, {}
        return observation, reward, self.iteration == self.max_iterations or self.robot.foodCollected == 11, {}
    