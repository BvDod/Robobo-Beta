from asyncio import base_tasks
from betacode.BetaRobot import BetaRobot

import sys
import signal

import numpy as np



def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


class BetaRobotEnv():
    def __init__(self, physical=False, max_iterations=500, screen_segments=3, add_bottom_segment=False, generalist=False):
        self.physical = physical
        self.robot = BetaRobot(physical=physical, screen_segments=screen_segments, add_bottom_segment=add_bottom_segment, generalist=generalist)
        self.iteration = 0
        self.max_iterations = max_iterations
        self.screen_segments = screen_segments
        
        
        self.firstSecuredFoodIteration = 0
        self.currentlyHasFood = False
        self.timesLostFood = 0

        if not physical:
            self.robot.resetRobot()
    
    def reset(self):
        if not self.physical:
            self.robot.resetRobot()
        observation =  self.robot.readCamera()
        
        self.iteration = 0
        self.firstSecuredFoodIteration = 0
        self.timesLostFood = 0
        self.currentlyHasFood = False

        return observation
    
    def step(self, action):
        self.iteration += 1
        self.robot.makeMove(action)
        observation, has_red_block =  self.robot.readCamera()

        if has_red_block & (self.firstSecuredFoodIteration == 0):
            self.firstSecuredFoodIteration = self.iteration
        
        if has_red_block & (not self.currentlyHasFood):
            self.currentlyHasFood = True

        if (not has_red_block) & self.currentlyHasFood:
            self.currentlyHasFood = False
            self.timesLostFood += 1
            
        if not self.physical:
            reward = self.robot.getFitness()
            self.robot.updateEvalStats()

        if self.physical:
            return observation, 0, False, {}
        return observation, reward, self.iteration == self.max_iterations or reward >= 100, {}
    