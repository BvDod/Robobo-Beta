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

from betacode.HelperFunctions import random_point_in_circle



class BetaRobot:
    def __init__(self, physical=False):
        self.physical = physical
        if not physical:
            self.rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        else:
            self.rob = robobo.HardwareRobobo().connect("10.15.3.235")
        
        if not physical:
            self.robotID = self.getObjectHandle("Robobo")

        self.TotalIterations = 0
        
        # Parameters
        self.maxSpeed = 25          # Should be between 0 and 100, used to scale the "moves"
        self.collidingDistance = 0.035       # Distance from which we count something as a collision
        self.maxDetectableDistance = 0.20   # Dont measure distance to obstacle above this, defined in vrep also!
        self.update_interval = 1000/8     # ms the robot should read sensors and move (1000/30 = 30hz for example)

        # Paramters for reseting env if robot stuck
        self.terminateDuration = 150       # Amount of iterations after which we reset environment if no movement seen
        self.terminateDisplacement = 0.005  # Amount of displacement we consider "no movement"
        self.terminateDisplacement *= (self.maxSpeed/100)

        # The relative speed of the tires when making specific moves
        self.moves = {
            "fullLeft" : (-0.25, 0.5),
            "halfLeft" : (-0.25, 0.5),
            "straight" : (1, 1),
            "halfRight" : (0.5, -0.25),
            "fullRight" : (0.5, -0.25),
            "stop" : (0, 0),
        }

        self.moves_length = {
            "fullLeft" : 2000,
            "halfLeft" : 1000,
            "straight" : 500,
            "halfRight" : 1000,
            "fullRight" : 2000,
            "stop" : 500,
        }

        self.int_to_moves = {0: "fullLeft", 1:"halfLeft", 2:"straight", 3:"halfRight", 4: "fullRight"}
    
        # Ids of objects to use with api
        if not physical:
            self.objectHandles, self.distanceHandles = self.getAllHandles()

        # Data that is updated each "iteration"
        self.lastSpeed = (0, 0)
        self.lastPosition = [0, 0]
        self.averageDisplacement = 0
        self.positionIterations = 0
        self.lastMove = None

        
        self.EvalUpdateRate = 1000
        self.EvalStartTime = time.time()
        self.evalTimes = []

        # Evaluation stats
        self.CollisionCount = 0
        self.CollisionRate = []
        self.WheelSpeedSum = 0
        self.AvgWheelSpeed = []
        self.consecutiveCollision = 0


        # Location and angles of starting positions that work
        self.starting_locations = [
        [[1.5906763022571035, -0.0556484690257477, 0.25], [2.1975553278341806, 0, 0]],
        [[0.5016798490535783, -0.15607350686774132, 0.25], [0.7817498281686088, 0, 0]],
        [[-0.9720390282859366, 0.06129349846488446, 0.25], [-0.6767864147074749, 0, 0]],
        [[-1.0003626201620628, 0.7837414851760482, 0.25], [-0.31387512686168506, 0, 0]],
        [[-0.5660457988203468, -1.3148253477324483, 0.25], [-0.16884032293871787, 0, 0]],
        [[0.5617552702981955, -0.017153199243642312, 0.25], [0.36331885824583976, 0, 0]],
        [[0.7876358767404442, -0.5243433050741434, 0.25], [-0.030591235493026225, 0, 0]],
        [[-0.8498098565039783, 1.4949962169182174, 0.25], [-0.4626504830438387, 0, 0]],
        [[-1.6839245043784712, -0.30083414218809573, 0.25], [1.455193192961179, 0, 0]],
        [[-0.5318329738725882, -0.5227905118361207, 0.25], [1.5674624045287064, 0, 0]],
        [[-0.10064955234833874, -0.6907052118532505, 0.25], [0.9667288710305453, 0, 0]],
        [[1.2802100353798298, 0.28901793185901054, 0.25], [1.5960659102294805, 0, 0]],
        [[-1.2114305718894163, 0.4153677616138217, 0.25], [-1.752219097786276, 0, 0]],
        [[0.2560259236324979, -0.2617935404828697, 0.25], [-0.0029854381520393325, 0, 0]],]

        

    def makeMove(self, move, dontIterate=False):
        """ Function used to execute one of the predefined moves (see self.moves in __init__())"""
        if not dontIterate:
            self.TotalIterations += 1

        if type(move) != str:
            move = self.int_to_moves[move]

        
        if move not in self.moves:
            print("Error: move does not exist")
        
        base_speed = (self.maxSpeed, self.maxSpeed)
        # Calculate speed of left and right tire and execute move
        move_speed_multiplier = self.moves[move]
        speed_left, speed_right = base_speed[0] * move_speed_multiplier[0], base_speed[1] * move_speed_multiplier[1]
        self.rob.move(int(speed_left), int(speed_right), self.moves_length[move])
        
        self.lastSpeed = (speed_left, speed_right)
        self.lastMove = move
    

    def readIR(self):
        """Function used to read and transform the IR"""
        if self.physical:
            values = np.array(self.rob.read_irs()[3:])
            values = values - np.array([0,35,0,25,0])
            print(values)
            self.min_value = 20
            values[values < self.min_value] = 0
            values = np.flipud(values)
            return values

        values = np.log(np.array(self.rob.read_irs()[3:]))/10 * -1
        values[values == np.inf] = 0
        values = np.flipud(values)
        if not np.all(values == 0):
            values = self.normalizeIRVector(values)
        return values

    def normalizeIRVector(self, vector):
        """ Applies vector normalization """
        norm = np.linalg.norm(vector)
        normal_array = vector/norm
        return normal_array
    
    def getFitness(self):
        """ Get current fitness of the robot """
        distanceToClosest, isColliding = self.getDistance()
        totalSpeed = self.lastSpeed[0] + self.lastSpeed[1]

        # Fitness has 3 different components
        speedFitness = int(self.lastMove == "straight")*4 + -2   # range is -2 to 2
        distanceFitness = (distanceToClosest / self.maxDetectableDistance) * 2    # range is 0 to 1
        # collisionFitness = (isColliding) * -20

        fitness = speedFitness + distanceFitness
        return fitness

    def updateAvgDisplacement(self):
        """ Update the moving average of the displacement of the robot """
        displacement = self.getDisplacement()
        self.averageDisplacement += (1/self.terminateDuration) * (displacement - self.averageDisplacement)


    def checkIfStuck(self):
        """ Check if robot has been stuck somewhere for too long, reset environment if this is the case """
        self.updateAvgDisplacement()
        if self.positionIterations > self.terminateDuration:
            if self.averageDisplacement < self.terminateDisplacement:
                print("Robot is stuck, resetting...")
                return True
        if self.consecutiveCollision >= 5:
            return True
        return False
    

    def resetRobot(self):
        """ Reset the current environment and the robot itself """
        self.stopWorld()
        self.rob.wait_for_stop()

        # Randomize location and angle of robot by randomzing location and angle of the environment
        self.setStartLocation([0, 0, 0.25])
        self.setStartAngle([0,0.5*math.pi,0], mode="world")

        location, angle = random.choice(self.starting_locations)

        print("Randomzing starting location, starting location and angle: ")
        print(location, angle)
        print()

        self.setStartLocation(location)
        angle[1] += 0.5*math.pi
        self.setStartAngle(angle)

        # angle = self.randomizeStartAngle()
        # location = self.randomizeStartLocation()

        # print(f"Random angle: {angle}, Random location: {location}")
        self.startSimulation()
        
        # Reset variables
        self.lastSpeed = (0, 0)
        self.lastPosition = [0, 0]
        self.averageDisplacement = 0
        self.positionIterations = 0
        self.consecutiveCollision = 0

        
        for i in range(5):
            self.makeMove("stop", dontIterate=True)
        
        self.rob.wait_for_ping()
    

    def randomizeStartAngle(self):
        """ Function used to randomize the angle of the environment """
        random_euler_angles = (random.uniform(-math.pi, math.pi), 0, 0)
        vrep.unwrap_vrep(vrep.simxSetObjectOrientation(self.rob._clientID, self.objectHandles[0], self.objectHandles[0], random_euler_angles, vrep.simx_opmode_blocking))
        return random_euler_angles
    

    def randomizeStartLocation(self):
        """ Function used to randomize the location of the environment """
        center = [0, 0, 0.25]
        offset = random_point_in_circle(1.8)
        new_location = [center[0] + offset[0], center[1] + offset[1], center[2]]
        self.setStartLocation(new_location)
        return new_location
    
    def randomizeStartAngle(self):
        """ Function used to randomize the angle of the environment """
        random_euler_angles = (random.uniform(-math.pi, math.pi), 0, 0)
        self.setStartAngle(random_euler_angles)
        return random_euler_angles
    

    def setStartLocation(self, location):
        """ Function used to randomize the location of the environment """
        vrep.unwrap_vrep(vrep.simxSetObjectPosition(self.rob._clientID, self.objectHandles[0], -1, location, vrep.simx_opmode_blocking))
    
    def setStartAngle(self, angles, mode="self"):
        """ Function used to randomize the location of the environment """
        if mode == "self":
            handle = self.objectHandles[0]
            angles[1] += 0.5*math.pi
        if mode == "world":
            handle = -1
        vrep.unwrap_vrep(vrep.simxSetObjectOrientation(self.rob._clientID, self.objectHandles[0], handle, angles, vrep.simx_opmode_blocking))
        

    def getDisplacement(self):
        """ Get displacement from last to current iteration """
        self.positionIterations += 1

        x_curr, y_curr = self.rob.position()[:2]
        x_last, y_last = self.lastPosition
        displacement = math.hypot(abs(x_curr - x_last), abs(y_curr - y_last))
        self.lastDisplacement = displacement
        self.lastPosition = (x_curr, y_curr)
        return displacement
    

    def getDistance(self):
        """ Get distance from collidable closest to robot to the robot. Has max distance from which it triggers"""
        
        distance = vrep.unwrap_vrep(vrep.simxReadDistance(self.rob._clientID, self.distanceHandles[0], vrep.simx_opmode_blocking))
        if distance > 1000:                         # if the closest object is further than max distance, vrep returns 10e37
            distance = self.maxDetectableDistance   # reduce this to max detectable distance

        if distance < self.collidingDistance:
            if self.lastMove == "straight":
                self.consecutiveCollision += 1
            else:
                self.consecutiveCollision += 2.5
            self.isColliding = True
            return distance, True
        self.isColliding = False
        self.consecutiveCollision = 0
        return distance, False
    

    def getPosition(self):
        """ Get Current x,y position of the robot """
        return self.rob.position()[:2]


    def getAllHandles(self):
        """ Function used to get all object handles (ID) to use with the API"""
        objectHandles = []
        distanceHandles = []

        objectHandles.append(self.getObjectHandle("Cuboid2"))
        distanceHandles.append(self.getDistanceHandle("Distance"))
        
        return objectHandles, distanceHandles
    
    def executeBaseline(self):
        """ Baseline controller """
        sensor_values = list(self.readIR())
        if all([value == 0 for value in sensor_values]):
            move = "straight"
        else:
            if sum(sensor_values[0:2]) > sum(sensor_values[3:]):  
                move = "fullRight"
            else:
                move = "fullLeft"
        self.makeMove(move)

    
    def updateEvalStats(self):
        """ Function to update the evaluation stats"""
        self.CollisionCount += int(self.isColliding)
        self.WheelSpeedSum += (self.moves[self.lastMove][0] + self.moves[self.lastMove][1])
        if self.TotalIterations % self.EvalUpdateRate == 0:
            self.CollisionRate.append(self.CollisionCount/ self.EvalUpdateRate)
            self.CollisionCount = 0
            self.AvgWheelSpeed.append(self.WheelSpeedSum/ self.EvalUpdateRate)
            self.WheelSpeedSum = 0
            currentTime = self.rob.get_sim_time()
            self.evalTimes.append((currentTime - self.EvalStartTime)/1000)
            self.EvalStartTime = currentTime
            print(f"Eval stats last {self.EvalUpdateRate} it: {self.CollisionRate[-1]} collisions, avg wheel speed: {self.AvgWheelSpeed[-1]}, took {self.evalTimes[-1]}s")
        
        

    def getObjectHandle(self, name):
        """ Get handle (ID) of an object"""
        return vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, name, vrep.simx_opmode_blocking))
    
    def getDistanceHandle(self, name):
        """ Get handle (ID) of an distance object"""
        return vrep.unwrap_vrep(vrep.simxGetDistanceHandle(self.rob._clientID, name, vrep.simx_opmode_blocking))

    def startSimulation(self):
        self.rob.play_simulation()
    
    def pauseSimulation(self):
        self.rob.pause_simulation()
    
    def stopWorld(self):
        self.rob.stop_world()
