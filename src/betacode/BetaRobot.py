#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np
import math
import time
import random

from robobo import SimulationRobobo
import cv2
import sys
import signal
import prey
import vrep

from betacode.HelperFunctions import random_point_in_circle
from betacode.Camera import Camera
import betacode.startLocations as startLocations


class BetaRobot:
    def __init__(self, physical=False, screen_segments=3, add_bottom_segment=False, generalist=False):
        self.physical = physical
        if not physical:
            self.rob = SimulationRobobo().connect(address='127.0.0.1', port=19997)
        else:
            from robobo import HardwareRobobo
            self.rob = HardwareRobobo(camera=True).connect("10.15.3.235")
        
        if not physical:
            self.robotID = self.getObjectHandle("Robobo")
            self.foodHandle = self.getObjectHandle("Food")
            self.baseHandle = self.getObjectHandle("Base")

        self.TotalIterations = 0
        self.generalist = generalist
        
        # Parameters
        if self.physical:
            self.maxSpeed = 20         # Should be between 0 and 100, used to scale the "moves"
        else:
            self.maxSpeed = 30
        self.collidingDistance = 0.035       # Distance from which we count something as a collision
        self.maxDetectableDistance = 0.20   # Dont measure distance to obstacle above this, defined in vrep also!

        # Paramters for reseting env if robot stuck
        self.terminateDuration = 150       # Amount of iterations after which we reset environment if no movement seen
        self.terminateDisplacement = 0.005  # Amount of displacement we consider "no movement"
        self.terminateDisplacement *= (self.maxSpeed/100)

        # The relative speed of the tires when making specific moves
        self.moves = {
            "fullLeft" : (-0.4, 0.5),
            "halfLeft" : (-0.4, 0.5),
            "straight" : (1, 1),
            "halfRight" : (0.5, -0.4),
            "fullRight" : (0.5, -0.4),
            "stop" : (0, 0),
        }

        self.moves_length = {
            "fullLeft" : 2000,
            "halfLeft" : 500,
            "straight" : 1000,
            "halfRight" : 500,
            "fullRight" : 2000,
            "stop" : 500,
        }

        self.int_to_moves = {0: "halfLeft", 1:"straight", 2: "halfRight"}
    
        # Ids of objects to use with api
        if not physical:
            self.objectHandles, self.distanceHandles = self.getAllHandles()

        # Data that is updated each "iteration"
        self.lastSpeed = (0, 0)
        self.lastPosition = [0, 0]
        self.averageDisplacement = 0
        self.positionIterations = 0
        self.lastMove = None

        self.foodCollected = 0
        
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
        self.starting_locations = startLocations.startLocations


        self.screen_segments = screen_segments
        self.camera = Camera(screen_segments=self.screen_segments, add_bottom_segment=add_bottom_segment, physical=physical)
        if self.physical:
            self.rob.set_phone_tilt(110, 100)
        else:
            self.rob.set_phone_tilt(1/4*math.pi, 100)


        

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
    
    def readCamera(self, debug=True):
        """Function used to read and transform the IR"""
        image = self.rob.get_image_front()
        if self.physical:
            image = cv2.flip(image, 0)
        
        self.foodIsInGrip = self.camera.hasSecuredFood(image)

        if self.foodIsInGrip == True:
            mask_ratio, blobs = self.camera.getBlobs(image, mode="green")
        else:
            mask_ratio, blobs = self.camera.getBlobs(image, mode="red")


        self.mask_ratio = mask_ratio
        camera_area = self.camera.getCameraArea(blobs)
        
        if not self.generalist:
            if self.foodIsInGrip:
                camera_area += (self.screen_segments + 1)

        return camera_area, self.foodIsInGrip


    def normalizeIRVector(self, vector):
        """ Applies vector normalization """
        norm = np.linalg.norm(vector)
        normal_array = vector/norm
        return normal_array
    
    def getFitness(self):
        """ Get current fitness of the robot """
        fitness = (self.mask_ratio * 16) + (self.isFoodAtBase() * 100) + (self.foodIsInGrip * 4) 
        """
        print(self.mask_ratio * 16)
        print(self.isFoodAtBase() * 100)
        print(self.foodIsInGrip * 4)
        """
        return fitness
    
    def isFoodAtBase(self):
        self.foodIsAtBase = self.rob.base_detects_food()
        return self.foodIsAtBase
    
    def isNewFoodCollected(self):
        amount = self.rob.collected_food()
        if not amount == self.foodCollected:
            self.foodCollected = amount
            return True
        return False


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
        location_robot, location_food, location_base = random.choice(self.starting_locations)
        self.setStartLocation(location_robot)
        self.randomizeStartAngle()
        self.setLocation(self.foodHandle, location_food)
        self.setLocation(self.baseHandle, location_base)
        self.startSimulation()
        print(f"[{location_robot}, {location_food}, {location_base}],")

        
        # Reset variables
        self.lastSpeed = (0, 0)
        self.lastPosition = [0, 0]
        self.averageDisplacement = 0
        self.positionIterations = 0
        self.consecutiveCollision = 0
        self.foodCollected = 0

        if self.physical:
            self.rob.set_phone_tilt(110, 100)
        else:
            self.rob.set_phone_tilt(1/4*math.pi, 100)
            
        for i in range(2):
            self.makeMove("stop", dontIterate=True)
        
        self.rob.wait_for_ping()
    

    
    def randomizeStartLocation(self):
        """ Function used to randomize the location of the environment """
        center = [0, 0, 0.037]
        offset = random_point_in_circle(1.8)
        new_location = [center[0] + offset[0], center[1] + offset[1], center[2]]
        self.setStartLocation(new_location)
        return new_location
    
    def randomizeStartAngle(self):
        """ Function used to randomize the angle of the environment """
        random_euler_angles = (random.uniform(0, math.pi) , 0, 0)
        self.setStartAngle(random_euler_angles)
        return random_euler_angles
    

    def setStartLocation(self, location):
        """ Function used to randomize the location of the environment """
        vrep.unwrap_vrep(vrep.simxSetObjectPosition(self.rob._clientID, self.robotID, -1, location, vrep.simx_opmode_blocking))
    
    def setLocation(self, handle, location):
        """ Function used to randomize the location of the environment """
        vrep.unwrap_vrep(vrep.simxSetObjectPosition(self.rob._clientID, handle, -1, location, vrep.simx_opmode_blocking))
    
    def setStartAngle(self, angles, mode="world"):
        """ Function used to randomize the location of the environment """
        vrep.unwrap_vrep(vrep.simxSetObjectOrientation(self.rob._clientID, self.robotID, self.objectHandles[0], angles, vrep.simx_opmode_blocking))
        

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

        objectHandles.append(self.getObjectHandle("Cuboid"))
        # distanceHandles.append(self.getDistanceHandle("Distance"))
        
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
        '''
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
        '''

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
    
