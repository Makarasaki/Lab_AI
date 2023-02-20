#!/usr/bin/python3

from numpy.core.numeric import Infinity
import vrep
import sys
import time
import numpy as np
from tank import *
import skfuzzy as fuzz
from skfuzzy import control as ctrl

vrep.simxFinish(-1) # closes all opened connections, in case any prevoius wasnt finished
clientID=vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # start a connection

if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")
    
#create instance of Tank
tank=Tank(clientID)

proximity_sensor=0
error_code, proximity_sensor_handler = vrep.simxGetObjectHandle(clientID,"Proximity_sensor", vrep.simx_opmode_blocking)
 
err_code,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,proximity_sensor_handler,vrep.simx_opmode_streaming)

# distance control
distance = ctrl.Antecedent(np.arange(0, 7, 0.5), 'distance')
distance['very close'] = fuzz.trimf(distance.universe, [0, 1, 1.5])
distance['close'] = fuzz.trimf(distance.universe, [1, 2, 3])
distance['medium'] = fuzz.trimf(distance.universe, [1, 3, 6])
distance['far'] = fuzz.trimf(distance.universe, [4, 10, 20])

# velocity control
velocity = ctrl.Consequent(np.arange(-1, 10, 0.5), 'velocity')
velocity['stop'] = fuzz.trimf(velocity.universe, [-1, 0, 1])
# velocity['very slow'] = fuzz.trimf(velocity.universe, [0, 1, 2])
velocity['slow'] = fuzz.trimf(velocity.universe, [1, 4, 5])
velocity['medium'] = fuzz.trimf(velocity.universe, [4, 6, 8])
velocity['fast'] = fuzz.trimf(velocity.universe, [6, 8, 10])

rule1 = ctrl.Rule(distance['far'], velocity['fast'])
rule2 = ctrl.Rule(distance['medium'], velocity['medium'])
rule3 = ctrl.Rule(distance['close'], velocity['slow'])
rule4 = ctrl.Rule(distance['very close'], velocity['stop'])

velocity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

if __name__ == '__main__':
    tank.forward()
    #continue reading and printing values from proximity sensors
    err_code, handler = vrep.simxGetObjectHandle(clientID,"Proximity_sensor", vrep.simx_opmode_blocking)
    t = time.time()
    it = 0
    while (time.time()-t)<360:
        it += 1
        err_code,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID,handler,vrep.simx_opmode_buffer)
        
        velocity_ctrl_sim = ctrl.ControlSystemSimulation(velocity_ctrl)
        velocity_ctrl_sim.input['distance'] = np.linalg.norm(detectedPoint)
        velocity_ctrl_sim.compute()
        
        tank.leftvelocity = velocity_ctrl_sim.output['velocity']
        tank.rightvelocity = velocity_ctrl_sim.output['velocity']
        tank.setVelocity()
        if it%1000==0:
            print( err_code,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector)
            print('Velocity:', velocity_ctrl_sim.output['velocity'])