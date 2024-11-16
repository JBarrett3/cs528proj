import argparse
import json
import pickle
from serial import Serial
import time
import numpy as np
from djitellopy import Tello
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument('--noDrone', action='store_true', help="indicates if drone is enabled")
args = parser.parse_args()
if args.noDrone:
    print("Running without drone")
# You can run `master.py` or `master.py --noDrone` if you want the drone included or not
# This is useful for testing the model

# control IMU
print("starting IMU")
idf_path = os.environ.get("IDF_PATH", "/Users/jamesbarrett/esp/esp-idf")  # idf_path depends on the local setup, so you may need to change it
script_to_run = "startIMU.sh"
command = f"source {idf_path}/export.sh && bash {script_to_run}"
with open(os.devnull, 'w') as devnull:
    imuProcess = subprocess.run(command, shell=True, executable="/bin/bash", stdout=devnull, stderr=devnull)
if imuProcess.returncode == 0:
    print("IMU started successfully")
else:
    print("IMU started unsuccessfully with return code:", imuProcess.returncode)

# Load model
print("loading Model")
with open('genModel/model.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)
print("Model loaded successfully")

# Load drone controller
print("starting drone")
if not args.noDrone:
    tello = Tello()
    tello.connect()
settings_path = os.path.join(os.getcwd(), '.vscode', 'settings.json')
with open(settings_path, 'r') as file:
    settings = json.load(file)
    usbPath = settings.get("idf.port")
ser = Serial(usbPath, baudrate=115200, timeout=1)
print("drone started successfully\n\n\n")

motionTime = 4
numToWord = {1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'forward', 7: 'turn-left', 8: 'turn-right', 9: 'idle'}
isFlying = False
while True:
    # signal that we're collecting data
    print("collecting data now")
    t_end = time.time() + motionTime
    data = np.zeros((0,6))
    while time.time() < t_end:
        readline = ser.readline().decode('utf-8').strip()
        splitLine = readline.split(", ")
        if len(splitLine) == 6:
            acceXs, acceYs, acceZs, gyroXs, gyroYs, gyroZs = splitLine[0].split(":"), splitLine[1].split(":"), splitLine[2].split(":"), splitLine[3].split(":"), splitLine[4].split(":"), splitLine[5].split(":")
            if len(acceXs) == 3 and len(acceYs) == 2 and len(acceZs) == 2 and len(gyroXs) == 2 and len(gyroYs) == 2 and len(gyroZs) == 2:
                acceX, acceY, acceZ, gyroX, gyroY, gyroZ = acceXs[2], acceYs[1], acceZs[1], gyroXs[1], gyroYs[1], gyroZs[1]
                data = np.append(data, [np.array([acceX, acceY, acceZ, gyroX, gyroY, gyroZ])], axis=0)
    # signal we're predicting label from data
    print("predicting label")
    flatData = data[:300].flatten() # crop to 3 seconds at 100hz, shape = (300,6)
    predictedLabelNum = svm_classifier.predict([flatData])[0]
    predictedLabel = numToWord[predictedLabelNum]
    print("predicted", predictedLabel)
    # signal that we're responding now
    print("sending command to drone")
    if not args.noDrone:
        if not isFlying:
            if predictedLabel == 'up':
                # on ground and accepts command to take off
                tello.takeoff()
                print("taking off")
                isFlying = True
            else:
                # on ground and can't accept any command except take off
                print("drone has not taken off")
        else:
            # in air and ready to take other commands
            match predictedLabel:
                case 'up':
                    tello.move_up(30)
                    print("going up")
                case 'down':
                    if tello.get_height() < 30: # TODO test get_height() accuracy
                        # withing 30 centimeters of ground
                        tello.land()
                        isFlying = False
                        print("landing")
                    else:
                        tello.move_down(30)
                        print("going down")
                case 'left':
                    tello.move_left(30)
                    print("going left")
                case 'right':
                    tello.move_right(30)
                    print("going right")
                case 'forward':
                    tello.move_forward(30)
                    print("going forward")
                case 'backward':
                    tello.move_back(30)
                    print("going backward")
                case 'turn-left':
                    tello.rotate_counter_clockwise(90)
                    print("turning left")
                case 'turn-right':
                    tello.rotate_clockwise(90)
                    print("turning right")
                case 'still': # TODO if stills not being classified as still, add more data with partially still and partially moving
                    print("idleing")
    # signal that we're resting now
    print("resting for three seconds")
    print()
    time.sleep(3)