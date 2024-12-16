import argparse
import pickle
import time
from djitellopy import Tello
import subprocess
import os
from playsound import playsound
from collectIMU import setupIMU, readIMU

parser = argparse.ArgumentParser()
parser.add_argument('--noDrone', action='store_true', help="indicates if drone is enabled")
args = parser.parse_args()
if args.noDrone:
    print("Running without drone")
# You can run `master.py` or `master.py --noDrone` if you want the drone included or not
# This is useful for testing the model

# Load imu controller
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

# Load imu collector
ser = setupIMU()

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
    battery_level = tello.get_battery()
    print(f"Battery level: {battery_level}%")
print("drone started successfully\n\n\n")

# main loop
numToWord = {1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'forward', 7: 'turn-left', 8: 'turn-right', 9: 'still'}
isFlying = False
while True:
    # signal that we're collecting data
    print("collecting data now")
    playsound('sounds/collecting.mp3')
    # collect IMU data
    data = readIMU(ser)
    # signal we're predicting label from data
    print("predicting label")
    flatData = data[:300].flatten() # crop to 3 seconds at 100hz, shape = (300,6)
    predictedLabel = numToWord[svm_classifier.predict([flatData])[0]]
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
                case 'still':
                    print("idleing")
    # signal that we're resting now
    # print("resting for three seconds")
    # time.sleep(3)
    print()