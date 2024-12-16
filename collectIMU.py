import time 
import numpy as np
import json
from serial import Serial
import os

def setupIMU():
    settings_path = os.path.join(os.getcwd(), '.vscode', 'settings.json')
    with open(settings_path, 'r') as file:
        settings = json.load(file)
        usbPath = settings.get("idf.port")
    ser = Serial(usbPath, baudrate=115200, timeout=1)
    return ser

def readIMU(ser, motionTime=5):
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
    return data