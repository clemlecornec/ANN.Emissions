# Clemence Le Cornec - 03.07.2018
# Python script to test the Vehicle Class

from Vehicles import *
import numpy as np

# Create artificially data to test the Vehicle class
data = np.array([[1,2,3],[4,5,6],[7,8,9]])
dataList = ['1','2','3']
minValues = np.array([0, 2, 1])
maxValues = np.array([10, 8, 12])
manufacturer = 'Volvo'
aftertreatment = 'LNT'

# Create a Vehicle object
vehicle = Vehicles(data = data, dataList = dataList, minValues = minValues,\
					maxValues = maxValues, manufacturer = manufacturer, aftertreatment = aftertreatment)



vehicle.displayData()
print(vehicle.data)
vehicle.normalizeData()
print(vehicle.data)
vehicle.denormalizeData()
print(vehicle.data)
PIDextracted = vehicle.extractOnePID('1')
print(PIDextracted)


