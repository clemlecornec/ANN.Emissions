# Clemence Le Cornec - 03.07.2018
# Python script to test the Emissions Class

from Emissions import *
import numpy as np

# Create artificially data to test the Emissions class
data = np.array([[0.00021],[0.00052],[0.00123]])
minMaxValues = np.array([0, 0.014])
vehicleID = 12

# Create an Emissions object
emissions = Emissions(data = data, minMaxValues = minMaxValues, vehicleID = vehicleID)

# Test functions
print(emissions.data)
emissions.normalizeData()
print(emissions.data)
emissions.denormalizeData()
print(emissions.data)
