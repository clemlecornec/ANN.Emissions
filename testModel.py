# Clemence Le Cornec - 11.07.2018
# Python script to try to validate the MLP model

# The data comes from DfT
from MLP import *
from Vehicles import *
from Emissions import *
from utils import *
import numpy as np
import os

# Find files
path = os.getcwd() + "\\Data\\"
fileTraining = [f for f in os.listdir(path + "Training\\")][0]
fileValidation = [f for f in os.listdir(path + "Testing\\")][0]
pathSaveModel = os.getcwd() + "\\Results\\" + fileValidation.split(".")[0] + "\\"

# MLP parameters
nInputs = 2
nOutputs = 1
nSequences = 5
stddev = 0.1
dropout_rate = 0.20
batch_size = 64
total_epochs = 16
learning_rate = 0.001
min_delta = 1 * 10**-5
patience = 12
nLayers = 5
nNeuronsPerLayer = [100,75,50,25,1]

#################################################### TRAINING ####################################################

# Load the data
dataTraining = load_DfT_data(path + "Training\\", fileTraining, sheetName = "R-Combined", skipRows = 0, skipFooter = 0)
dataTraining = computeAcceleration(dataTraining, columnSpeed = "GPS_VehicleSpeed")
dataValidation = load_DfT_data(path + "Testing\\", fileValidation, sheetName = "R-Combined", skipRows = 0, skipFooter = 0)
dataValidation = computeAcceleration(dataValidation, columnSpeed = "GPS_VehicleSpeed")

# Create a vehicle training object
data = dataTraining[['GPS_VehicleSpeed', 'acceleration']].as_matrix()
dataList = ['speed', 'acceleration']
minValues = [np.min(dataTraining['GPS_VehicleSpeed'].values), np.min(dataTraining['acceleration'].values)]
maxValues = [np.max(dataTraining['GPS_VehicleSpeed'].values), np.max(dataTraining['acceleration'].values)]
manufacturer = fileTraining.split(".")[0]
aftertreatment = "Unknown"
vehicle = Vehicles(data = data, dataList = dataList, minValues = minValues,\
					maxValues = maxValues, manufacturer = manufacturer, aftertreatment = aftertreatment)

# Normalize data
vehicle.normalizeData()

# Create an emission training object
noxEmissions = computeEmissionsInGPerS(emissionsToConvert = dataTraining['GA_NOxConc'].values, flowRate = dataTraining['PF_ExhaustFlowRate'].values)
noxEmissions [noxEmissions < 0] = 0
minMaxValues = [np.min(noxEmissions), np.max(noxEmissions)]
vehicleID = 0
emissions = Emissions(data = noxEmissions, minMaxValues = minMaxValues, vehicleID = vehicleID)
# Normalize data
emissions.normalizeData()

# Create an LSTM object
MLPModel = MLP(nb_inputs = nInputs, nb_outputs = nOutputs, nb_sequences = nSequences, nb_layers = nLayers,
				nb_neurons_per_layer = nNeuronsPerLayer, stddev = stddev, dropout_rate = dropout_rate, 
				learning_rate = learning_rate, min_delta = min_delta, initialisation = True)
				
# Test prepare dataset
input_data, target = MLPModel.prepare_dataset(vehicle = vehicle, emissions = emissions)

# Train the model
MLPModel.train(input_data, target, batch_size, total_epochs, pathSaveModel)

# Plot error
plotError(MLPModel, save = True, name = pathSaveModel + "plotError.png")

################################################# PREDICTION ##########################################################

# Create a vehicle validation object
data = dataValidation[['GPS_VehicleSpeed', 'acceleration']].as_matrix()
dataList = ['speed', 'acceleration']
manufacturer = fileValidation.split(".")[0]
aftertreatment = "Unknown"
vehicleValidation = Vehicles(data = data, dataList = dataList, minValues = minValues,\
					maxValues = maxValues, manufacturer = manufacturer, aftertreatment = aftertreatment)
# Normalize data
vehicleValidation.normalizeData()

# Create an emission validation object
noxEmissionsValidation = computeEmissionsInGPerS(emissionsToConvert = dataValidation['GA_NOxConc'].values, flowRate = dataValidation['PF_ExhaustFlowRate'].values)
noxEmissionsValidation [noxEmissionsValidation < 0] = 0
vehicleID = 0
emissionsValidation = Emissions(data = noxEmissionsValidation, minMaxValues = minMaxValues, vehicleID = vehicleID)

input_data_validation, target_validation = MLPModel.prepare_dataset(vehicle = vehicleValidation, emissions = emissionsValidation)

prediction = MLPModel.prediction(input_data_validation)

# Denormalize the results
prediction = prediction * (np.tile(minMaxValues[1],len(prediction)).reshape((len(prediction),1))\
					- np.tile(minMaxValues[0],len(prediction)).reshape((len(prediction),1)))\
					+ np.tile(minMaxValues[0],len(prediction)).reshape((len(prediction),1))

prediction [prediction < 0] = 0

# Compute the statistical indices
distance = np.sum(dataValidation['GPS_VehicleSpeed'].values/3600)
realEmission, predictedEmission, FB, ER, NMSE, Corr, MG, VG, FAC2 = statisticalIndices(real = target_validation, prediction = prediction, distance = distance, save = True, path = pathSaveModel, name = "Statistics.txt")

# Plot the results
suptitle = "Manufacturer: %s, Type of test: Real driving"%(manufacturer)
title = ["Real-time instantaneous NO$_x$ emissions", "Residuals","Real-time cumulative NO$_x$ emissions", "Correlation"]
xlabel = ["Time [s]", "Time [s]","Time [s]", "emitted NO$_x$ [g/s]"]
ylabel = ["NO$_x$ [g/s]", "NO$_x$ [g/s]","NO$_x$ [g]", "predicted NO$_x$ [g/s]"]
plotPrediction(prediction, target_validation, nSequences, suptitle, title, xlabel, ylabel, save = True, name = pathSaveModel + 'Plot.png')

########################################### RESTORE AND PREDICT ########################################################

tf.reset_default_graph()

ModelRestored = MLP(nb_inputs = nInputs, nb_outputs = nOutputs, nb_sequences = nSequences, nb_layers = nLayers,
				nb_neurons_per_layer = nNeuronsPerLayer, stddev = stddev, dropout_rate = dropout_rate,
				learning_rate = learning_rate, min_delta = min_delta, initialisation = False)

modelName = [f for f in os.listdir(pathSaveModel) if f.endswith('.meta')][0].split('.meta')[0]
ModelRestored.restore(pathSaveModel, model = modelName)

# Make a prediction
predictionModelRestored = ModelRestored.prediction(input_data_validation)

# Denormalize the results
predictionModelRestored = predictionModelRestored * (np.tile(minMaxValues[1],len(predictionModelRestored)).reshape((len(predictionModelRestored),1))\
					- np.tile(minMaxValues[0],len(predictionModelRestored)).reshape((len(predictionModelRestored),1)))\
					+ np.tile(minMaxValues[0],len(predictionModelRestored)).reshape((len(predictionModelRestored),1))

predictionModelRestored [predictionModelRestored < 0] = 0

# Compute the statistical indices
distance = np.sum(dataValidation['GPS_VehicleSpeed'].values/3600)
realEmission, predictedEmission, FB, ER, NMSE, Corr, MG, VG, FAC2 = statisticalIndices(real = target_validation, prediction = predictionModelRestored, distance = distance, save = True, path = pathSaveModel, name = "Statistics_ModelRestored.txt")

# Plot the results
suptitle = "Manufacturer: %s, Type of test: Real driving"%(manufacturer)
title = ["Real-time instantaneous NO$_x$ emissions", "Residuals","Real-time cumulative NO$_x$ emissions", "Correlation"]
xlabel = ["Time [s]", "Time [s]","Time [s]", "emitted NO$_x$ [g/s]"]
ylabel = ["NO$_x$ [g/s]", "NO$_x$ [g/s]","NO$_x$ [g]", "predicted NO$_x$ [g/s]"]
plotPrediction(predictionModelRestored, target_validation, nSequences, suptitle, title, xlabel, ylabel, save = True, name = pathSaveModel + 'Plot_ModelRestored.png')