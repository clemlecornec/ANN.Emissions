# Clemence Le Cornec
# 09/07/2018

import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Use latex font for graphs
plt.rc('text', usetex = True)
plt.rc('font', family = 'sherif')
plt.rc('font', size = 14)

def list_of_files(path):
	""" Parse through the indicated folder and find the files to load"""
	listOfFiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	
	return listOfFiles

def load_DfT_data(path,file_name, sheetName, skipRows, skipFooter):
	""" Load the DfT vehicle data into arrays """
	path_data = path + "\\" + file_name
	data = pd.read_excel(path_data,sheet_name = sheetName, skiprows=skipRows,skipfooter = skipFooter).fillna(0)
	return data
	
def computeEmissionsInGPerS(emissionsToConvert, flowRate):
	""" Converts the emissions from ppm/s to g/s"""
	#emissions = emissionsToConvert * (41.3/24.45) * (flowRate / 1000)
	emissions = emissionsToConvert * (flowRate / (60*1000))
	
	return emissions

def computeAcceleration(data, columnSpeed):
	""" Compute the acceleration from the speed """
	
	data.loc[:,columnSpeed] = data.loc[:,columnSpeed] * 1000 / 3600
	
	acceleration = data.loc[1:len(data),columnSpeed].reset_index().values[:,1] - \
			data.loc[0:len(data)-2,columnSpeed].reset_index().values[:,1]
	acceleration = np.concatenate(([0],acceleration))
	
	data.loc[:,columnSpeed] = data.loc[:,columnSpeed] * 3600 / 1000
	
	data['acceleration'] = acceleration
	
	return data
		
def findIntPanisCoefficient(data, emissions):
	"""Computes the Int Panis coefficient from the training data - cf Methodes Quantitatives 1 EPFL"""
	
	data[:,0] = data[:,0] #* 1000 / 3600
	
	indexLowerThan5 = np.where(data[:,1]<-0.5)[0]
	indexHigherThan5 = np.where(data[:,1]>=-0.5)[0]
	
	dataPrepared_1 = np.transpose(np.vstack((np.ones((len(indexLowerThan5),1))[:,0], data[indexLowerThan5,0], data[indexLowerThan5,0]**2, data[indexLowerThan5,1], data[indexLowerThan5,1]**2, data[indexLowerThan5,0]*data[indexLowerThan5,1])))
	dataPrepared_2 = np.transpose(np.vstack((np.ones((len(indexHigherThan5),1))[:,0], data[indexHigherThan5,0], data[indexHigherThan5,0]**2, data[indexHigherThan5,1], data[indexHigherThan5,1]**2, data[indexHigherThan5,0]*data[indexHigherThan5,1])))
	
	intPanisCoefficients_1 = np.linalg.lstsq(dataPrepared_1, emissions[indexLowerThan5,0])
	intPanisCoefficients_2 = np.linalg.lstsq(dataPrepared_2, emissions[indexHigherThan5,0])
	
	intPanisCoefficients = np.vstack((intPanisCoefficients_1[0], intPanisCoefficients_2[0]))
	
	return intPanisCoefficients
	
def computeEmissionsAsIntPanis(data, intPanisCoefficients):
	"""Compute emissions as Int Panis et al., 2006, Modelling instantaneous traffic emission and the influence of traffic 
	speed limits. The parameters f1, f2, ..., f6 are in the format [f at a < -0.5, f at a >= -0.5]"""
	
	f1 = intPanisCoefficients[:,0]
	f2 = intPanisCoefficients[:,1]
	f3 = intPanisCoefficients[:,2]
	f4 = intPanisCoefficients[:,3]
	f5 = intPanisCoefficients[:,4]
	f6 = intPanisCoefficients[:,5]
	
	data[:,0] = data[:,0] #* 1000 / 3600
	
	# Prepare the emissions
	emissions = np.zeros((len(data),1))
	# Extract elements where the acceleration < -0.5
	index = np.where(data[:,1] < -0.5)[0]
	emissions[index,0] = f1[0] + f2[0] * data[index,0] + f3[0] * data[index,0] * data[index,0] + f4[0] * data[index, 1] + f5[0] * data[index, 1] * data[index, 1] + f6[0] * data[index, 0] * data[index, 1]
	# Extract the elements where the acceleration >= -0.5
	index = np.where(data[:,1] >= -0.5)
	emissions[index,0] = f1[1] + f2[1] * data[index,0] + f3[1] * data[index,0] * data[index,0] + f4[1] * data[index, 1] + f5[1] * data[index, 1] * data[index, 1] + f6[1] * data[index, 0] * data[index, 1]
	# Remove non zero emissions
	emissions[emissions < 0] = 0
	
	return emissions
	
def buildLookUpTable(data, NOxEmissions, binSize):
	"""Compute a look up table based on the training set data, uses here speed and acceleration """
	binSizeX = binSize[0]
	binSizeY = binSize[1]
	
	minX = np.min(data[:,0])
	maxX = np.max(data[:,0])
	minY = np.min(data[:,1])
	maxY = np.max(data[:,1])
	
	lookupTable = np.zeros((int(np.ceil((maxX-minX)/binSizeX)+1), int(np.ceil((maxY-minY)/binSizeY)+1)))
	
	# Find position in the look up table of each element
	x = np.floor(data[:,0] / binSizeX).astype(int)
	y = np.floor((data[:,1])-minY / binSizeY).astype(int)#-minY
	
	# Create a dataframe
	dataFinal = pd.DataFrame({'x':x, 'y':y, 'NOx':NOxEmissions[:,0]})
	
	# Group by same x and y and compute the mean
	final = dataFinal.groupby(['x','y'])['NOx'].mean().reset_index()
	
	# Build a numpy array with the results
	lookupTable[final['x'], final['y']] = final['NOx']
	
	return final, lookupTable
	
def computeEmissionsAsLookUpTable(data, lookupTable, binSize, minY):
	"""Uses look-up table to compute the emissions"""
	binSizeX = binSize[0]
	binSizeY = binSize[1]
	
	x = np.floor(data[:,0] / binSizeX).astype(int)
	y = np.floor((data[:,1]-minY) / binSizeY).astype(int)

	# Read the corresponding NOx emissions in the look-up table
	x[x > (len(lookupTable)-1)] = len(lookupTable)-1
	y[y > len(lookupTable[0])-1] = len(lookupTable[0])-1
	
	x[x < -len(lookupTable)] = 0
	y[y < -len(lookupTable[0])] = 0
	
	# Read the corresponding emission in the lookup table
	emissions = lookupTable[x,y]

	return emissions
	
def plotLookUpTable(lookupTable, binSize):
	""" Plot the look up table"""
	fig = plt.figure()
	plt.scatter(lookupTable['x'].values*binSize[0], lookupTable['y'].values*binSize[1], c = lookupTable['NOx'].values, s = 12, marker = 's', cmap = 'RdPu')
	plt.xlabel('Speed [km/h]')
	plt.ylabel('Acceleration [m/s2]')
	plt.title('Look up table')
	cb = plt.colorbar()
	cb.set_label('NOx emissions [g/s]')
	#plt.show()
	
def applyCOPERT(COPERTfile, speed, interval, category, euroStandard):
	"""Apply the COPERT coefficient to compute emissions"""
	
	# Read the file containing the COPERT emission factors
	copert = pd.read_excel(COPERTfile, sheet_name = "Cars", skiprows = range(0,7))
	index = copert[(copert["Fuel / Size"] == category) & (copert["Euro standard"] == euroStandard)]
	
	# Extract emissions factors
	alpha = index[["ALPHA"]].values[0][0]
	beta = index[["BETA"]].values[0][0]
	gamma = index[["GAMMA"]].values[0][0]
	delta = index[["DELTA"]].values[0][0]
	epsilon = index[["EPSILON"]].values[0][0]
	zeta = index[["ZITA"]].values[0][0]
	eta = index[["ITA"]].values[0][0] 
	theta = index[["THITA"]].values[0][0]
	RF = index[["RF"]].values[0][0]
	
	# Split the speed vector into one interval pieces and compute the mean on these sections
	numberOfSegments = math.ceil(np.cumsum(speed)[-1] / interval)  # speed is in m/s and the data resolution is at 1 Hz
	
	ElementBegin  = 0
	v = np.zeros((len(speed),1))
	
	for k in range(1,numberOfSegments+1):
		
		ElementStop = np.where(np.cumsum(speed) <= k * interval)[0][-1]
		meanSpeed = np.mean(speed[ElementBegin:ElementStop]) * 3600 / 1000 # conversion to km/h
		
		if meanSpeed < 10:
			meanSpeed = 10
		if meanSpeed > 130:
			meanSpeed = 130
		
		v[ElementBegin:ElementStop] = meanSpeed * np.ones((len(speed[ElementBegin:ElementStop]),1))
		
		ElementBegin  = ElementStop
	
	# Last element missing
	v[-1] = v[-2]
	
	# Compute emissions for each segment [Euro 3,4,5,6 formula]
	emissionFactor = ((alpha + gamma * v + epsilon * v**2 + (zeta / v)) / (1 + beta * v + delta * v**2))* (1 - RF)
	
	return emissionFactor
	
def statisticalIndices(real, prediction, distance, save = False, path = "/", name = "*.txt"):
	""" List of statistical indices used to quantify the accordance between numerical and experimental
		datasets: FB is the fractional bias, ER is the relative error, NMSE the normal mean square error,
		Corr the correlation coefficient, MG the geometric mean bias, VG the geometric variance and FAC2
		the fraction of data that satisfies 0.5 < Cpredicted/Creal < 2"""
	realEmission = np.sum(real) / distance
	predictedEmission = np.sum(prediction) / distance
	real = real + 0.01 * np.ones((len(real),1))
	prediction = prediction + 0.01 * np.ones((len(prediction),1))
	FB = 2 * (np.mean(real) - np.mean(prediction)) / (np.mean(real) + np.mean(prediction))
	ER = np.mean(2 * np.abs(real - prediction) / (real + prediction) )
	NMSE = (np.mean((real - prediction)**2)/(np.mean(real) * np.mean(prediction)))**0.5
	Corr = np.mean((real - np.mean(real)) * (prediction - np.mean(prediction))) / (np.mean((real - np.mean(real))**2 * (prediction - np.mean(prediction))**2))**0.5
	mask = (real > 0) & (prediction > 0)
	MG = np.exp(np.mean(np.log(real[mask])) - np.mean(np.log(prediction[mask])))
	VG = np.exp(np.mean((np.log(real[mask]) - np.log(prediction[mask]))**2))
	fraction = prediction / real
	mask = (0.5 < fraction) & (fraction < 2)
	FAC2 = len(fraction[mask]) / len(fraction)
	
	# Save the results in a text file
	if not os.path.exists(path):
		os.makedirs(path)
	if save is True:
		file = open(path + name, "w")
		file.write("real emission [g/km] = %s" % realEmission)
		file.write("\n")
		file.write("predicted emission [g/km] = %s" % predictedEmission)
		file.write("\n")
		file.write("FB = %s" % FB)
		file.write("\n")
		file.write("ER = %s" % ER)
		file.write("\n")
		file.write("NMSE**0.5 = %s" % NMSE)
		file.write("\n")
		file.write("Corr = %s" % Corr)
		file.write("\n")
		file.write("MG = %s" % MG)
		file.write("\n")
		file.write("VG = %s" % VG)
		file.write("\n")
		file.write("FAC2 = %s" % FAC2)
		file.close()
	
	return realEmission, predictedEmission, FB, ER, NMSE, Corr, MG, VG, FAC2
	
	
def plotError(model, save, name):
	"""Plot the evoluation of error during training"""
	fig = plt.figure()
	plt.plot(model.errorsOnTrainingSet,'m')
	plt.plot(model.errorsOnTestingSet,'k')
	plt.xlabel("Iterations")
	plt.ylabel("MSE")
	plt.title("Evolution of the error during training")
	if save == True:
		plt.savefig(name, format = 'png', dpi = 300)
	# plt.show()
	
def plotPrediction(prediction, target, nbSequence, suptitle, title, xlabel, ylabel, save = False, name = 'defaultName.eps'):
	"""Plot the results of the training procedure (prediction)"""
	fig = plt.figure(figsize = (12,10))
	gs = gridspec.GridSpec(4,2)
	# Times series (first subplot)
	ax0 = plt.subplot(gs[1:4,0])
	ax0.plot(range(nbSequence,nbSequence+len(target)),target,'b',label='Real emission')
	#ax0.plot(range(0,len(target)),target,'b',label='Real emission')
	ax0.plot(range(nbSequence,nbSequence+len(target)),prediction,'r',label='Prediction')
	ax0.legend(frameon = False)
	ax0.set_xlabel(xlabel[0])
	ax0.set_ylabel(ylabel[0])
	ax0.set_title(title[0])
	# Residual (second subplot)
	ax1 = plt.subplot(gs[0,0], sharex = ax0)
	ax1.plot(range(nbSequence,nbSequence+len(target)),target - prediction[0:len(target)],'k',label='Residuals')
	ax1.set_title(title[1])
	ax1.set_ylabel(ylabel[2])
	# Plot the cumulative sum
	ax2 = plt.subplot(gs[2:4,1]) 
	ax2.plot(range(nbSequence,nbSequence+len(target)),np.cumsum(target),'b',label='Real emission')
	#ax2.plot(range(0,len(target)),np.cumsum(target),'b',label='Real emission')
	ax2.plot(range(nbSequence,nbSequence+len(target)),np.cumsum(prediction[0:len(target)]),'r',label='Prediction')
	ax2.set_xlabel(xlabel[2])
	ax2.set_ylabel(ylabel[2])
	ax2.set_title(title[2])
	ax2.legend(frameon = False)
	# Correlation plot
	ax3 = plt.subplot(gs[0:2,1]) 
	ax3.scatter(target,prediction,marker='o',color='b', facecolors = 'none')
	ax = plt.gca()
	lims = [np.min([ax.get_xlim(),ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
	ax3.plot(lims,lims,'k-',alpha=0.75)
	ax3.set_xlabel(xlabel[3])
	ax3.set_ylabel(ylabel[3])
	ax3.set_title(title[3])
	plt.suptitle(suptitle)
	plt.subplots_adjust(hspace = 1, wspace = 0.5)
	if save is True:
		plt.savefig(name, format = 'png', dpi = 300)
	#plt.show()
	plt.close("all")