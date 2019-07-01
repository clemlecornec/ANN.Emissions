# Clemence Le Cornec
# 03/07/2018
import numpy as np

class Emissions(object):
	""" Represent a emission data object """
	def __init__(self, data, minMaxValues, vehicleID):
		""" Class initialization """
		self.data = data
		self.minMaxValues = minMaxValues
		self.vehicleID = vehicleID
		self.normalized = 0
	
	def displayData(self):
		""" Display the data available"""
		print(self.__dict__)
		
	def normalizeData(self):
		""" Normalization of the emissions"""
		self.data = (self.data - np.tile(self.minMaxValues[0],len(self.data)).reshape(self.data.size))\
					/ (np.tile(self.minMaxValues[1],len(self.data)).reshape(self.data.size)\
					- np.tile(self.minMaxValues[0],len(self.data)).reshape(self.data.size))
		self.normalized = 1
	
	def denormalizeData(self):
		""" Denormalize the data """
		self.data = self.data * (np.tile(self.minMaxValues[1],len(self.data)).reshape(self.data.size)\
					- np.tile(self.minMaxValues[0],len(self.data)).reshape(self.data.size))\
					+ np.tile(self.minMaxValues[0],len(self.data)).reshape(self.data.size)
		self.normalized = 0
	