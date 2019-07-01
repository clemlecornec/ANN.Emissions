# Clemence Le Cornec
# 03/07/2018
import numpy as np

class Vehicles(object):
	""" Represent a vehicle object """
	def __init__(self, data, dataList, minValues, maxValues, manufacturer, aftertreatment):
		""" Class initialization """
		""" To make the class more flexible, the input data is given as an array and the description 
		of the data available is in the list variable. In this list, the order of the name is the same 
		than the order of the data. minValues and maxValues are the minimum and maximun values possible
		for each of the input data"""
		self.data = data
		self.list = dataList
		self.minValues = minValues
		self.maxValues = maxValues
		self.manufacturer = manufacturer
		self.aftertreatment = aftertreatment
		self.normalized = 0
		
	def displayData(self):
		""" Display the data available"""
		print(self.__dict__)
		
	def extractOnePID(self, PIDToExtract):
		"""Extraction of one PID"""
		indexPID = self.list.index(PIDToExtract)
		PID = self.data[:,indexPID]
		return PID
		
	def normalizeData(self):
		""" Normalize the data """
		self.data = (self.data - np.tile(self.minValues,len(self.data)).reshape((len(self.data),len(self.list))))\
					/ (np.tile(self.maxValues,len(self.data)).reshape((len(self.data),len(self.list)))\
					- np.tile(self.minValues,len(self.data)).reshape((len(self.data),len(self.list))))
		self.normalized = 1
	
	def denormalizeData(self):
		""" Denormalize the data """
		self.data = self.data * (np.tile(self.maxValues,len(self.data)).reshape((len(self.data),len(self.list)))\
					- np.tile(self.minValues,len(self.data)).reshape((len(self.data),len(self.list))))\
					+ np.tile(self.minValues,len(self.data)).reshape((len(self.data),len(self.list)))
		self.normalized = 0