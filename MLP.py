# Clemence Le Cornec
# 06/07/2018
import tensorflow as tf
import numpy as np
import random
import os
from ANN import ANN

class MLP(ANN):
	""" MLP structure. Inherit from the ANN class """
	def __init__(self, nb_inputs, nb_outputs, nb_sequences, nb_layers, nb_neurons_per_layer, stddev,
			dropout_rate, learning_rate, min_delta, initialisation):
		""" Initialization of the model """
		self.nbLayers = nb_layers
		self.nbNeuronsPerLayer = nb_neurons_per_layer
		self.stddev = stddev
		self.nbInputs = nb_inputs
		self.nbOutputs = nb_outputs
		self.nbSequences = nb_sequences
		self.dropoutRateTraining = dropout_rate
		
		# Define input data, placeholders...
		self.input_data = tf.placeholder(tf.float32, shape = (None, self.nbInputs*self.nbSequences), name='input_data')
		self.target = tf.placeholder(tf.float32, shape = (None, self.nbOutputs),name='target')
		self.output_data = tf.placeholder(tf.float32, shape = (None, self.nbOutputs),name='output_data')
		self.DropoutRate = tf.placeholder(tf.float32, shape = (), name='DropoutRate')
		
		# Initialise the dictionaries containing the weights and biases
		self.weights = {}
		self.biases = {}
		
		# Input layer weights
		self.weights['h1'] = tf.Variable(tf.truncated_normal([self.nbInputs*self.nbSequences, self.nbNeuronsPerLayer[0]], mean=0.0,stddev=self.stddev))
		self.biases['b1'] = tf.Variable(tf.truncated_normal([self.nbNeuronsPerLayer[0]],mean=0.0,stddev=self.stddev))
		
		# Hidden layers weights
		for i in range(0,len(self.nbNeuronsPerLayer)-2):
			self.weights['h%s'%str(i+2)] = tf.Variable(tf.truncated_normal([self.nbNeuronsPerLayer[i], self.nbNeuronsPerLayer[i+1]], mean=0.0,stddev=self.stddev))
			self.biases['b%s'%str(i+2)] = tf.Variable(tf.truncated_normal([self.nbNeuronsPerLayer[i+1]],mean=0.0,stddev=self.stddev))
			
		# Output layer weights
		self.weights['hout'] = tf.Variable(tf.truncated_normal([self.nbNeuronsPerLayer[len(self.nbNeuronsPerLayer)-2], self.nbOutputs], mean=0.0,stddev=self.stddev))
		self.biases['bout'] = tf.Variable(tf.truncated_normal([self.nbOutputs],mean=0.0,stddev=self.stddev))
		
		self.__build_model()
		
		ANN.__init__(self, learning_rate, min_delta, initialisation)
	
	def displayData(self):
		""" Display the data available"""
		print(self.__dict__)
					
	def prepare_dataset(self, vehicle, emissions):
		""" Prepare the dataset for the training procedure"""
		step = 0
		
		input_data = np.zeros((len(vehicle.data) - self.nbSequences,self.nbInputs * self.nbSequences))
		target = np.zeros((len(emissions.data) - self.nbSequences, self.nbOutputs))
		
		while step < len(vehicle.data) - self.nbSequences:
			input_data[step][:] = np.reshape(np.transpose(np.vstack((vehicle.extractOnePID(\
								vehicle.list[0])[step:step+self.nbSequences],vehicle.extractOnePID(\
								vehicle.list[1])[step:step+self.nbSequences]))),
								[1,self.nbInputs*self.nbSequences])
			
			target[step][:] = emissions.data[step+self.nbSequences]
			step = step + 1
		
		return input_data, target
		
	def __build_model(self):
		""" Define the MLP architecture """
		h = tf.nn.elu(tf.matmul(self.input_data,self.weights['h1']) + self.biases['b1'])
		h = tf.nn.dropout(x = h, keep_prob = 1 - self.DropoutRate)
		for k in range(1,self.nbLayers-1):
			h = tf.nn.elu(tf.matmul(h,self.weights['h%s'%(str(k+1))]) + self.biases['b%s'%(str(k+1))])
			h = tf.nn.dropout(x = h, keep_prob = 1 - self.DropoutRate)
		output_data = tf.nn.elu(tf.matmul(h,self.weights['hout']) + self.biases['bout'])
		self.output_data = output_data