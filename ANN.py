# Clemence Le Cornec
# 06/07/2018
import tensorflow as tf
import numpy as np
import random
import os

class ANN(object):
	"""Defines the ANN structure, this is a parent class"""
	def __init__(self, learning_rate, min_delta, initialisation):
		""" Initialization of the model """
		self.sess = tf.Session()
		self.learningRate = learning_rate
		self.min_delta = min_delta
		
		# Build the model, the loss and the optimizer
		self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.output_data))
		self.__optimizer()
		self.saver = tf.train.Saver()
		
		# Initialization if the model is new. If the model is restored, initialisation is False
		if initialisation == True:
			init = tf.global_variables_initializer()
			self.sess.run(init)
			
	def displayData(self):
		""" Display the data available"""
		print(self.__dict__)
			
	def __earlyStopping (self, errorEpoch, errorPrevious, errorBest, counterDecreasePerformance, epoch, path):
		""" Early stopping if the error is constant or start to increase after patience epoch min_delta is the tolerance"""
		# Check if after the epoch the model has improved or not
		print(errorEpoch, errorPrevious, errorBest)
		
		if errorEpoch <= errorBest + self.min_delta:
			counterDecreasePerformance = 0
			errorBest = errorEpoch
			# If the folder to contain the model doesn't exist, create it
			if not os.path.exists(path):
				os.makedirs(path)
			self.save(epoch, path)
		elif errorEpoch > errorBest + self.min_delta:
			counterDecreasePerformance = counterDecreasePerformance + 1
		elif errorEpoch < errorPrevious + self.min_delta and errorEpoch > errorPrevious - self.min_delta:
			counterDecreasePerformance = counterDecreasePerformance + 1		
		
		# If the error isn't improving or is decerasing for self.patience consecutive epochs, stop training
		if counterDecreasePerformance <= self.patience:
			stopTraining = False
		else:
			stopTraining = True
			
		return counterDecreasePerformance, errorBest, stopTraining
		
	def __clipGradientIfNotNone(self, grad):
		""" Clip the gradient if it is not None"""
		if grad is None:
			return grad
		return tf.clip_by_value(grad, -1, 1)
		
	def __optimizer(self):
		""" Definition of the optimizer, clipped gradients are used to limit the overfitting
			This method is private and therefore not accessible from outside the object"""
		optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
		gvsAcc = optimizer.compute_gradients(self.loss)
		capped_gvsAcc = [(self.__clipGradientIfNotNone(grad), var) for grad, var in gvsAcc]
		self.minimize = optimizer.apply_gradients(capped_gvsAcc)
		
	def __next_batch(self, x_train, y_train, index_in_epoch, epochs_completed ,shuffle = True):
		""" Choose the next batch to pass to the model during the training, private method can
			only be called from inside the MLP object"""
		start = index_in_epoch
		# Shuffle for the first epoch
		if epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(len(x_train))
			np.random.shuffle(perm0)
			x_train = x_train[perm0]
		# Go to the next epoch
		if start + self.batchSize > len(x_train):
			# Finished epoch
			epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = len(x_train) - start
			images_rest_part = x_train[start:len(x_train)]
			y_rest_part = y_train[start:len(x_train)]
			# Shuffle the data
			if shuffle:
				perm = np.arange(len(x_train))
				np.random.shuffle(perm)
				x_train = x_train[perm]
				y_train = y_train[perm]
				# Start next epoch
				start = 0
				index_in_epoch = self.batchSize - rest_num_examples
				end = index_in_epoch
				images_new_part = x_train[start:end]
				return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((y_rest_part, y_train[start:end]), axis=0), index_in_epoch, epochs_completed
		else:
			index_in_epoch += self.batchSize
			end = index_in_epoch
		return x_train[start:end], y_train[start:end], index_in_epoch, epochs_completed
		
	def train(self, input_data, target, batch_size, total_epochs, path):
		""" Training procedure """
		
		self.batchSize = batch_size
		self.totalEpochs = total_epochs
		
		totalNbObservations = len(input_data)
		# Select randomly data for training and testing set
		trainingDataset = random.sample(range(0,totalNbObservations-1),int(0.90 * totalNbObservations))
		testingDataset = list(set(range(0,totalNbObservations-1))-set(trainingDataset))
		# Training set
		x_train = input_data[trainingDataset]
		y_train = target[trainingDataset]
		# Testing set
		x_test = input_data[testingDataset]
		y_test = target[testingDataset]
		
		# Total number of batch
		total_batch = int(len(x_train)/self.batchSize)
		
		# Prepare 
		errors = np.zeros((self.totalEpochs*total_batch,1))
		errorsTesting = np.zeros((self.totalEpochs*total_batch,1))
		
		# Initialize the variables for the early stopping
		counterDecreasePerformance = 0
		errorBest = 1000

		epochs_completed = 0
		index_in_epoch = 0
		step = 0

		for epoch in range(self.totalEpochs):
			# Loop over all batches
			for i in range(total_batch):
				x_batch, y_batch, index_in_epoch, epochs_completed = self.__next_batch(x_train, 
																	y_train, self.batchSize, 
																	index_in_epoch, epochs_completed)
				
				input = np.reshape(x_batch,[-1,self.nbInputs*self.nbSequences])
				output = np.reshape(y_batch,[-1,self.nbOutputs])
				self.sess.run(self.minimize,{self.input_data: input, self.target: output, self.DropoutRate : self.dropoutRateTraining})
				
				prediction = self.sess.run(self.output_data,{self.input_data:input, self.DropoutRate : 0})
				errors[step] = self.sess.run(self.loss,{self.target: output, self.output_data: prediction})
				errorsTesting[step] = self.sess.run(self.loss,{self.output_data:self.sess.run(self.output_data,
									{self.input_data:np.reshape(x_test,[-1,self.nbInputs*self.nbSequences]), self.DropoutRate:0}),
									self.target: np.reshape(y_test,[-1,self.nbOutputs]), self.DropoutRate:0})
				
				step = step + 1
			
			errorEpoch = np.mean(np.abs(errorsTesting[step-total_batch:step]))
			
			if epoch > 5:
				errorPrevious = np.mean(np.abs(errorsTesting[step-5*total_batch:step]))
			else:
				errorPrevious = np.mean(np.abs(errorsTesting[step-(epoch + 1)*total_batch:step]))
			print('Epoch ', epoch, ' in ', self.totalEpochs)
			
			# Check the performance and whereas or not we should stop the code
			#counterDecreasePerformance, errorBest, stopTraining = self.__earlyStopping (errorEpoch, errorPrevious,\
			# 														errorBest, counterDecreasePerformance, epoch, path)
			#if stopTraining is True:
				#print("Early Stopping")
				#break
		# Restore the last model saved
		# model = int(sorted(os.listdir(path), key=lambda x: os.path.getctime(path+x))[-1].split('.')[0])
		# self.restore(path, model)
		
		if not os.path.exists(path):
			os.makedirs(path)
		self.save(self.totalEpochs, path)
		self.errorsOnTrainingSet = errors
		self.errorsOnTestingSet = errorsTesting
	
	def prediction(self, input_data):
		""" Once the network is trained, predict the results for unknown data """
		input_data = np.reshape(input_data,[-1,self.nbInputs*self.nbSequences])
		prediction = self.sess.run(self.output_data,{self.input_data:input_data, self.DropoutRate : 0})
		return prediction
	
	def save(self,epoch, path):
		""" Save the trained model"""
		self.saver.save(sess =  self.sess, save_path = path, global_step =  int(epoch))
		print("Model saved!")
	
	def restore(self, path, model):
		""" Restore a trained model"""
		# Restore the model
		saved = tf.train.import_meta_graph(path + str(model) + ".meta")
		saved.restore(sess = self.sess, save_path = tf.train.latest_checkpoint(path))
		print("Model restored!")