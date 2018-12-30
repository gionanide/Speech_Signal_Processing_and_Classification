#!usr/bin/python 
from __future__ import division
import random
import numpy as np
from numpy import cov
from numpy.linalg import eig, inv

def PrincipalComponentAnalysis(dimensions_output,kernel_option,c):
	#make a random array with samples lets say 100 samples with dimension 20
	samples = np.random.rand(40,3)

	#print samples
	print 'samples shape: ',samples.shape,'\n'

	#for every sample I took the square distance of the mean, this is the variable that I want to maximize
	#calculate the mean values of each columns, so we have to transpose the matrix because the argument axis refers to row
	mean = np.mean(samples.transpose(),axis=1)

	print 'mean shape: ',mean.shape,'\n'

	#print mean
	#print mean.shape

	#we are going to center our matrix(the points) to the origin (0,0) by substracting the column means
	#samples = samples - mean


	#calculate the covariance matrix between two features
	#the arrays are inserted as transposed that why they are transposed again
	if (kernel_option):
		print 'Using KernelPCA with rbf kernel \n'
		#here we are taking the dimensions of the samples array, so the dimensions of the data
		#for every sample
		#initialize a numpy array in the shape of the sample array
		covSamples = np.zeros((samples.shape[0],samples.shape[1]))
		for x in range(samples.shape[0]):
			#for all the dimensions of the samples
			for y in range(samples.shape[1]):
				#insert in numpy array for the first row(first sample) the first column is the first feature
				#minus the mean of the first feature		 
				np.put(covSamples[x],y,np.exp(-(np.linalg.norm(samples[x][y] - mean[y])**2/c)))
				#break
			#break
		#samples = np.absolute(samples - mean)**2/c
		#print samples.shape
		covSamples = np.matmul(covSamples.transpose(),covSamples)          
	else:
		print 'Using linear PCA \n'
		covSamples = np.matmul((samples - mean).transpose(),(samples - mean))

	print 'covariance matrix shape: ',covSamples.shape,'\n'
	print covSamples,'\n'

	#print covSamples.shape
	
	#print covSamples.shape
	#print covSamples

	#find eigenvalues and eigenvectors
	eigenvalues, eigenvectors = eig(covSamples)

	print 'eigenvectors shape: ',eigenvectors.shape,'\n'
	print eigenvectors,'\n'

	#short eigenvectors
	sorted_eigenvalues = eigenvalues.argsort()[::-1]

	print 'sorted eigenvalues: ',sorted_eigenvalues,'\n'

	print 'eigenvalues',eigenvalues,'\n'
	

	#deterine the dimensions you want to keep based on the eigenvectors you want to multiple the smples
	dimensions =  eigenvectors[:, sorted_eigenvalues]

	print 'w array shape: ',dimensions.shape,'\n'

	print 'w array sorted based on eigenvalues, every column represent one eigenvector \n',dimensions,'\n'

	#print dimensions.shape

	w = dimensions[:, :dimensions_output]

	print 'w final shape: ',w.shape,'\n'
	print 'w final array: \n',w,'\n'


	print 'final vector multiplication samples:',samples.shape,'w array:',w.shape,'\n'

	samples = np.dot(samples,w)
	#dimensions = np.dot(samples)

	print 'dimensionality reduction samples shape: ',samples.shape,'\n'
	#print samples[0]

	#for x in eigenvectors:
	#	print np.linalg.norm(x)

	

	


#PrincipalComponentAnalysis(dimensions_output=1,kernel_option=True,c=1)
