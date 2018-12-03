#!usr/bin/python
from python_speech_features import mfcc
import scipy.io.wavfile as wavv
import numpy as np
import os
from sklearn.decomposition import IncrementalPCA, PCA

'''
Make a numpy array with length the number of mfcc features,
for one input take the sum of all frames in a specific feature and divide them with the number of frames. Because we extract 13 features
from every frame now we have to add them and take the mean of them in order to describe the sample. In our previous example we take
the mean of all this features, in this case we are using PCA to conclude to a single feature vector (1,13) with dimensionality reduction.
'''
def mean_features(mfcc_features,wav,folder):
	#here we are taking all the mfccs from every frame and we are not taking the average of them, instead we
	#are taking PCA in order to reduce the dimension of our data

	#make the list of lists as a numpy array in order to keep from them 15578x13 just one samples 1x13	
	flattend_mfcc = np.array(mfcc_features)

	#just to check the shape of the array before transapose
	#print flattend_mfcc.shape

	#because the shape of the array is (1199,13) se if we apply PCA we are going to keep just the number of columns we define
	#but this is not the point we want to keep all the columns but just one row (dimensionality reduction)
	#so we reshape our array in order to be (13,1199) so keeping all the rows, all our features, but just one column
	#so we reduce our dimension from 1199 to 13
	flattend_mfcc = flattend_mfcc.transpose()

	#confirm that we trnaspose the arraty
	#print flattend_mfcc.shape
	#initialize the pca
	pca = PCA(n_components=1)
	
	#fit the features in the model
	pca.fit(flattend_mfcc)

	#apply PCA and keep just one column, which means one feature vector with 13 features
	sample = pca.transform(flattend_mfcc)

	#because the result is (13,1) we want to make it a feature vector se we want to reshape it like (1,13)
	sample = sample.transpose()
	
	#transform it to a list in order to satisfy the format for writing the feature vector in the file
	pca_features = sample.tolist()

	#and keep just the first list, because it returns you a list of lists with only one list
	pca_features = pca_features[0]

	print pca_features
