#!usr/bin/python
from __future__ import division
from python_speech_features import mfcc
import scipy.io.wavfile as wavv
import os
from sklearn.decomposition import IncrementalPCA, PCA
import sys
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC # support vectors for classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import timeit
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler

'''
We read the input file, we take the rate of the signal and the signal and then the mfcc feature extraction is on.
N numpy array with size of the number of frames, each row has one feature vector.
'''
def mfcc_features_extraction(wav):
	inputWav,wav = readWavFile(wav)
	print inputWav
	rate,signal = wavv.read(inputWav)
	mfcc_features = mfcc(signal,rate)
	return mfcc_features,wav

'''
Make a numpy array with length the number of mfcc features,
for one input take the sum of all frames in a specific feature and divide them with the number of frames. Because we extract 13 features
from every frame now we have to add them and take the mean of them in order to describe the sample.
'''
def mean_features(mfcc_features,wav,folder,general_feature_list,general_label_list):
	#here we are taking all the mfccs from every frame and we are not taking the average of them, instead we
	#are taking PCA in order to reduce the dimension of our data

	if (folder=='HC'):
		#map the lists, in the first position of the general_label_list it will be the label
		#of the sample which is in the first position in the list general_feature_list
		#and we are making this in order to write the sample to the file with the right labels
		general_label_list.append(0)
	elif(folder == 'PD'):
		general_label_list.append(1)

	#initialize the flattend list
	flattend_list = []
	
	#flat the list, for every frame take the 13 features and put them in one array
	for sublist in mfcc_features:
		for feature in sublist:
			flattend_list.append(feature)

	#check if a sample has les length than the length we determine
	if(len(flattend_list)<12800):
		print len(flattend_list)
		print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'

	#make the list of lists as a numpy array in order just one sample 1x(number_of_frames x features)
	#so for every sample we have all the features from all the frames in a single row.	
	#here we append in a list of lists the samples, we want to fill this list with all the samples
	general_feature_list.append(flattend_list)
	
	#in this function we just filling the two lists one with the features and one with the labels
	
	#sys.exit()
	


'''
given a path from the keyboard to read a .wav file this is for male , female. 
inputWav = '/home/gionanide/Theses_2017-2018_2519/MEEI-RainBow'+wav

this is for healthy, parskinson
inputWav = '/home/gionanide/Theses_2017-2018_2519/Gkagkos/Audio_Files'+wav
'''
def readWavFile(wav):
	#wav = raw_input('Give me the path of the .wav file you want to read: ')
	#inputWav = '/home/gionanide/Theses_2017-2018_2519/MEEI-RainBow'+wav
	inputWav = '/home/gionanide/Theses_2017-2018_2519/Gkagkos/Audio_Files'+wav
	return inputWav,wav

	


'''
write in a txt file the output vectors of every sample
parkinson,healthy database: featuresNewDatabase.
'''
def writeFeatures(general_feature_list,general_label_list,wav,folder):
	
	f = open('/home/gionanide/mfcc_parkinson_healthy_pca_feature.txt','a')
	

	#we have to iterato all the general_feature_list
	for x in range(len(general_feature_list)):
		#append the last element before you write to the file because it is the label

		print len(general_feature_list[x])
		
		#write it to the file after you append it
		np.savetxt(f,general_feature_list[x],newline=",")
		#write the label
		f.write(str(general_label_list[x]))
		#and change line
		f.write('\n')
	

'''
if i want to keep only the gender (male,female)
wav = wav.split('/')[1].split('-')[1], this is only for male,female classification
wav = wav.split('/')[1].split('-')[0], this is for edema,paralysis classification
wav.split('/')[1], for healthy,parkinson classification
'''

def makeFormat(folder):
	if (folder=='HC'):
		wav='0'
	elif(folder == 'PD'):
		wav='1'
	return wav


'''
def readCases():
	#healthy cases = 1.825 , we are going to mark these cases with 0(zero)
	#parkinson cases = 278 , and theses cases with 1(one)

	- now we want to take all the file names of a directory and them read them accordingly

	healthyCases = os.listdir('/home/gionanide/Theses_2017-2018_2519/Gkagkos/Audio_Files/HC')
	parkinsonCases = os.listdir('/home/gionanide/Theses_2017-2018_2519/Gkagkos/Audio_Files/PD')
	
	return healthyCases , parkinsonCases
'''

'takes the csv file and split the label from the features'
def splitData(data):
	# Split-out the set in two different arrayste
	array = data.values
	#features array contains only the features of the samples
	features = array[:,0:12800]
	#labels array contains only the lables of the samples
	labels = array[:,12800]	

	return features,labels

'''
make this class in order to train the model with the same amount of samples of each class, because we have bigger support from class 0
than class1, particularly it is 9 to 1.'''
def equalizeClasses(data):
	#take all the samples from the data frame that they have Label value equal to 0 and in the next line equal to 1
	class0 = data.loc[data['Label'] == 0]#class0 and class1 are dataFrames
	class1 = data.loc[data['Label'] == 1]


	#check which class has more samples, by divide them and check if the number is bigger or smaller than 1
	weight = len(class0) // len(class1) #take the results as an integer in order to split the class, using prior knowledge that 
	#class0 has more samples, if it is bigger class0 has more samples and to be exact weight to 1 

	balance = (len(class0) // weight) #this is the number of samples in order to balance our classes

	#the keyword argument frac specifies the fraction of rows to return in the random sample, so fra=1 means, return random all rows
	#we kind of a way shuffle our data in order not to take the same samples in every iteration
	#class0 = class0.sample(frac=1)
	
	#samples array for training taking the balance number of samples for the shuffled dataFrame
	newClass0 = class0.sample(n=balance)
	
	#and now combine the new dataFrame from class0 with the class1 to return the balanced dataFrame
	newData = pd.concat([newClass0, class1])	
	
	#return the new balanced(number of samples from each class) dataFrame
	return newData



'''we use this function in order to apply greedy search for finding the parameters that best fit our model. We have to mention
that we start this procedure from a very large field and then we tried to focues to the direction where the results
appear better. For example for the C parameter, the first range was [0.0001, 0.001, 0.01, 0.1, 1, 10 ,100 ,1000], the result was that
the best value was 1000 so then we tried [100, 1000, 10000, 100000] and so on in order to focues to the area which give us
the best results. This function is in comments because we found the best parameters and we dont need to run it in every trial.'''
def paramTuning(features_train, labels_train, nfolds):
	#using the training data and define the number of folds
	#determine the range of the Cs range you want to search
	Cs = [1000, 10010,10000, 10060, 100000, 1000000]

	#determine the range of the gammas range you want to search
	gammas = [0.00001, 0.0001, 0.005, 0.003 ,0.001, 0.01, 0.1]

	#make the dictioanry
	param_grid = {'C': Cs, 'gamma': gammas}

	#start the greedy search using all the matching sets from above
	grid_search = GridSearchCV(SVC(kernel='rbf'),param_grid,cv=nfolds)

	#fit your training data
	grid_search.fit(features_train, labels_train)

	#visualize the best couple of parameters
	print grid_search.best_params_



'''Classify Parkinson and Helathy. Building a model which is going to be trained with of given cases and test according to new ones'''
def classifyPHC(general_feature_list,general_label_list):
	#because we took features and labels seperatly we have to put them in the same list
	#and because for every signal we have different frames we took the first 12800 features
	for x in range(len(general_feature_list)):
		general_feature_list[x] = general_feature_list[x][:12800]	
		general_feature_list[x].append(general_label_list[x])

	#here because we have to make the dataframe again because the inputs are two lists 
	headers = []	
	#we initialize the headers/features
	for x in range(1,12801):
		headers.append('Feature'+str(x))
	headers.append('Label')
	
	print len(general_feature_list)
	print len(general_feature_list[0])
	
	#build the dataframe
	data = pd.DataFrame(general_feature_list,columns=headers)

	#equalize classes
	data = equalizeClasses(data)

	#data = equalizeClasses(data)
	features,labels = splitData(data)
	
	#determine the training and testing size in the range of 1, 1 = 100%
	validation_size = 0.2
	
	#here we are splitting our data based on the validation_size into training and testing data
	features_train, features_validation, labels_train, labels_validation = model_selection.train_test_split(features, labels, 
			test_size=validation_size)


	#determine the pca, and determine the dimension you want to end up
	pca = PCA(n_components=500)

	#fit only the features train
	pca.fit(features_train)

	#dimensionality reduction of features train
	features_train = pca.transform(features_train)

	#dimensionality reduction of fatures validation
	features_validation = pca.transform(features_validation)
	
	
	#normalize data in the range [-1,1]
	scaler = MinMaxScaler(feature_range=(-1, 1))
	#fit only th training data in order to find the margin and then test to data without normalize them
	scaler.fit(features_train)

	features_train = scaler.transform(features_train)

	#trnasform the validation features without fitting them
	features_validation = scaler.transform(features_validation)

	#we can see the shapes of the array just to check
	print 'feature training array: ',features_train.shape,'and label training array: ',labels_train.shape
	print 'feature testing array: ',features_validation.shape,'and label testing array: ',labels_validation.shape,'\n'


	#take the best couple of parameters from the procedure of greedy search
	#paramTuning(features_train, labels_train, 5)

	#we initialize our model
	svm = SVC(kernel='rbf',C=1000,gamma=1e-05,decision_function_shape='ovr')
	#svm = NearestNeighbors(n_neighbors=5)

	

	#train our model with the data that we previously precessed
	svm.fit(features_train,labels_train)

	#now test our model with the test data
	predicted_labels = svm.predict(features_validation)
	accuracy = accuracy_score(labels_validation, predicted_labels)
	print 'Classification accuracy: ',accuracy*100,'\n'

	#see the accuracy in training procedure
	predicted_labels_train = svm.predict(features_train)
	accuracy_train = accuracy_score(labels_train, predicted_labels_train)
	print 'Training accuracy: ',accuracy_train*100,'\n'

	#confusion matrix to illustrate the faulty classification of each class
	conf_matrix = confusion_matrix(labels_validation, predicted_labels)
	print 'Confusion matrix: \n',conf_matrix,'\n'
	print 'Support    class 0   class 1:'
	#calculate the support of each class
	print '          ',conf_matrix[0][0]+conf_matrix[0][1],'     ',conf_matrix[1][0]+conf_matrix[1][1],'\n'

	#calculate the accuracy of each class
	hC = (conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]))*100
	pC = (conf_matrix[1][1]/(conf_matrix[1][0]+conf_matrix[1][1]))*100

	#see the inside details of the classification
	print 'For class 0 man cases:',conf_matrix[0][0],'classified correctly and',conf_matrix[0][1],'missclassified,',hC,'accuracy \n'
	print 'For class 1 woman cases:',conf_matrix[1][1],'classified correctly and',conf_matrix[1][0],'missclassified,',pC,'accuracy\n'


	#try 5-fold cross validation
	scores = cross_val_score(svm, features_train, labels_train, cv=5)
	print 'cross validation scores for 5-fold',scores,'\n'
	print 'parameters of the model: \n',svm.get_params(),'\n'

	print 'number of samples used as support vectors',len(svm.support_vectors_),'\n'

	return svm.support_vectors_

'''
read all the files from both directories based on the keyboard input HC for healthy cases, PD fro parkinson disease
'''
def mainParkinson():
	general_feature_list = []
	general_label_list = []
	folder = raw_input('Give the name of the folder that you want to read data: ')
	if(folder == 'PD'):
		healthyCases = os.listdir('/home/gionanide/Theses_2017-2018_2519/Gkagkos/Audio_Files/PD')
		for x in healthyCases:
			wav = '/'+folder+'/'+str(x)
			mfcc_features,inputWav = mfcc_features_extraction(wav)
			mean_features(mfcc_features,inputWav,folder,general_feature_list,general_label_list)
		folder = raw_input('Give the name of the folder that you want to read data: ')
		if(folder == 'HC'):
			parkinsonCases = os.listdir('/home/gionanide/Theses_2017-2018_2519/Gkagkos/Audio_Files/HC')
			for x in parkinsonCases:
				wav = '/'+folder+'/'+str(x)
				mfcc_features,inputWav = mfcc_features_extraction(wav)
				mean_features(mfcc_features,inputWav,folder,general_feature_list,general_label_list)
		#print general_feature_list, general_label_list
		#writeFeatures(general_feature_list,general_label_list,wav,folder)
		classifyPHC(general_feature_list,general_label_list)
		
'''
main function, this example is for male,female classification 
given an input from the keyboard that determines the name of the File from which we want to read the samples, and
the number of the samples that we want to read
'''
def mainMaleFemale():
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	for x in range(1,int(amount)+1):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		mfcc_features,inputWav = mfcc_features_extraction(wav)
		mean_features(mfcc_features,inputWav,folder)



def main():
	#calculate the time
	import time
	start_time = time.time()

	#we are making an array in order to keep the support vectors and feed the function with them for the next iteration
	mainParkinson()

	time = time.time()-start_time
	print 'time: ',time


main()



