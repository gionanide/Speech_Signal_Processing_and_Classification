#!usr/bin/python
from __future__ import division
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC # support vectors for classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import timeit
import numpy as np
import itertools
import sys


'''this function takes as an input the path of a file with features and labels and returns the content of this file as a csv format in
the form feature1.........feature13,Label'''
def readFile():
	#make the format of the csv file. Our format is a vector with 13 features and a label which show the condition of the
	#sample hc/pc : helathy case, parkinson case
  	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
	'Feature10','Feature11','Feature12','Feature13','Label']

	#path to read the samples, samples consist from healthy subjects and subject suffering from Parkinson's desease.
	path = '/home/gionanide/Theses_2017-2018_2519/features/parkinson_healthy/mfcc_parkinson_healthy.txt'
	#read file in csv format
	data = pd.read_csv(path,names=names )
	
	#return an array of the shape (2103, 14), lines are the samples and columns are the features as we mentioned before
	return data

'takes the csv file and split the label from the features'
def splitData(data):
	# Split-out the set in two different arrayste
	array = data.values
	#features array contains only the features of the samples
	features = array[:,0:13]
	#labels array contains only the lables of the samples
	labels = array[:,13]	

	return features,labels

'''
make this class in order to train the model with the same amount of samples of each class, because we have bigger support from class 0
than class1, particularly it is 9 to 1. We made this function in order to make a loop, the equalized data take only a small piece of the existing data, so with this 
loop we are going to take iteratably all the data, but from every iteration we are keeping only the samples who were support
vectors, the samples only the class which we are taking a piece of it's samples'''
''''''
def equalizeClasses(data):
	#take all the samples from the data frame that they have Label value equal to 0 and in the next line equal to 1
	class0 = data.loc[data['Label'] == 0]#class0 and class1 are dataFrames
	class1 = data.loc[data['Label'] == 1]


	#check which class has more samples, by divide them and check if the number is bigger or smaller than 1
	weight = len(class0) // len(class1) #take the results as an integer in order to split the class, using prior knowledge that 
	#class0 has more samples, if it is bigger class0 has more samples and to be exact weight to 1 

	#check division with zero
	if(weight == 0):
		print 'Now the amount of samples in class0 is smaller than half the amount of samples in class1 because we reduce the class0 samples by taking only the support vectors'
		if(len(class0)<(len(class1)/2)):
			#if the amount of samples in class0 is below the amount of half of the samples in class1 terminate the script
			sys.exit()
		else:
			#else, take all the samples from class0
			weight = 1
	else:
		balance = (len(class0) // weight) #this is the number of samples in order to balance our classes

	#the keyword argument frac specifies the fraction of rows to return in the random sample, so fra=1 means, return random all rows
	#we kind of a way shuffle our data in order not to take the same samples in every iteration
	#class0 = class0.sample(frac=1)
	
	#samples array for training taking the balance number of samples for the shuffled dataFrame
	#split the dataFrame based on the weight, so here we are making units of samples in the amount of balance in order
	#to train our model with an iteration procedure
	newData = np.array_split(class0, weight)
	

	#and now combine the new dataFrame from class0 with the class1 to return the balanced dataFrame
	#newData = pd.concat([newClass0, class1])	
	
	#return the new balanced(number of samples from each class) dataFrame
	#return both classes in order to compine them later
	return newData, class1


'''we use this function in order to apply greedy search for finding the parameters that best fit our model. We have to mention
that we start this procedure from a very large field and then we tried to focues to the direction where the results
appear better. For example for the C parameter, the first range was [0.0001, 0.001, 0.01, 0.1, 1, 10 ,100 ,1000], the result was that
the best value was 1000 so then we tried [100, 1000, 10000, 100000] and so on in order to focues to the area which give us
the best results. This function is in comments because we found the best parameters and we dont need to run it in every trial.'''
def paramTuning(features_train, labels_train, nfolds):
	#using the training data and define the number of folds
	#determine the range of the Cs range you want to search
	Cs = [1, 10, 100, 1000, 10000]

	#determine the range of the gammas range you want to search
	gammas = [0.00000001 ,0.00000001 ,0.0000001, 0.000001, 0.00001]

	#make the dictioanry
	param_grid = {'C': Cs, 'gamma': gammas}

	#start the greedy search using all the matching sets from above
	grid_search = GridSearchCV(SVC(kernel='rbf'),param_grid,cv=nfolds)

	#fit your training data
	grid_search.fit(features_train, labels_train)

	#visualize the best couple of parameters
	return grid_search.best_params_



'''Classify Parkinson and Helathy. Building a model which is going to be trained with of given cases and test according to new ones'''
def classifyPHC(data):
	#take the array with the units of samples of class0 divided properly to train the model in a balanced dataset
	data1,class1 = equalizeClasses(data)
	#run this procedure by using all the units
	support_vectors=[]
	for newdata in data1:
		data = pd.concat([newdata, class1])
		features,labels = splitData(data)

		#determine the training and testing size in the range of 1, 1 = 100%
		validation_size = 0.2
	
		#here we are splitting our data based on the validation_size into training and testing data
		features_train, features_validation, labels_train, labels_validation = model_selection.train_test_split(features, labels, 
				test_size=validation_size)

		#we can see the shapes of the array just to check
		print 'feature training array: ',features_train.shape,'and label training array: ',labels_train.shape
		print 'feature testing array: ',features_validation.shape,'and label testing array: ',labels_validation.shape,'\n'


		#take the best couple of parameters from the procedure of greedy search
		#paramTuning(features_train, labels_train, 5)

		#we initialize our model
		svm = SVC(kernel='rbf',C=1000,gamma=1e-07,decision_function_shape='ovo')

		#train our model with the data that we previously precessed
		svm.fit(features_train,labels_train)

		#now test our model with the test data
		predicted_labels = svm.predict(features_validation)
		accuracy = accuracy_score(labels_validation, predicted_labels)
		print 'Classification accuracy: ',accuracy*100,'\n'

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
		#print 'parameters of the model: \n',svm.get_params(),'\n'

		print 'number of samples used as support vectors',len(svm.support_vectors_),'\n'

		#keep the support vectors of every iteration, until the units of samples of the class0 finishes
		support_vectors.append(svm.support_vectors_)

	return support_vectors, class1
	
	
'''make this function because we need to keep only the support vectors from the class with bigger amount of samples in order to train
the model with the support vectors only the class0 and all the samples from the class1, also we need to remove the duplicates because
it is possible that we took duplicates as support vectors, and to delete the support vectors from class1. In this function we are doing the same procedure as previous in order to classify with SVM, but we are using only the samples
from class0 that in our previous iterations they appear themselves as support vectors and all the samples from the class1. We are 
doing this because we have discrepancies in the amount of samples of the two classes. Trying to get better training results.'''
def initSupportVectors(support_vectors, class1):
	flattened_list = []

	#run the list of lists, every list contains on single samples which is support vector
	for x in support_vectors:
		#for every samples in the support vector list
		for y in x:
			flattened_list.append(list(y))


	print 'Amount of support vectors with duplicates: ',len(flattened_list)

	#use this command to remove all the duplicates of the list
	uniqueSupportVectors = [list(l) for l in set(tuple(l) for l in flattened_list)]

	#now we need to remove all the samples that are support vectors but they come from the class1, so we are going to check which
	#samples of our list are in the class1 list as well.

	#convert the dataFrame into a list which has sublists and every lists contains the features, for exampes the sublist[0] contains
	#all the Feature1 of every samples ans so on, so we have to divide it in order to make the real samples


	
	#1825-2102, take every row of the data frame add put it in a list of lists
	class1SamplesInaList = []
	#here we iterate the dataFrame, particularly the rows we define with the range
	for x in range(1825,2103):
		#here we take the specific row
		class1Sample = class1.loc[[x]]	
		#we need to take all the features from this row but the label, because it returns a list of lists we need to join
		#this list of lists into one list which contains one single sample of the class1
		class1SamplesInaList.append(list(itertools.chain.from_iterable(class1Sample.values.T.tolist()[:13])))

	#continue this procedure we are going to check if a vector is in both array, if this is true it means tha this vector
	#is a support vector because it belongs in the support_vector list and it belongs to the class1 because it is in the
	#class1SamplesInaList array so with erase this element from the support vector array, when this procedure is over it means
	#that the elements that remain in the support_vector_array is from class one and support_vectors, this is the goal we define.
	
	#class1SamplesInaList the array which contains as a list every sample of class1 one in a list of lists
	#uniqueSupportVectors contains all the samples that used as support_vectors in every iteration erasing the duplicates
	#becuase the samples from class1 took place in every iteration
	

	#iterate every sample of the class1 in order to check if it exists in the list
	for x in class1SamplesInaList:
		#if it exists, we need to delete it
		if(x in uniqueSupportVectors):
			#remove the specific list from the support_vectors that we are going to use
			uniqueSupportVectors.remove(x)

	print 'Amount of support vectors without duplicates', len(uniqueSupportVectors),'\n'

	#so we know that this array contains the support_vectors which are samples only from class0 and there is no duplicates,
	#so we have to add a last element in every array to declare the label of the samples and it is going to be 0 because
	#we know the class that the samples come from
	for x in uniqueSupportVectors:
		x.append(0)

	#initialize the dataframe that we want to return 
	support_vectors_dataframe = pd.DataFrame(columns=['Feature1','Feature2','Feature3','Feature4','Feature5','Feature6','Feature7','Feature8','Feature9','Feature10','Feature11','Feature12','Feature13','Label'])                             
	for x in range(len(uniqueSupportVectors)):
		#we need to add the columns and the rows of the dataframe so we are going to do it manually
		support_vectors_dataframe.loc[x] = [uniqueSupportVectors[x][y] for y in range(len(uniqueSupportVectors[x]))]

	
	#return the dataframe which contains all the support vectors from all the iteration of training with all the units of samples
	#of only the class0, and now we are ready to train with them and all the samples of the class1
	return pd.concat([support_vectors_dataframe,class1])
	#returns the new data ready to train the model
	#the samples from class0 which were support_vectros and all the samples from class1
	


def main():
	#calculate the time
	import time
	start_time = time.time()

	data = readFile()

	while True:

		#we are making an array in order to keep the support vectors and feed the function with them for the next iteration
		support_vectors, class1 = classifyPHC(data)

		#flat the list of lists into one list
		data = initSupportVectors(support_vectors, class1)

		print 'END OF ITERATION NOW WE ARE TRAINING WITH A NEW REDUCED SET OF SUPPORT VECTORS FROM CLASS 0 \n\n\n\n\n\n\n\n'
	

	time = time.time()-start_time
	print 'time: ',time

main()



