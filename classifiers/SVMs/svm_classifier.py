#!usr/bin/python
from __future__ import division
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC # support vectors for classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV


'''this function takes as an input the path of a file with features and labels and returns the content of this file as a csv format in
the form feature1.........feature13,Label'''
def readFile():
	#make the format of the csv file. Our format is a vector with 13 features and a label which show the condition of the
	#sample hc/pc : helathy case, parkinson case
  	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
	'Feature10','Feature11','Feature12','Feature13','Label']

	#path to read the samples, samples consist from healthy subjects and subject suffering from Parkinson's desease.
	path = '/home/gionanide/theses_cotropoulos_PythonCode/features/parkinson_healthy/mfcc_parkinson_healthy.txt'
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
	class0 = class0.sample(frac=1)
	
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
	Cs = [1000, 10000, 10000, 1000000]

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
def classifyPHC():
	data = readFile()
	data = equalizeClasses(data)
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
	#C_best, gamma_best = paramTuning(features_train, labels_train, 5)

	#we initialize our model
	svm = SVC(kernel='rbf',C=1000,gamma=1e-07)


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
	print 'For class 0 healthy cases:',conf_matrix[0][0],'classified correctly and',conf_matrix[0][1],'missclassified,',hC,'accuracy \n'
	print 'For class 1 parkinson cases:',conf_matrix[1][1],'classified correctly and',conf_matrix[1][0],'missclassified,',pC,'accuracy\n'

	#try 5-fold cross validation
	scores = cross_val_score(svm, features_train, labels_train, cv=5)
	print 'cross validation scores for 5-fold',scores,'\n'
	print 'parameters of the model: \n',svm.get_params(),'\n'

	print 'number of samples used as support vectors',len(svm.support_vectors_)

	

classifyPHC()

