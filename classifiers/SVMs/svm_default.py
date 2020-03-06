#!usr/bin/python
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC # support vectors for classification
from sklearn.metrics import accuracy_score, confusion_matrix


'this function takes as an input the path of a file with features and labels and returns the content of this file as a csv format in'
'the form feature1.........feature13,Label'
def readFile():
	#make the format of the csv file. Our format is a vector with 13 features and a label which show the condition of the
	#sample hc/pc : helathy case, parkinson case
  	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
	'Feature10','Feature11','Feature12','Feature13','Label']

	#path to read the samples, samples consist from healthy subjects and subject suffering from Parkinson's desease.
	path = 'PATH_TO_SAMPLES.txt'
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


'Classify Parkinson and Helathy. Building a model which is going to be trained with of given cases and test according to new ones'
def classifyPHC():
	data = readFile()
	features,labels = splitData(data)
	
	#determine the training and testing size in the range of 1, 1 = 100%
	validation_size = 0.2
	
	#here we are splitting our data based on the validation_size into training and testing data
	features_train, features_validation, labels_train, labels_validation = model_selection.train_test_split(features, labels, 
			test_size=validation_size)

	#we can see the shapes of the array just to check
	print 'feature training array: ',features_train.shape,'and label training array: ',labels_train.shape
	print 'feature testing array: ',features_validation.shape,'and label testing array: ',labels_validation.shape,'\n'

	#we initialize our model
	svm = SVC(kernel='sigmoid',C=0.8)

	#train our model with the data that we previously precessed
	svm.fit(features_train,labels_train)

	#now test our model with the test data
	predicted_labels = svm.predict(features_validation)
	accuracy = accuracy_score(labels_validation, predicted_labels)
	print 'Classification accuracy: ',accuracy,'\n'

	#confusion matrix to illustrate the faulty classification of each class
	conf_matrix = confusion_matrix(labels_validation, predicted_labels)
	print 'Confusion matrix: \n',conf_matrix,'\n'
	print 'Support    class 0   class 1:'
	#calculate the support of each class
	print '          ',conf_matrix[0][0]+conf_matrix[0][1],'     ',conf_matrix[1][0]+conf_matrix[1][1],'\n'

	#see the inside details of the classification
	print 'For class 0 healthy cases:',conf_matrix[0][0],'classified correctly and',conf_matrix[0][1],'missclassified \n'
	print 'For class 1 parkinson cases:',conf_matrix[1][1],'classified correctly and',conf_matrix[1][0],'missclassified \n'

	

classifyPHC()

