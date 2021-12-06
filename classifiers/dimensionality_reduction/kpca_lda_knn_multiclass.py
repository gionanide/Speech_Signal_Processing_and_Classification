#!usr/bin/python
from __future__ import division
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC # support vectors for classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import IncrementalPCA, PCA, KernelPCA
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


'''this function takes as an input the path of a file with features and labels and returns the content of this file as a csv format in
the form feature1.........feature13,Label'''
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


'''we use this function in order to apply greedy search for finding the parameters that best fit our model. We have to mention
that we start this procedure from a very large field and then we tried to focues to the direction where the results
appear better. For example for the C parameter, the first range was [0.0001, 0.001, 0.01, 0.1, 1, 10 ,100 ,1000], the result was that
the best value was 1000 so then we tried [100, 1000, 10000, 100000] and so on in order to focues to the area which give us
the best results. This function is in comments because we found the best parameters and we dont need to run it in every trial.'''
def paramTuning(features_train, labels_train, nfolds):
	#using the training data and define the number of folds
	#determine the range of the Cs range you want to search
	Cs = [0.001 ,0.01 ,0.1 ,1 , 10, 100, 1000, 10000]

	#determine the range of the gammas range you want to search
	gammas = [0.00000001 ,0.00000001 ,0.0000001, 0.000001, 0.00001 , 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

	#make the dictioanry
	param_grid = {'C': Cs, 'gamma': gammas}

	#start the greedy search using all the matching sets from above
	grid_search = GridSearchCV(SVC(kernel='rbf'),param_grid,cv=nfolds)

	#fit your training data
	grid_search.fit(features_train, labels_train)

	#visualize the best couple of parameters
	print grid_search.best_params_


	
	





'''Building a model which is going to be trained with of given cases and test according to new ones'''
def classifyPHC():
	data = readFile()
	#data = equalizeClasses(data)
	features,labels = splitData(data)
	
	#determine the training and testing size in the range of 1, 1 = 100%
	validation_size = 0.2
	
	#here we are splitting our data based on the validation_size into training and testing data
	features_train, features_validation, labels_train, labels_validation = model_selection.train_test_split(features, labels, 
			test_size=validation_size)

	#normalize data in the range [-1,1]
	scaler = MinMaxScaler(feature_range=(-1, 1))
	#fit only th training data in order to find the margin and then test to data without normalize them
	scaler.fit(features_train)

	features_train_scalar = scaler.transform(features_train)

	#trnasform the validation features without fitting them
	features_validation_scalar = scaler.transform(features_validation)


	#determine the pca, and determine the dimension you want to end up
	pca = KernelPCA(n_components=5,kernel='rbf',fit_inverse_transform=True)

	#fit only the features train
	pca.fit(features_train_scalar)

	#dimensionality reduction of features train
	features_train_pca = pca.transform(features_train_scalar)

	#dimensionality reduction of fatures validation
	features_validation_pca = pca.transform(features_validation_scalar)

	#reconstruct data training error
	reconstruct_data = pca.inverse_transform(features_train_pca)
	
	
	error_percentage = (sum(sum(error_matrix))/(len(features_train_scalar)*len(features_train_scalar[0])))*100

	#len(features_train_scalar) = len(reconstruct_data) = 89
	#len(features_train_scalar[0]) = len(reconstruct_data[0]) = 13

	#len(error_matrix) = 89, which means for all the samples
	#len(error_matrix[0]) = 13, for every feature of every sample
	#we take the sum and we conlcude in an array which has the sum for every feature (error)
	#so we take the sum again and we divide it with the 89 samples * 13 features
	print 'Information loss of KernelPCA:',error_percentage,'% \n'


	lda = LinearDiscriminantAnalysis()

	lda.fit(features_train_pca,labels_train)

	features_train_pca = lda.transform(features_train_pca)

	features_validation_pca = lda.transform(features_validation_pca)

	

	#we can see the shapes of the array just to check
	print 'feature training array: ',features_train_pca.shape,'and label training array: ',labels_train.shape
	print 'feature testing array: ',features_validation_pca.shape,'and label testing array: ',labels_validation.shape,'\n'

	#take the best couple of parameters from the procedure of greedy search
	#paramTuning(features_train, labels_train, 5)

	#we initialize our model
	#svm = SVC(kernel='rbf',C=10,gamma=0.0001,decision_function_shape='ovo')
	svm = KNeighborsClassifier(n_neighbors=3)

	#train our model with the data that we previously precessed
	svm.fit(features_train_pca,labels_train)


	#now test our model with the test data
	predicted_labels = svm.predict(features_validation_pca)
	accuracy = accuracy_score(labels_validation, predicted_labels)
	print 'Classification accuracy: ',accuracy*100,'\n'

	#see the accuracy in training procedure
	predicted_labels_train = svm.predict(features_train_pca)
	accuracy_train = accuracy_score(labels_train, predicted_labels_train)
	print 'Training accuracy: ',accuracy_train*100,'\n'

	#confusion matrix to illustrate the faulty classification of each class
	conf_matrix = confusion_matrix(labels_validation, predicted_labels)
	print 'Confusion matrix: \n',conf_matrix,'\n'
	print 'Support    class 0   class 1    class2:'
	#calculate the support of each class  
	print '          ',conf_matrix[0][0]+conf_matrix[0][1]+conf_matrix[0][2],'       ',conf_matrix[1][0]+conf_matrix[1][1]+conf_matrix[1][2],'        ',conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[2][2],'\n'

	#calculate the accuracy of each class
	edema = (conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]+conf_matrix[0][2]))*100
	paralysis = (conf_matrix[1][1]/(conf_matrix[1][0]+conf_matrix[1][1]+conf_matrix[1][2]))*100
	normal = (conf_matrix[2][2]/(conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[2][2]))*100

	#see the inside details of the classification
	print 'For class 0 edema cases:',conf_matrix[0][0],'classified correctly and',conf_matrix[0][1]+conf_matrix[0][2],'missclassified,',edema,'accuracy \n'
	print 'For class 1 paralysis cases:',conf_matrix[1][1],'classified correctly and',conf_matrix[1][0]+conf_matrix[1][2],'missclassified,',paralysis,'accuracy\n'
	print 'For class 0 normal cases:',conf_matrix[2][2],'classified correctly and',conf_matrix[2][0]+conf_matrix[2][1],'missclassified,',normal,'accuracy \n'

	#try 5-fold cross validation
	scores = cross_val_score(svm, features_train_pca, labels_train, cv=5)
	print 'cross validation scores for 5-fold',scores,'\n'
	print 'parameters of the model: \n',svm.get_params(),'\n'

	
	#PLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOTS
	
	#sns.pairplot(data, hue='Label')
	#plt.savefig('data_visualization.png')
	#plt.title('Data visualization')
	#plt.show()

	#print features_train.shape
	#print len(features_train[:,0])
	#print len(features_train[:,1])
	#print len(labels_train)

	#plot the training features before the kpca and the lda procedure
	#kpca_lda = pd.DataFrame({'Feature1': features_train[: ,0], 'Feature2': features_train[: ,1],'Feature3': features_train[: ,2], 'Feature4': features_train[: ,3],'Feature5': features_train[: ,4],'Feature6': features_train[: ,5],'Feature7': features_train[: ,6],'Feature8': features_train[: ,7],'Feature9': features_train[: ,8],'Feature10': features_train[: ,9],'Feature11': features_train[: ,10],'Feature12': features_train[: ,11],'Feature13': features_train[: ,12],'Label': labels_train})
	#'Feature10','Feature11','Feature12','Feature13','Label'])
	#sns.pairplot(kpca_lda, hue='Label')
	#plt.savefig('training_features.png')
	#plt.show()

	#plot the validation features before the kpca and the lda procedure
	#kpca_lda = pd.DataFrame({'Feature1': features_validation[: ,0], 'Feature2': features_validation[: ,1],'Feature3': features_validation[: ,2], 'Feature4': features_validation[: ,3],'Feature5': features_validation[: ,4],'Feature6': features_validation[: ,5],'Feature7': features_validation[: ,6],'Feature8': features_validation[: ,7],'Feature9': features_validation[: ,8],'Feature10': features_validation[: ,9],'Feature11': features_validation[: ,10],'Feature12': features_validation[: ,11],'Feature13': features_validation[: ,12],'Label': labels_validation})
	#'Feature10','Feature11','Feature12','Feature13','Label'])
	#sns.pairplot(kpca_lda, hue='Label')
	#plt.savefig('validation_features.png')
	#plt.show()

	#plot the training features after the kpca and the lda procedure
	#kpca_lda = pd.DataFrame({'Feature1': features_train_pca[: ,0], 'Feature2': features_validation_pca[: ,1],'Label': labels_validation})
	#sns.pairplot(kpca_lda, hue='Label')
	#plt.savefig('kpca_lda_knn_validationset.png')
	#plt.show()

	#plot the validation features after the kpca and the lda procedure
	#kpca_lda = pd.DataFrame({'Feature1': features_validation_pca[: ,0], 'Feature2': features_validation_pca[: ,1],'Label': labels_validation})
	#sns.pairplot(kpca_lda, hue='Label')
	#plt.savefig('kpca_lda_knn_validationset.png')
	#plt.show()

	#print 'number of samples used as support vectors',len(svm.support_vectors_)


def main():
	import time
	start_time = time.time()
	classifyPHC()
	time = time.time()-start_time
	print 'time: ',time

main()

