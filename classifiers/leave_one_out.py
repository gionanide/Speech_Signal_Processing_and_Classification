#!usr/bin/python
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import elm
import scipy as sk
import numpy as np
from sklearn.utils.testing import assert_greater, assert_raise_message,assert_allclose
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#GMLVQ , www.cs.rug.nl/~biehl
#forest, DTC

def readFeaturesFile():
	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
'Feature10','Feature11','Feature12','Feature13','Gender']
	data = pd.read_csv("PATH_TO_SAMPLES.txt",names=names )
	#the outcome is a list of lists containing the samples with the following format
	#[charachteristic,feature1,feature2.......,feature13]
	#characheristic based on what we want for classification , can be (male , female) , also can be (normal-female,edema-female)
	#in general characheristic is the target value .


	training(data)	
	#visualizeData(data)
	

#Implementation in order to measure algorithms' execution time
def compare_Algorithms(model,X_train,Y_train,kfold,scoring):
	#return the validation score for every algorithm
	return model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)




def preparingData(data):
	# Split-out validation dataset
	array = data.values
	#input
	X = array[:,0:13]
	#target 
	Y = array[:,13]
	return X,Y

def training(data):
	models = []
	#ML algorithms
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	names = []
	resultsLOO = []
	results = []
	meanResultsLOO=np.zeros(len(models))
	meanTimes=np.zeros(len(models))
	meanResults=np.zeros(len(models))
	n = int(raw_input('How many times you want to run the procedure? '))
	for x in range(n):
		X,Y = preparingData(data)
		#splitting into training and testing 
		validation_size = 0.20
		#test_size is the splitting between the training and testing data , for example 20% testing and 80% training in our case
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
		scoring = 'accuracy'
		#testing
		x=0
		print'\n'	
		print('Algorithm:     accuracy(k-fold)     accuracy(leave-one-out)        time')
		for name, model in models:
			#function for k-flods cross validation , k-folds = n_splits 
			#then i can determine if i want to shuffle my data every time before the validation True
			#and random_state , is the seed for the random number generator
			kfold = model_selection.KFold(n_splits=10, shuffle=True)
			#leave-one-out validation with no arguments
			leaveOneOut = model_selection.LeaveOneOut()
			import time
			#initialize the time
			start_time = time.time()
		 	cv_results = compare_Algorithms(model,X_train,Y_train,kfold,scoring)
			#compare algorithms with leave-one-out validation
			cv_resultsLOO = compare_Algorithms(model,X_train,Y_train,leaveOneOut,scoring)
			#count the time around the command
			time = time.time() - start_time
			#append the validation of every algorithm
			resultsLOO.append(cv_resultsLOO)
		    	results.append(cv_results)
		    	names.append(name)
			#visualize results
			msg = "%s:              %f                  %f             %f" % (name, cv_results.mean(),cv_resultsLOO.mean(),time)
			meanTimes[x]+=time
			meanResultsLOO[x]+=cv_resultsLOO.mean()
			meanResults[x]+=cv_results.mean()
			if(x<len(models)-1):
				x+=1
			else:
				x=0
			print msg
	#divide the outcomes with the number of iterations to take the mean of all the iterations
	meanTimes =  (meanTimes / n)
	meanResults = (meanResults / n)
	meanResultsLOO = (meanResultsLOO / n)
	print'\n\n'
	print('Mean of every iteration:')
	print('Algorithm:     accuracy(k=10-fold)     accuracy(leave-one-out)        time')
	for x in range(len(models)):
		msg = "%s:              %f                  %f             %f" % (names[x],meanResults[x],meanResultsLOO[x],meanTimes[x])
		print(msg)

def visualizeData(data):
	#Checking my data
	#data shape
	print(data.shape)
	#print the 20 first samples
	print(data.head(20))
	#This includes the count, mean, the min and max values as well as some percentiles
	print(data.describe())
	#class distribution
	print(data.groupby('Gender').size())
	#Visualize my data
	# box and whisker plots
	data.plot(kind='box', subplots=True, sharex=False, sharey=False)
	# box and whisker plots
	data.plot(kind='box', subplots=True, sharex=False, sharey=False)
	# scatter plot matrix
	scatter_matrix(data)
	plt.show()
	
def main():
	readFeaturesFile()

main()
