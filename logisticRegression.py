#!usr/bin/python
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sk
from sklearn.feature_selection import RFE
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def readFeaturesFile():
	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
'Feature10','Feature11','Feature12','Feature13','Gender']
	data = pd.read_csv("mfcc_featuresLR.txt",names=names )
	#the outcome is a list of lists containing the samples with the following format
	#[charachteristic,feature1,feature2.......,feature13]
	#characheristic based on what we want for classification , can be (male , female) , also can be (normal-female,edema-female)
	#in general characheristic is the target value .
	return data

def preparingData(data):
	# Split-out validation dataset
	array = data.values
	#input
	X = array[:,0:13]
	#target 
	Y = array[:,13]
	return X,Y
	
def LR_ROC(data):
	#we initialize the random number generator to a const value
	#this is important if we want to ensure that the results
	#we can achieve from this model can be achieved again precisely
	#Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.	
	mean = np.mean(data,axis=0)
	std = np.std(data,axis=0)
	#print 'Mean: \n',mean
	#print 'Standar deviation: \n',std
	X,Y = preparingData(data)
	x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20)
	# convert integers to dummy variables (i.e. one hot encoded)
	lr = LogisticRegression(class_weight='balanced')
	lr.fit(x_train,y_train)
	#The score function of sklearn can quickly assess the model performance
	#due to class imbalance , we nned to evaluate the model performance
	#on every class. Which means to find when we classify people from the first team wrong


	#feature selection RFE is based on the idea to repeatedly construct a model and choose either the best
	#or worst performing feature, setting the feature aside and then repeating the process with the rest of the 
	#features. This process is applied until all features in the dataset are exhausted. The goal of RFE is to select
	# features by recursively considering smaller and smaller sets of features
	rfe = RFE(lr,13)
	rfe = rfe.fit(x_train,y_train)
	#print rfe.support_

	#An index that selects the retained features from a feature vector. If indices is False, this is a boolean array of shape 
	#[# input features], in which an element is True iff its corresponding feature is selected for retention

	#print rfe.ranking_

	#so we have to take all the features

	#model fitting
	
	#predicting the test set results and calculating the accuracy
	y_pred  = lr.predict(x_test)
	print 'Accuracy of logistic regression classifier on the test set: ', lr.score(x_test,y_test)

	#cross validation
	kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=7)
	modelCV = LogisticRegression()
	scoring = 'accuracy'
	results = model_selection.cross_val_score(modelCV, x_train,y_train,cv=kfold,scoring=scoring)
	print '10-fold cross validation average accuracy: ', results.mean()

	#confusion matrix
	confusionMatrix = confusion_matrix(y_test,y_pred)
	print 'Confusion matrix: '
	print confusionMatrix
	print 'We had ',confusionMatrix[0][0] + confusionMatrix[1][1], 'correct predictions'
	print 'And ',confusionMatrix[1][0] + confusionMatrix[0][1],'incorrect prediction'
	print ''

	#The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
	#The recall is intuitively the ability of the classifier to find all the positive samples.
	#The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
	#The support is the number of occurrences of each class in y_test.

	#classification report
	print(classification_report(y_test,y_pred))

	#roc curve
	logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
	fpr , tpr , thresholds = roc_curve(y_test,lr.predict_proba(x_test)[:,1])
	
	#AUC is a measure of the overall performance of a diagnostic test and is 
	#interpreted as the average value of sensitivity for all possible values of specificity
	
	fprtpr = np.hstack((fpr[:,np.newaxis],tpr[:,np.newaxis]))

	hull = ConvexHull(fprtpr)

	hull_indices = np.unique(hull.simplices.flat)
	hull_points = fprtpr[hull_indices,:]
	hull_points_y=[]
	hull_points_x=[]
	for x in range(len(hull_points)):
		coordinates =  np.split(hull_points[x],2)
		hull_points_y.append(coordinates[0])
		hull_points_x.append(coordinates[1])
		
		
	plt.figure(1)
	plt.title('ROC curve smooth')
	plt.scatter(hull_points_y,hull_points_x)
	area_under =  metrics.auc(hull_points_y,hull_points_x)
	plt.plot(hull_points_y,hull_points_x,label='Area under the curve = %0.2f' %area_under)
	plt.legend(loc='lower right')
	
	

	plt.figure(2)
	plt.scatter(fpr,tpr)
	plt.title('Convex Hull')
	#plt.plot(fpr[hull.vertices],tpr[hull.vertices])
	plt.plot(fprtpr[:,0], fprtpr[:,1], 'o')
	for simplex in hull.simplices:
	     plt.plot(fprtpr[simplex, 0], fprtpr[simplex, 1],'r--',lw=2)
	
	plt.figure(3)
	plt.plot(fpr,tpr,label='Logistic Regression (area = %0.2f)' %logit_roc_auc)
	plt.plot([0,1],[0,1],'r--')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc='lower right')
	plt.show()

	#It generally means that your model can only provide discrete predictions, rather than a continous score. This can often be
	# remedied by adding more samples to your dataset, having more continous features in the model, more features in general or using
	# a model specification that provides a continous prediction output. The reason why it occurs in a decision tree is that you 
	#often do binary splits; this is efficient computationally, but only gives 2^n groupings. Unless your n number of splits are very 
	#large, you'll only have 16/32/64/128 groups, whereas if you used an algorithm such as logistic regression and used continous 
	#variables, your prediction would fall in the continous range between 0 and 1. I'm not familiar with the type of data you listed,
	# but I suspect you have a lot of categorical data.It's not necessarily a problem to have a ROC that is discrete rather than 
	#smooth, it really depends on your goals for the model (descriptive vs prescriptive), as well as how well your model fits on 
	#out-of-sample datasets. Many of the problems I've solved in my career just needed a Yes/No line drawn (such as email this 
	#person/don't email), so having a continous and smooth prediction along the range of inputs wasn't necessary.
	



def main():
	data = readFeaturesFile()
	LR_ROC(data)

main()
