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
from sklearn.neighbors import KNeighborsClassifier

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


def knn_ROC(data):
	X,Y = preparingData(data)
	accuracy=0
	#keep tha rates and the number of neighbours for the plots
	rates=[]
	n_neighbours=[]
	#initiate the lists of true positive etc
	tp=[]
	tn=[]
	fp=[]
	fn=[]
	k=0
	for n in range(5,20):
		n_neighbours.append(n)
		#KNN for variable number of neighbors , check the rates and plot them according to the number of neighbors
		x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20)
		
		
		knn = KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
		for l in range(len(Y)):
			if Y[l]==knn.predict(x_test)[l]:
				#every time the prediction is right
				accuracy+=1
		#calcuate the rate %
		rates.append((accuracy / len(Y))*100)
		accuracy=0
		trueInput = data.ix[data['Gender']==1].iloc[:,0:13]
		trueOutput = data.ix[data['Gender']==1].iloc[:,13]
		#true positive rate	
		tp.append(np.mean(knn.predict(trueInput)==trueOutput))
		#true negative	
		falseInput = data.ix[data['Gender']==0].iloc[:,0:13]
		falseOutput = data.ix[data['Gender']==0].iloc[:,13]
		#true negative rate
		tn.append(np.mean(knn.predict(falseInput)==falseOutput))
		#false positive
		fp.append(1 - tp[k])
		#flase negative
		fn.append(1 - tn[k])
		k+=1
	#print rates



	#visualize
	x = [n for n in range(5,21)]
	y = [n for n in range(80,96,2)]
	#figure 1 : plot the rating based on the neighbours number
	plt.figure(1)
	plt.plot(n_neighbours, rates, marker='o', linestyle='--', color='k', label='Square') 
	plt.title('KNN k-validation')
	plt.xticks(x)
	plt.yticks(y)
	black_patch = mpatches.Patch(color='k', label='Accuracy')
	plt.legend(handles=[black_patch])
	plt.ylabel('100%', fontsize=10)
	plt.xlabel('K (neighbours)', fontsize=8)
	
	#figure 2: plot the true positive etc based on the neighbours number to compare the missclassification 
	#and how important they are
	plt.figure(2)
	red_patch = mpatches.Patch(color='red', label='True positive')
	blue_patch = mpatches.Patch(color='blue', label='True negative')
	green_patch = mpatches.Patch(color='green', label='False positive')
	magenta_patch = mpatches.Patch(color='magenta', label='False negative')
	plt.legend(handles=[red_patch,blue_patch,green_patch,magenta_patch])
	plt.title('KNN classify-validation')
	plt.xticks(x)
	plt.ylabel('Rating', fontsize=10)
	plt.xlabel('K (neighbours)', fontsize=8)
	plt.plot(n_neighbours, tp, marker='d', linestyle='--', color='r', label='Square') 
	plt.plot(n_neighbours, tn, marker='d', linestyle='--', color='b', label='Square') 
	plt.plot(n_neighbours, fp, marker='d', linestyle='--', color='g', label='Square') 
	plt.plot(n_neighbours, fn, marker='d', linestyle='--', color='m', label='Square') 
	#plot
	plt.show()


	#cross-validation
	kfold = model_selection.KFold(n_splits=10,random_state=7,shuffle=True)
	x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20)
	optimalNeighboursNumber = 5+rates.index(max(rates))
	modelCV = KNeighborsClassifier(n_neighbors=optimalNeighboursNumber)
	scoring = 'accuracy'
	results = model_selection.cross_val_score(modelCV,x_train,y_train,cv=kfold,scoring=scoring)
	print '10-fold cross validation average accuracy: ',results.mean()

	#confusion matrix
	knn = KNeighborsClassifier(n_neighbors=optimalNeighboursNumber)
	knn.fit(x_train,y_train)
	y_pred = knn.predict(x_test)
	#print y_pred
	#print y_test
	print 'KNN classifier accuracy: ', knn.score(x_test,y_test)
	confusionMatrix = confusion_matrix(y_test,y_pred)
	print 'Confusion matrix: '
	print confusionMatrix
	print 'We had ',confusionMatrix[0][0] + confusionMatrix[1][1], 'correct predictions'
	print 'And ',confusionMatrix[1][0] + confusionMatrix[0][1],'incorrect prediction'
	print ''

	print(classification_report(y_test,y_pred))


	
	
	
	
	
	
def main():
	data = readFeaturesFile()
	knn_ROC(data)

main()

