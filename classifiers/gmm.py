#!usr/bin/python
from __future__ import division
import pickle
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt


#lpc durbin-levinson


#gmm training python
def GaussianMixtureModel_only_for_testing(data):
	#A GMM attempts to find a mixture of multidimensional Gaussian probability distributions that best model any input dataset.
	#In the simplest case , GMM's can be used for finding clusters in the same manner as k-means.
	X,Y = preparingData(data)
	#taking only the first two features
	#Y = target variable
	gmm =  GaussianMixture(n_components=2)
	#Estimate model parameters with the EM algorithm.
	gmm.fit(X)
	labels = gmm.predict(X)
	print labels

	plt.figure(1)
	#because of the probabilistic approach of GMM's it is possible to find a probabilistic cluster assignments.
	#porbs : is a matrix [samples, nClusters] which contains the probability of any point belongs to the given cluster
	probs = gmm.predict_proba(X).round(3)
	#which measures the probability that any point belongs to the given cluster:
	print probs

	#we can visualize this uncertainty . For instance let's make the size of each point proortional to the certainty
	#of its prediction. We are going to point the points at the boundaries between clusters.
	size = 50 * probs.max(1) ** 2  # square emphasizes differences
		
	#the weights of each mixture components
	weights = gmm.weights_
	#the mean of each mixture component
	means = gmm.means_
	#the covariance of each mixture component
	covars = gmm.covariances_

	print 'weights: ',weights
	print 'means: ', means

	print gmm.score(X)
	#Predict the labels for the data samples in X using trained model.
	print labels[0]
	print Y
	print Y[0]
	

	#plots 
	plt.scatter(X[:,5],X[:,6],c=labels,s=40,cmap='viridis')
	plt.show()

def readFeaturesFile(gender):
	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
'Feature10','Feature11','Feature12','Feature13','Gender']
	
	#check the gender
	if(int(gender)==1):
		data = pd.read_csv("gmm_female.txt",names=names )
	elif(int(gender)==0):
		data = pd.read_csv("gmm_male.txt",names=names )
	else:
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

def GaussianMixtureModel(data,gender):
	#A GMM attempts to find a mixture of multidimensional Gaussian probability distributions that best model any input dataset.
	#In the simplest case , GMM's can be used for finding clusters in the same manner as k-means.
	X,Y = preparingData(data)
	#print data.head(n=5)
	
	x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20)

	#takes only the first feature to redefine the problem as 1-D problem
	#dataFeature1 =  data.as_matrix(columns=data.columns[0:1])
	#plot histogram
	#sns.distplot(dataFeature1,bins=20,kde=False)
	#plt.show()

	
	
	#Y = target variable
	gmm =  GaussianMixture(n_components=8,max_iter=200,covariance_type='diag',n_init=3)
	gmm.fit(x_train)
	
		

	#save the model to disk
	filename = 'finalizedModel_'+gender+'.gmm'
	pickle.dump(gmm,open(filename,'wb'))
	print 'Model saved in path: /home/gionanide/'+filename


	return X
	#load the model from disk
	'''loadedModel = pickle.load(open(filename,'rb'))
	result = loadedModel.score(X)
	print result'''

def testModels(data,threshold_input,x_test,y_test):
	gmmFiles = ['/home/gionanide/Theses_2017-2018_2519/finalizedModel_0.gmm','/home/gionanide/Theses_2017-2018_2519/finalizedModel_1.gmm']
	models = [pickle.load(open(filename,'r')) for filename in gmmFiles]
	log_likelihood = np.zeros(len(models))
	genders = ['male','female']
	assessModel = []
	prediction = []
	features = X
	for i in range(len(models)):
		gmm = models[i]
		scores = np.array(gmm.score(x_test))
		#first take for the male model all the log likelihoods and then the same procedure for the female model
		assessModel.append(gmm.score_samples(x_test))
		log_likelihood[i] = scores.sum()
	#higher the value it is, the more likely your model fits the model
	for x in range(len(assessModel[0])):
		#the division is gmm(Malemodel) / gmm(Femalemodel) if the result is > 1 then the example is male
		
		
		#if the prediction for male in negative and the prediction for female positive we dont have to check
		#because the difference is obvious and we are pretty sure that it is female
		if(assessModel[0][x] < 0 and assessModel[1][x] > 0):
			# x / y and x is < 0 , so we have to classify this as female
			# we have to be sure that the prediction will be above the threshold 
			prediction.append(float(threshold_input) + 1)
			
		#same as above , we need to be sure that the prediction is below the threshold (male) because we are pretty
		#sure from the model's outcome that this sample is female
		elif(assessModel[0][x] > 0 and assessModel[1][x] < 0):
			prediction.append(float(threshold_input) - 1)
		else:
			prediction.append( abs(( assessModel[0][x] / assessModel[1][x] )) ) 
	

	#take an array with the predictions and check if they are true(correct classification) or false(wrong classification)
	assessment=[]
	true_negative=0
	true_positive=0
	false_positive=0
	false_negative=0
	for x in range(len(prediction)):#reject option
		if(prediction[x]<1.019 and prediction[x]>1.012):
			print prediction[x] , ' can not decide'
		elif(prediction[x]<float(threshold_input)):#the model predict male and we check if it is indeed male
			#print prediction[x], ( Y[x] == 0 )
			decision = (y_test[x] == 0)
			assessment.append(decision)
			if(decision):
				true_negative+=1
			else:
				false_negative+=1
		else:
			decision1 = (Y[x] == 1)#the model predict female and we check if it is indeed female
			#print prediction[x], ( Y[x] == 1 )
			assessment.append(decision1)
			if(decision1):
				true_positive+=1
			else:
				false_positive+=1
	for x in range(len(assessment)):
		if(assessment[x]==False):
			print prediction[x],assessment[x]
			print assessModel[0][x],assessModel[1][x]
	#construct confusion matrix
	confusion_matrix= [[0 for x in range(2)] for y in range(2)]
	confusion_matrix[0][0]= true_negative
	confusion_matrix[0][1]= false_positive
	confusion_matrix[1][0]= false_negative
	confusion_matrix[1][1]= true_positive

	
	#sensitivity/ true positive rate
	tpr = (true_positive)/(true_positive + false_negative)
	#fall-out/ false positive rate
	fpr = (false_positive)/(false_positive + true_negative)

	#ROC curve with error and reject option
	
				
	winner = np.argmax(log_likelihood)
	print ''
	print "\tdetected as - ", genders[winner],"\n\tscores:male ",log_likelihood[0],",female ", log_likelihood[1],"\n"

	male_support = confusion_matrix[0][0] + confusion_matrix[0][1]
	female_support = confusion_matrix[1][0] + confusion_matrix[1][1]

	print 'Confusion matrix:       support: '
	print '               ',confusion_matrix[0],'    0.0: ',male_support
	print '               ',confusion_matrix[1],'    1.0: ',female_support
	print ''
	print 'Accuracy: ', ( assessment.count(True) / len(assessment) ) * 100 
	
	
	#with reject option
	tpr_plot = np.array([0,tpr,1])
	fpr_plot = np.array([0,fpr,1])
	#without reject option
	tpr_plot_wr = np.array([0,0.983606557377,1])
	fpr_plot_wr = np.array([0,0.0588235294118,1])

	#area under the curve with the reject option
	area_under_plot =  metrics.auc(fpr_plot,tpr_plot)
	#area under the curve without the reject option
	area_under_wr = metrics.auc(fpr_plot_wr,tpr_plot_wr)
	
	#with reject option
	thresholds = np.array([0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10,1.11,1.12,1.13])
	accuracy_plot = np.array([89.09090909,90.0,90.909090,92.727272,93.636363,94.545454,94.545454,94.545454,97.272727,97.272727,98.18181818,98.18181818,98.18181818,97.272727,96.363636,95.45454545,95.45454545,92.727272,91.818181,90.909090,90.909090,90.909090,90.909090,90.0,90.0,88.181818,88.181818,87.272727])
	
	#without reject option
	accuracy_wr = np.array([87.5,88.392875,89.285714,91.071428,91.9642785,92.8571428,92.8571428,93.75,95.535714,95.535714,96.42785,96.42785,96.42785,95.535714,94.64285,93.75,95.535714,92.8571428,91.9642785,91.071428,91.071428,91.071428,91.071428,90.178571,90.178571,88.392857,88.1818181,87.5])
	


	plt.figure(1)
	green_patch = mpatches.Patch(color='green', label='With reject option (area = %0.2f)' %area_under_plot)
	blue_patch = mpatches.Patch(color='blue', label='Without reject option (area = %0.2f)' %area_under_wr)
	plt.legend(handles=[green_patch,blue_patch])
	plt.plot(fpr_plot,tpr_plot,marker='d',linestyle='--',color='g')
	plt.plot(fpr_plot_wr,tpr_plot_wr,marker='d',linestyle='--',color='b')
	plt.plot([0,1],[0,1],'r--')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc='lower right')
	
	
	
	plt.figure(2)
	plt.plot(thresholds,accuracy_plot,marker='o',linestyle='--')
	plt.xlabel('thresholds')
	plt.ylabel('accuracy')
	plt.title('Optimal threshold(with reject option)')

	plt.figure(3)
	plt.plot(thresholds,accuracy_wr,marker='o',linestyle='--')
	plt.xlabel('thresholds')
	plt.ylabel('accuracy')
	plt.title('Optimal threshold(without reject option)')
	plt.show()



def determineComponents(data):
	X,Y = preparingData(data)
	n_components = np.arange(1,10)
	bic = np.zeros(n_components.shape)

	for i,n in enumerate(n_components):
		#fit gmm to data for each value of components
		gmm = GaussianMixture(n_components=n,max_iter=200, covariance_type='diag' ,n_init=3)
		gmm.fit(X)
		#store BIC scores
		bic[i] = gmm.bic(X)

	#Therefore, Bayesian Information Criteria (BIC) is introduced as a cost function composing of 2 terms; 
	#1) minus of log-likelihood and 2) model complexity. Please see my old post. You will see that BIC prefers model 
	#that gives good result while the complexity remains small. In other words, the model whose BIC is smallest is the winner
	#plot the results
	plt.plot(bic)
	plt.show()

def main():
	gender = raw_input('Choose the gender for training press 1(Female) or 0(Male) and any other number for testing: ')
	data = readFeaturesFile(gender)
	#determineComponents(data)
	#GaussianMixtureModel(data,gender)
	threshold = raw_input('Threshold: ')
	testModels(data,threshold,x_test,y_test)
	#GaussianMixtureModel_only_for_testing(data)

main()
