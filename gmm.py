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

	#takes only the first feature to redefine the problem as 1-D problem
	#dataFeature1 =  data.as_matrix(columns=data.columns[0:1])
	#plot histogram
	#sns.distplot(dataFeature1,bins=20,kde=False)
	#plt.show()

	
	
	#Y = target variable
	gmm =  GaussianMixture(n_components=8,max_iter=200,covariance_type='diag',n_init=3)
	gmm.fit(X)
	
		

	#save the model to disk
	filename = 'finalizedModel_'+gender+'.gmm'
	pickle.dump(gmm,open(filename,'wb'))
	print 'Model saved in path: /home/gionanide/'+filename


	return X
	#load the model from disk
	'''loadedModel = pickle.load(open(filename,'rb'))
	result = loadedModel.score(X)
	print result'''

def testModels(data,threshold_input):
	gmmFiles = ['/home/gionanide/Theses_2017-2018_2519/finalizedModel_0.gmm','/home/gionanide/Theses_2017-2018_2519/finalizedModel_1.gmm']
	models = [pickle.load(open(filename,'r')) for filename in gmmFiles]
	log_likelihood = np.zeros(len(models))
	genders = ['male','female']
	X,Y = preparingData(data)
	assessModel = []
	prediction = []
	features = X
	for i in range(len(models)):
		gmm = models[i]
		scores = np.array(gmm.score(features))
		#first take for the male model all the log likelihoods and then the same procedure for the female model
		assessModel.append(gmm.score_samples(features))
		log_likelihood[i] = scores.sum()
	#higher the value it is, the more likely your model fits the model
	for x in range(len(assessModel[0])):
		#the division is gmm(Malemodel) / gmm(Femalemodel) if the result is > 1 then the example is
		if(assessModel[0][x] < 0 and assessModel[1][x] > 0):
			# x / y and x is < 0 , so we have to classify this as female
			# we have to be sure that the prediction will be above the threshold 
			prediction.append(float(threshold_input) + 1)
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
	for x in range(len(prediction)):
		if(prediction[x]<1.019 and prediction[x]>1.012):
			print prediction[x] , ' can not decide'
		elif(prediction[x]<float(threshold_input)):#the model predict male and we check if it is indeed male
			#print prediction[x], ( Y[x] == 0 )
			decision = (Y[x] == 0)
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
	testModels(data,threshold)
	#GaussianMixtureModel_only_for_testing(data)

main()
