#!usr/bin/python
from __future__ import division
import pickle
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.patches as mpatches

def readFeaturesFile(gender):
	names = ['Feature1', 'Feature2', 'Feature3', 'Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',
'Feature10','Feature11','Feature12','Feature13','Gender']
	
	#check the gender
	if(int(gender)==0):
		data = pd.read_csv("/home/gionanide/Theses_2017-2018_2519/features/gmm_healthy_mfcc.txt",names=names )
	elif(int(gender)==1):
		data = pd.read_csv("/home/gionanide/Theses_2017-2018_2519/features/gmm_captured_mfcc.txt",names=names )
	else:
		data = pd.read_csv("/home/gionanide/Theses_2017-2018_2519/features/mfcc_featuresNewDatabase.txt",names=names )
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


def testModels(data,threshold_input):

	gmmFiles = ['/home/gionanide/Theses_2017-2018_2519/models/finalizedModel_0_healthy.gmm','/home/gionanide/Theses_2017-2018_2519/models/finalizedModel_1_parkinson.gmm']
	models = [pickle.load(open(filename,'r')) for filename in gmmFiles]
	log_likelihood = np.zeros(len(models))
	genders = ['healthy','parkinson']
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
	condition=[]
	true_negative=0
	true_positive=0
	false_positive=0
	false_negative=0
	for x in range(len(prediction)):
		if(prediction[x]<1.04790 and prediction[x]>0.97890):
			print prediction[x] , ' can not decide'
			print '\n'
			continue
		if(prediction[x]<float(threshold_input)):#the model predict healthy and we check if it is indeed male
			#print prediction[x], ( Y[x] == 0 )
			decision = (Y[x] == 0)
			condition.append('healthy')
			assessment.append(decision)
			if(decision):
				true_negative+=1
			else:
				false_negative+=1
		else:
			decision1 = (Y[x] == 1)#the model predict parkinson and we check if it is indeed female
			#print prediction[x], ( Y[x] == 1 )
			condition.append('parkinson')
			assessment.append(decision1)
			if(decision1):
				true_positive+=1
			else:
				false_positive+=1
	for x in range(len(assessment)):
		if(assessment[x]==False):
			print prediction[x],assessment[x],condition[x]
			print assessModel[0][x],assessModel[1][x]
			print '\n'
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

	print 'tpr: ',tpr
	print 'fpr : ',fpr

	#ROC curve with error and reject option
				
	winner = np.argmax(log_likelihood)
	print ''
	print "\tdetected as - ", genders[winner],"\n\tscores: healthy ",log_likelihood[0],",parkinson ", log_likelihood[1],"\n"

	male_support = confusion_matrix[0][0] + confusion_matrix[0][1]
	female_support = confusion_matrix[1][0] + confusion_matrix[1][1]

	print 'Confusion matrix:       support: '
	print '               ',confusion_matrix[0],'    0.0: ',male_support
	print '               ',confusion_matrix[1],'    1.0: ',female_support
	print ''
	print 'Accuracy without reject option: 88.1122206372 \n'
	print 'Accuracy with reject option: ', ( assessment.count(True) / len(assessment) ) * 100
	
	
	#with reject option
	tpr_plot = np.array([0,tpr,1])
	fpr_plot = np.array([0,fpr,1])
	#without reject option
	tpr_plot_wr = np.array([0,0.23381294964,1])
	fpr_plot_wr = np.array([0,0.0202739726027,1])

	#area under the curve with the reject option
	area_under_plot =  metrics.auc(fpr_plot,tpr_plot)
	#area under the curve without the reject option
	area_under_wr = metrics.auc(fpr_plot_wr,tpr_plot_wr)

	
	

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
	plt.show()




def main():
	#gender = raw_input('Choose the gender for training press 1(Female) or 0(Male) and any other number for testing: ')
	data = readFeaturesFile(gender=6)

	#threshold = raw_input('Threshold: ')
	threshold = 1.05
	testModels(data,threshold)


main()

