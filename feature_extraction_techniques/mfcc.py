#!usr/bin/python
from python_speech_features import mfcc
import scipy.io.wavfile as wavv
import numpy as np

def normalizeDataStd(data):
	#normalize with mean and std 
	#norm = (x_i - mean) / std
	mean = np.mean(data,axis=0)
	std = np.std(data,axis=0)
	data = (data - mean) / std

def normalizeDataMM(mean_features):
	#normalize with min , max 
	#norm = (x_i - min ) / (max - min)
	dataMin = np.amin(data,axis=0)
	dataMax = np.amax(data,axis=0)
	base = dataMax - dataMin
	data = (data - dataMin) / base

def mfcc_features_extraction(wav):
	inputWav,wav = readWavFile(wav)
	rate,signal = wavv.read(inputWav)
	mfcc_features = mfcc(signal,rate)
	#n numpy array with size of the number of frames , each row has one feature vector
	return mfcc_features,wav

def mean_features(mfcc_features,wav):
	#make a numpy array with length the number of mfcc features
	mean_features=np.zeros(len(mfcc_features[0]))
	#for one input take the sum of all frames in a specific feature and divide them with the number of frames
	for x in range(len(mfcc_features)):
		for y in range(len(mfcc_features[x])):
			mean_features[y]+=mfcc_features[x][y]
	mean_features = (mean_features / len(mfcc_features)) 
	print mean_features
	writeFeatures(mean_features,wav)

def readWavFile(wav):
	#given a path from the keyboard to read a .wav file
	#wav = raw_input('Give me the path of the .wav file you want to read: ')
	inputWav = 'PATH_TO_WAV'+wav
	return inputWav,wav

def writeFeatures(mean_features,wav):
	#write in a txt file the output vectors of every sample
	f = open('mfcc_features.txt','a')#sample ID
	#f = open('mfcc_featuresLR.txt','a')#only to initiate the input for the ROC curve
	wav = makeFormat(wav)
	np.savetxt(f,mean_features,newline=",")
	f.write(wav)
	f.write('\n')
	

def makeFormat(wav):
	#if i want to keep only the gender (male,female)
	wav = wav.split('/')[1].split('-')[1]
	#only to make the format for Logistic Regression
	'''if (wav=='Female'):
		wav='1'
	else:
		wav='0'''
	return wav
	

def main():
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	for x in range(1,int(amount)):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		mfcc_features,inputWav = mfcc_features_extraction(wav)
		mean_features(mfcc_features,inputWav)

main()
