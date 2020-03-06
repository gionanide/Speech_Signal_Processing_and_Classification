#!usr/bin/python

from pysptk.sptk import *
from scipy.signal import hamming
import numpy.matlib
import scipy
import scipy.io.wavfile as wav
import numpy as np
import wave
from python_speech_features.sigproc import *
from math import *

def readWavFile(wav):
	#given a path from the keyboard to read a .wav file
	#wav = raw_input('Give me the path of the .wav file you want to read: ')
	inputWav = 'PATH_TO_WAV'+wav
	return inputWav

#reading the .wav file (signal file) and extract the information we need 
def initialize(inputWav):
	rate , signal  = wav.read(readWavFile(inputWav)) # returns a wave_read object , rate: sampling frequency 
	sig = wave.open(readWavFile(inputWav))
	# signal is the numpy 2D array with the date of the .wav file
	# len(signal) number of samples
	sampwidth = sig.getsampwidth()
	print 'The sample rate of the audio is: ',rate
	print 'Sampwidth: ',sampwidth	
	return signal ,  rate 

#implementation of the low-pass filter
def lowPassFilter(signal, coeff=0.97):
	return np.append(signal[0], signal[1:] - coeff * signal[:-1]) #y[n] = x[n] - a*x[n-1] , a = 0.97 , a>0 for low-pass filters 


def preEmphasis(wav):
	#taking the signal
	signal , rate = initialize(wav)
	#Pre-emphasis Stage
	preEmphasis = 0.97
	emphasizedSignal = lowPassFilter(signal)
	Time=np.linspace(0, len(signal)/rate, num=len(signal))
	EmphasizedTime=np.linspace(0, len(emphasizedSignal)/rate, num=len(emphasizedSignal))
	return emphasizedSignal, signal , rate

def writeFeatures(mgca_features,wav):
	#write in a txt file the output vectors of every sample
	f = open('mel_generalized_features.txt','a')#sample ID
	#f = open('mfcc_featuresLR.txt','a')#only to initiate the input for the ROC curve
	wav = makeFormat(wav)
	np.savetxt(f,mgca_features,newline=",")
	f.write(wav)
	f.write('\n')
	

def makeFormat(wav):
	#if i want to keep only the gender (male,female)
	wav = wav.split('/')[1].split('-')[1]
	#only to make the format for Logistic Regression
	if (wav=='Female'):
		wav='1'
	else:
		wav='0'
	return wav


def mgca_feature_extraction(wav):
	#I pre-emphasized the signal with a low pass filter
	emphasizedSignal,signal,rate = preEmphasis(wav)
	
	
	#and now I have the signal windowed
	emphasizedSignal*=np.hamming(len(emphasizedSignal))
	
	mgca_features = mgcep(emphasizedSignal,order=12)

	writeFeatures(mgca_features,wav)
		



def mel_Generalized():
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	print 'Mel-Generalized Cepstrum analysis github implementation '
	for x in range(1,int(amount)+1):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		mgca_feature_extraction(wav)
		
		

def main():
	mel_Generalized()

main()
