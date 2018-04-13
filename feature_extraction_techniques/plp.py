#!usr/bin/python

import numpy
import numpy.matlib
import scipy
from scipy.fftpack.realtransforms import dct
from sidekit.frontend.vad import pre_emphasis
from sidekit.frontend.io import *
from sidekit.frontend.normfeat import *
from sidekit.frontend.features import *
import scipy.io.wavfile as wav
import numpy as np


def readWavFile(wav):
	#given a path from the keyboard to read a .wav file
	#wav = raw_input('Give me the path of the .wav file you want to read: ')
	inputWav = '/home/gionanide/Theses_2017-2018_2519/MEEI-RainBow'+wav
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

def PLP():
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	for x in range(1,int(amount)+1):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		#inputWav = readWavFile(wav)
		signal,rate = initialize(wav)
		#returns PLP coefficients for every frame 
		plp_features = plp(signal,rasta=True)
		meanFeatures(plp_features[0])	


#compute the mean features for one .wav file (take the features for every frame and make a mean for the sample)
def meanFeatures(plp_features):
	#make a numpy array with length the number of plp features
	mean_features=np.zeros(len(plp_features[0]))
	#for one input take the sum of all frames in a specific feature and divide them with the number of frames
	for x in range(len(plp_features)):
		for y in range(len(plp_features[x])):
			mean_features[y]+=plp_features[x][y]
	mean_features = (mean_features / len(plp_features)) 
	print mean_features
	


def main():
	PLP()

main()
