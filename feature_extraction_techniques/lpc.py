#!usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import wave
import scipy.io.wavfile as wav
from scipy import signal
import scipy as sk
from audiolazy import *
from audiolazy import lpc
from sklearn import preprocessing
import scipy.signal as sig
import scipy.linalg as linalg


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
	#plots using matplotlib
	'''plt.figure(figsize=(9, 7)) 
	plt.subplot(211, facecolor='darkslategray')
	plt.title('Signal wave')
	plt.ylim(-50000, 50000)
	plt.ylabel('Amplitude', fontsize=16)
	plt.plot(Time,signal,'C1')
	plt.subplot(212, facecolor='darkslategray')
	plt.title('Pre-emphasis')
	plt.ylim(-50000, 50000)
	plt.xlabel('time(s)', fontsize=10)
	plt.ylabel('Amplitude', fontsize=16)
	plt.plot(EmphasizedTime,emphasizedSignal,'C1')
	plt.show()'''
	return emphasizedSignal, signal , rate


def visualize(rate,signal):
	#taking the signal's time
	Time=np.linspace(0, len(signal)/rate, num=len(signal))
	#plots using matplotlib
	plt.figure(figsize=(10, 6)) 
	plt.subplot(facecolor='darkslategray')
	plt.title('Signal wave')
	plt.ylim(-40000, 40000)
	plt.ylabel('Amplitude', fontsize=16)
	plt.xlabel('Time(s)', fontsize=8)
	plt.plot(Time,signal,'C1')
	plt.draw()
	#plt.show()

def framing(fs,signal):	
	#split the signal into frames
	windowSize = 0.025 # 25ms
	windowStep = 0.01 # 10ms
	overlap = int(fs*windowStep)
	frameSize = int(fs*windowSize)# int() because the numpy array can take integer as an argument in the initiation
	numberOfframes = int(np.ceil(float(np.abs(len(signal) - frameSize)) / overlap ))
	print 'Overlap is: ',overlap 
	print 'Frame size is: ',frameSize
	print 'Number of frames: ',numberOfframes
	frames = np.ndarray((numberOfframes,frameSize))# initiate a 2D array with numberOfframes rows and frame size columns
	#assing samples into the frames (framing)
	for k in range(0,numberOfframes):
		for i in range(0,frameSize):
			if((k*overlap+i)<len(signal)):
				frames[k][i]=signal[k*overlap+i]
			else:
				frames[k][i]=0
	return frames,frameSize

def hamming(frames,frameSize):
	# Windowing with Hamming
	#Hamming implementation : W[n] = 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frameSize - 1))  
	# y[n] = s[n] (signal in a specific sample) * w[n] (the window function Hamming) 
	frames*=np.hamming(frameSize)
	'''plt.figure(figsize=(10, 6)) 
	plt.subplot(facecolor='darkslategray')
	plt.title('Hamming window')
	plt.ylim(-40000, 40000)
	plt.ylabel('Amplitude', fontsize=16)
	plt.xlabel('Time(ms)', fontsize=8)
	plt.plot(frames,'C1')
	plt.show()'''
	return frames
	
def autocorrelation(hammingFrames):
	correlateFrames=[]
	for k in range(len(hammingFrames)):
		correlateFrames.append(np.correlate(hammingFrames[k],hammingFrames[k],mode='full'))
	#print 'Each frame after windowing and autocorrelation: \n',correlateFrames
	yolo =  correlateFrames[len(correlateFrames)/2:]
	return yolo
	
	
	

def levinsonDurbin(correlateFrames):
	#normalizedCF = preprocessing.normalize(correlateFrames, norm='l2')
	filt1 = levinson_durbin(correlateFrames,13)
	print filt1.numerator[1:]


def myLPC():
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	for x in range(1,int(amount)+1):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		emphasizedSignal,signal,rate = preEmphasis(wav)
		#visualize(rate,signal)
		frames , frameSize = framing(rate,signal)
		hammingFrames = hamming(frames,frameSize)
		correlateFrames = autocorrelation(hammingFrames)
		merged=correlateFrames[0]
		for x in range(1,len(correlateFrames)-1):
			merged = np.append(merged,correlateFrames[x])
		lev_Dur = levinsonDurbin(merged)
		

def LPC_autocorrelation(order=13):
	#Takes in a signal and determines lpc coefficients(through autocorrelation method) and gain for inverse filter.
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	for x in range(1,int(amount)+1):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		#preemhasis filter
		emphasizedSignal,signal,rate = preEmphasis(wav)
		length = emphasizedSignal.size
		#prepare the signal for autocorrelation , fast Fourier transform method
		autocorrelation = sig.fftconvolve(emphasizedSignal, emphasizedSignal[::-1])
		#autocorrelation method
		autocorr_coefficients = autocorrelation[autocorrelation.size/2:][:(order + 1)]
		
		
		#using levinson_durbin method instead of solving toeplitz
		lpc_coefficients_levinson = levinson_durbin(autocorr_coefficients,13)
		print 'With levinson_durbin instead of toeplitz ' , lpc_coefficients_levinson.numerator
		
		
		#The Toeplitz matrix has constant diagonals, with c as its first column and r as its first row. If r is not given
		R = linalg.toeplitz(autocorr_coefficients[:order])
		#Given a square matrix a, return the matrix ainv satisfying
		lpc_coefficients = np.dot(linalg.inv(R), autocorr_coefficients[1:order+1])
		#(Multiplicative) inverse of the matrix (inv),  Returns the dot product of a and b. If a and b are both scalars 
		#or both 1-D arrays then a scalar is returned; otherwise an array is returned. If out is given, then it is returned  (np.dot())
		lpc_features=[]
		for x in lpc_coefficients:
			lpc_features.append(x)
		print lpc_features
	

def LPC():
	folder = raw_input('Give the name of the folder that you want to read data: ')
	amount = raw_input('Give the number of samples in the specific folder: ')
	for x in range(1,int(amount)+1):
		wav = '/'+folder+'/'+str(x)+'.wav'
		print wav
		emphasizedSignal,signal,rate = preEmphasis(wav)
		filt = lpc(emphasizedSignal,order=13)
		lpc_features =  filt.numerator[1:]
		print len(lpc_features)
		print lpc_features
		

def main():
	LPC()
	#myLPC()	
	LPC_autocorrelation()

		




main()
	
