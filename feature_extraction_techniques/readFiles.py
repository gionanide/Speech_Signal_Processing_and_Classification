#!usr/bin/python
import os

def readCases():
	#healthy cases = 1.825 , we are going to mark these cases with 0(zero)
	#parkinson cases = 278 , and theses cases with 1(one)
	healthyCases = os.listdir('/home/gionanide/Theses_2017-2018_2519/gionanide/Audio_Files/HC')
	capturedCases = os.listdir('/home/gionanide/Theses_2017-2018_2519/gionanide/Audio_Files/PD')
	#using the os libary that Python provides to read all the files from a scertain directory
	#the function return two arrays with all the file names that are in the specific directory
