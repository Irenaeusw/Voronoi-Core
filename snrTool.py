import numpy as np
import cv2 
import timeit 
import os
from matplotlib import pyplot as plt 
import math 
# from scipy.signal import convolve2d
# from scipy.signal import medfilt
# from scipy import signal 
# import copy
# import openpiv.tools 
import tkinter
from tkinter.filedialog import askopenfilename 
from tkinter.filedialog import askdirectory
# import getpass
# import glob 
# import csv
# import pywt 
import math 
# import imageCanvas as ic 
import tkinter as tk 
# import svm 
# import svmutil 
# from svm import * 
# from svmutil import * 
import math as m 
import sys 
# from scipy.special import gamma as tgamma 

def saveImage(src, saveDir, imageName, imageType=".tif"): 
    cv2.imwrite("{}\\{}{}".format(saveDir, imageName, imageType), src)