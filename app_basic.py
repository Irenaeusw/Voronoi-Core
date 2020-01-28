import os
from flask import Flask, render_template, request
import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns 
import pandas as pd 
import grainAnalysis 

def runImageAnalysis(src_path): 
    '''

    '''
    grainAnalysis.main(src_path) 

__author__ = 'ibininja'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        
        runImageAnalysis(destination)

    return render_template("complete.html")

if __name__ == "__main__":
    app.run(port=4555, debug=True)