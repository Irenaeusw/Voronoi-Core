#This python file is a compilation of various tools that will process images 
#to reveal key values/features to be used in denoising and stigmation correction
#algorithms. 

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


# Initialize global variables here: 
OPTICALFLOW_LINETRAIL = 0 
OPTICALFLOW_VECTORMAP = 1 
VECTORFIELD_ARROWS = 0 
VECTORFIELD_DOTS = 1 

SHARPNESS_EDGE_CHARACTERIZATION = 0  
SHARPNESS_EDGE_GRADIENT = 1 
SHARPNESS_BRISQUE = 2 

# stabilizeVideoSmoothControl
FRAMEBYFRAME = 0 
FRAMEBYFIRST = 1 
FRAMECOMBINATION = 2 

# Global Variables for Class AutoStig 
OPTIMAL_C = 10

# Mouse event global variables: 
refPt = [] 
cropping = False 
window_image = np.zeros( (1,1), dtype=np.uint8) 
window_copy = np.zeros( (1,1), dtype=np.uint8) 
temp_point = (0,0) 



def plotImage(src, figsize=(12,12)): 
    '''
    Displays an image read with OpenCV cv2.imread() function in a Jupyter notebook using 
    matplotlib. 
    '''

    display = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize) 
    plt.imshow(display) 


def setupMultipleDatasets(): 
    '''
    This function allows the user the set up multiple datasets of images to be 
    processed. When called, the user will be prompted to select a source folder 
    that houses folders of datasets that contain images to be processed. 

    Returns: 

        (return1): 
            (list[str...]): List of directories of folders containing image datasets to be 
                            processed. 
        (return2): 
            (str): Save directory for application to save data. 

    '''

    datasetLocation = loadFolderDir("Dataset Location") 
    dataset_list = listFolders(datasetLocation) 

    saveLocation = loadFolderDir("Save Directory") 

    return dataset_list, saveLocation 


def listFolders(folderDir): 
    folders = [] 
    ends = os.listdir(folderDir)
    for i in range(len(ends)): 
        folders.append( "{}\\{}".format(folderDir, ends[i])) 
    
    return folders

def blurImage(src, kernel_size, num_times, saveDir): 
    for i in range(num_times): 
        src = cv2.GaussianBlur(src, kernel_size, 0) 
        cv2.imwrite("{}/blur{}.png".format(saveDir,i+1), src)

def getPointsInLine(p1, p2, line_pixel_sample): 
    '''
    Returns array of points from line segment defined as 2 points. 
    '''

    dx = p2[0] - p1[0] 
    dy = p2[1] - p1[1] 
    print("dx={} and dy={}".format(dx, dy))
    counter = 1  

    # initialize y=mx+b equation 
    try: 
        m = dy/dx 
    except ZeroDivisionError: # When user creates veertical line 
        m = 100 
    b = p1[1] - m*p1[0] 

    points = [] 
    counter = 1 
    points.append(p1)

    # Determine if line is too steep...
    if np.abs(dy) > 35: 
        # Use x = (y-b)/m equation to find points 
        if dy > 0: 
            while(counter < np.abs( (p2[1] - p1[1])/line_pixel_sample )): 
                curr_y = p1[1] + counter*line_pixel_sample 
                points.append( ( int(np.around( (curr_y-b)/m ) ), int(np.around(curr_y) ) ) ) 
                counter += 1
        elif dy < 0: 
            while(counter < np.abs( (p2[1] - p1[1])/line_pixel_sample )): 
                curr_y = p1[1] - counter*line_pixel_sample 
                points.append( ( int(np.around( (curr_y-b)/m ) ), int(np.around(curr_y) ) ) ) 
                counter += 1
    # If line is not too steep, can just get points with y = mx + b
    else: 

        if dx > 0: 
            while(counter < np.abs( (p2[0] - p1[0])/line_pixel_sample )): 
                curr_x = p1[0] + counter*line_pixel_sample 
                points.append( ( int(np.around(curr_x) ), int(np.around(m*curr_x + b) ) ) ) 
                counter += 1
        else: 
            while(counter < np.abs( (p2[0] - p1[0])/line_pixel_sample )): 
                curr_x = p1[0] - counter*line_pixel_sample 
                points.append( (int(np.around(curr_x)) , int(np.around(m*curr_x + b) ) ) ) 
                counter += 1
    return points 


def cartesianToPolar(x, y): 
    '''
    This function takes in cartesian coordinates descriptor x and y, 
    and converts to its polar form. It returns the pixel coordinate in the form 
    (r, theta) where the starting point is the center of the image. 
    '''

    r = np.sqrt( x**2 + y**2 ) 

    # Conditionals: When y/x is negative 
    if x > 0 and y < 0: 
        theta = np.arctan( y/x )
    elif x < 0 and y > 0: 
        theta = np.arctan( y/x ) 
        theta -= np.pi 

    # Conditionals: When y/x is positive 
    elif x > 0 and y > 0: 
        theta = np.arctan( y/x ) 
    elif x < 0 and y < 0: 
        theta = np.arctan( y/x ) 
        theta += np.pi 

    # Conditionals: When (y,x) lies on x-axis, y = 0 
    elif y == 0 and x > 0: 
        theta = 0  
    elif y == 0 and x < 0: 
        theta = 3.142 
    
    # Conditionals: When (y,x) lies on y-axis, x = 0
    elif x == 0 and y > 0: 
        theta = 1.5708 
    elif x == 0 and y < 0: 
        theta = 4.712 
    
    # Conditionals: origin point, x = 0, y = 0 
    elif x == 0 and y == 0: 
        theta = 0 

    return r, theta 

def partitionImage8(src, points=None): 
    '''
    This function takes in a square image to be partitioned to 8 equal sections 
    to be used in FFT power spectrum regional analysis. Partitioned according to 
    "A Robust Focussing and Astigmatism Correction Method for the Scanning 
    Electron Microscope" by K.H. Ong, J.C.H. Phang, J.T.L. Thong. 

    If only certain points are to be checked, there is a points parameter where
    the desired points to be checked can be inputted. 
    '''

    assert len(src.shape) == 2 
    height, width = src.shape 
    assert height == width 

    center_coords = (height//2, width //2) 
    
    r1_partition = [] 
    r2_partition = [] 
    r3_partition = [] 
    r4_partition = [] 

    s1_partition = [] 
    s2_partition = [] 
    s3_partition = [] 
    s4_partition = [] 

    if points == None: 
        for i in range(height): 
            # Pre-processing to recenter origin 
            y = center_coords[0] - i 
            for j in range(width):
                # Pre-processing to recenter origin 
                x = j - center_coords[1] 

                # Converting to polar coordinate system 
                r, theta = cartesianToPolar(x, y) 

                # Check if in R1 
                if (0 <= theta and theta <= 0.393) or (-0.393 <= theta and theta <= 0): 
                    r1_partition.append( (i,j) )
                # Check if in R2 
                elif (-3.534 <= theta and theta <= -3.142) or (3.142 <= theta and theta <= 3.534): 
                    r2_partition.append( (i,j) ) 
                # Check if in R3 
                elif (1.178 <= theta and theta <= 1.571) or (-4.712 <= theta and theta <= -4.32): 
                    r3_partition.append( (i,j) ) 
                # Check if in R4 
                elif (4.32 <= theta and theta <= 4.712) or (-1.571 <= theta and theta <= -1.178): 
                    r4_partition.append( (i,j) ) 
                # Check if in S1
                elif (0.393 <= theta and theta <= 1.178): 
                    s1_partition.append( (i,j) ) 
                # Check if in S2 
                elif (3.534 <= theta and theta <= 4.32): 
                    s2_partition.append( (i,j) ) 
                # Check if in S3 
                elif (-4.32 <= theta and theta <= -3.534): 
                    s3_partition.append( (i,j) ) 
                # Check if in S4 
                elif (-1.178 <= theta and theta <= -0.393): 
                    s4_partition.append( (i,j) ) 
    else: 
        for point in points: 
            # Pre-processing of points to re-center origin 
            y = center_coords[0] - point[0] 
            x = point[1] - center_coords[1] 

            # Converting to polar coordinate system 
            r, theta = cartesianToPolar(x, y) 

            # Check if in R1 
            if (0 <= theta and theta <= 0.393) or (-0.393 <= theta and theta <= 0): 
                r1_partition.append( point )
            # Check if in R2 
            elif (-3.534 <= theta and theta <= -3.142) or (3.142 <= theta and theta <= 3.534): 
                r2_partition.append( point ) 
            # Check if in R3 
            elif (1.178 <= theta and theta <= 1.571) or (-4.712 <= theta and theta <= -4.32): 
                r3_partition.append( point ) 
            # Check if in R4 
            elif (4.32 <= theta and theta <= 4.712) or (-1.571 <= theta and theta <= -1.178): 
                r4_partition.append( point ) 
            # Check if in S1
            elif (0.393 <= theta and theta <= 1.178): 
                s1_partition.append( point ) 
            # Check if in S2 
            elif (3.534 <= theta and theta <= 4.32): 
                s2_partition.append( point ) 
            # Check if in S3 
            elif (-4.32 <= theta and theta <= -3.534): 
                s3_partition.append( point ) 
            # Check if in S4 
            elif (-1.178 <= theta and theta <= -0.393): 
                s4_partition.append( point ) 
    
    return r1_partition, r2_partition, r3_partition, r4_partition, s1_partition, s2_partition, s3_partition, s4_partition





# def getPointsInLine(p1, p2, line_pixel_sample):  
#     '''
#     Returns array of points from line segment defined as 2 points. 
#     '''
#     dx = p2[0] - p1[0] 
#     dy = p2[1] - p1[1] 
#     print("dx={} and dy={}".format(dx, dy))

#     points = [] 
#     points.append(p1) 

#     counter = 1 
#     while(True): 

#         # HARD CODE JUST FOR DEBUGGING. CAN REMOVE THIS BLOCK LATER 
#         if (dx == 0 and dy == 0): 
#             continue
#         elif dx == 0 and dy != 0: 
#             if dy > 0: 
#                 if (p1[1] + line_pixel_sample*counter) < p2[1]: 
#                     points.append( (p1[0], p1[1] + line_pixel_sample*counter ) )
#                     counter += 1 
#                 else: 
#                     break
#             elif dy < 0: 
#                 if (p1[1] - line_pixel_sample*counter) > p2[1]: 
#                     points.append( (p1[0], p1[1] - line_pixel_sample*counter ) )
#                     counter += 1 
#                 else: 
#                     break
#             else: 
#                 print("Something went wrong in getPointsInLine...") 
#                 return None 
#         elif dx != 0 and dy == 0: 
#             if dx > 0: 
#                 if (p1[0] + line_pixel_sample*counter) < p2[0]: 
#                     points.append( (p1[0] + line_pixel_sample*counter, p1[1]) )
#                     counter += 1 
#                 else: 
#                     break
#             elif dx < 0: 
#                 if ((p1[0] - line_pixel_sample*counter) > p2[0]): 
#                     points.append( (p1[0] - line_pixel_sample*counter, p1[1] ) )
#                     counter += 1 
#                 else: 
#                     break
#             else: 
#                 print("Something went wrong in getPointsInLine...") 
#                 return None 
#         # HARD CODE JUST FOR DEBUGGING. CAN REMOVE THIS BLOCK LATER

#         elif dx > 0 and dy > 0: 
#             if (p1[0] + line_pixel_sample*counter) < p2[0] and (p1[1] + line_pixel_sample*counter) < p2[1]: 
#                 points.append( (p1[0] + line_pixel_sample*counter, p1[1] + line_pixel_sample*counter ) )
#                 counter += 1 
#             else: 
#                 break
#         elif dx > 0 and dy < 0: 
#             if ((p1[0] + line_pixel_sample*counter) < p2[0]) and ((p1[1] - line_pixel_sample*counter) > p2[1]): 
#                 points.append( (p1[0] + line_pixel_sample*counter, p1[1] - line_pixel_sample*counter ) )
#                 counter += 1 
#             else: 
#                 break

#         elif dx < 0 and dy > 0: 
#             if (p1[0] - line_pixel_sample*counter) > p2[0] and (p1[1] + line_pixel_sample*counter) < p2[1]: 
#                 points.append( (p1[0] - line_pixel_sample*counter, p1[1] + line_pixel_sample*counter ) )
#                 counter += 1 
#             else: 
#                 break

#         elif dx < 0 and dy < 0: 
#             if (p1[0] - line_pixel_sample*counter) > p2[0] and (p1[1] - line_pixel_sample*counter) > p2[1]: 
#                 points.append( (p1[0] - line_pixel_sample*counter, p1[1] - line_pixel_sample*counter ) )
#                 counter += 1 
#             else: 
#                 break 
#         else: 
#             print("Something went wrong in getPointsInLine...") 
#             return None 
#     print(points)
#     return points 

def click_and_get_lines(event, x, y, flags, params): 
    # grab references to global variables 
    global refPt 
    global window_image
    global cropping 
    global window_copy 

    # if left mouse button was clicked, record p1 

    if (event == cv2.EVENT_LBUTTONDOWN and cropping == False): 
        refPt.append( [(x,y)] ) 
        cv2.circle(window_image, (x,y), 1, (0, 255, 0), -1) 
        cropping = True 
        window_copy = window_image.copy() 
    
    # Draw lines with each mouse movement
    
    # if (event == cv2.EVENT_MOUSEMOVE and cropping == True):
    #     window_image = window_copy  
    #     cv2.line(window_image, (x,y), (refPt[-1][0][0], refPt[-1][0][1]), (0, 0, 255), thickness=1) 
    
    if (event == cv2.EVENT_LBUTTONUP and cropping == True): 
        refPt[-1].append( (x,y) ) 
        cv2.circle(window_image, (x,y), 1, (0, 255, 0), -1) 
        print("Line Points Chosen: {}".format(refPt))  
        cv2.line(window_image, (x,y), (refPt[-1][0][0], refPt[-1][0][1]), (0, 255, 0), 1) 
        cropping = False 


    

def click_and_get_points(event, x, y, flags, params): 
    # grab references to global variables 
    global refPt
    global window_image

    # if the left mouse button was clicked, record the starting 
    # (x,y) coordinates and indicate that cropping is being performed... 

    if  event == cv2.EVENT_LBUTTONDOWN: 
        refPt.append( (x,y) ) 
        cv2.circle(window_image, (x,y), 2, (0, 255, 0), thickness=-1) 
        

def chooseMultiplePoints(reference_image, drag_lines=False, line_pixel_sample=2, wrap_for_cv2=False): 
    '''
    This method opens up a GUI window where the user can click and save multiple 
    points to be passed and returned by this function. The user has 3 options when
    this method is called. They can simply left click each reference point of interest, 
    or press 'r' on their keyboard to reset the points, or 'c' to close the window when
    they are done selecting points. 

    FOR NOW: 'r' functionality to reset points is not working...attend to later. 
    '''
    global refPt
    global window_image 

    window_image = reference_image 
    clone = reference_image.copy() 

    cv2.namedWindow("reference_image")
    cv2.setMouseCallback("reference_image", click_and_get_lines) 
    
    while(True): 
        cv2.imshow("reference_image", window_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") : 
            reference_image = clone.copy() 
        
        elif key == ord("c"): 
            cv2.destroyAllWindows()
            break 
    
    # Unwrap drag_lines points to standard points array 
    if drag_lines == True: 
        # Get points from each line storing in temporary array 
        line_points_temp = [] 
        for i in range(len(refPt)): 
            line_points_temp.append(getPointsInLine(refPt[i][0], refPt[i][1], 10)) 
            print("{}:{}".format(refPt[i][0], refPt[i][1]) )
        print(line_points_temp)
        
        # Store array of points into normal points array format 
        points = [] 
        for i in range(len(line_points_temp)): 
            for j in range(len(line_points_temp[i])): 
                points.append(line_points_temp[i][j]) 
        
        # Now it should be in propper points array format. 
        # Wrap for cv2 formatting if needed. 
        if wrap_for_cv2 == True:
            counter = 0 
            wrapped_refPt = [] 
            for i in range(len(points)): 
                wrapped_refPt.append([]) 
                wrapped_refPt[i].append(points[i]) 
            return wrapped_refPt 
        else: 
            return points 
        

    elif drag_lines == False: 
        if wrap_for_cv2 == True: 
            counter = 0 
            wrapped_refPt = [] 
            for i in range(len(refPt)): 
                wrapped_refPt.append([]) 
                wrapped_refPt[i].append(refPt[i]) 
            return wrapped_refPt 
        else: 
            return refPt 



def padVideo(videoDir, saveDir, outputFileName, padding_factor, fps=5): 
    cap = cv2.VideoCapture(videoDir) 
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    waitTime = int(1000//fps) 

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    top_border = int(padding_factor*height) 
    bottom_border = top_border  
    left_border = int(padding_factor*width)
    right_border = int(padding_factor*width) 

    width = width + left_border + right_border 
    height = height + top_border + bottom_border

    out = cv2.VideoWriter("{}/{}.avi".format(saveDir, outputFileName), cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height)) 
  
    border_type = cv2.BORDER_CONSTANT 

    for i in range(n_frames): 
        ret, curr_frame = cap.read() 
        if ret == False: 
            print("Aiyaaaaaaaa") 
            return None 
        
        curr_frame = cv2.copyMakeBorder(curr_frame, top_border, bottom_border, left_border, right_border, border_type)
        out.write(curr_frame) 
        cv2.imshow('Padded', curr_frame) 
        cv2.waitKey(waitTime) 
    
    cap.release() 
    out.release() 
    cv2.destroyAllWindows

def resizeImageStack(folderDir, saveDir, resize_factor, frameName='Frame', imgType='.tif', startNum=1, grayscale=True): 
    imgDir = orderFrames(folderDir, frameName=frameName, imgType=imgType, startNum=startNum)

    for i in range(len(imgDir)):
        if grayscale == True: 
            curr_img = cv2.imread(imgDir[i], 0) 
        else: 
            curr_img = cv2.imread(imgDir[i]) 
        curr_img = resizeImage(curr_img, resize_factor) 

        defaultNone = None 

        # This happens when there is a skipped number in an image
        # Ex. Neutrophils data has image 0, 1, 2, 4
        # No image 3, so needs to continue the for loop. 
        if type(curr_img) == type(defaultNone): 
            continue 

        cv2.imwrite("{}/Frame{}.tif".format(saveDir, startNum+i), curr_img)


def overlayGridVideo(videoDir, saveDir, grid_dimensions, initial_offset, fps=5, even_grid=False): 
    cap = cv2.VideoCapture(videoDir)
    ret, curr_frame = cap.read() 
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    waitTime = int(np.around(1000/fps))

    if len(curr_frame.shape) == 3: 
        height, width, dimensions = curr_frame.shape 
    elif len(curr_frame.shape) == 2: 
        height, width = curr_frame.shape 
    else: 
        print("The image shape is not 3 or 2. resizeImage will terminate.")
        return None

    out = cv2.VideoWriter("{}/GridOverlayed.avi".format(saveDir), cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height)) 
    
    mask = np.zeros_like(curr_frame) 
    gridPoints = getGridPoints(curr_frame, grid_dimensions=grid_dimensions, initial_offset=initial_offset)
    for point in gridPoints: 
        mask = cv2.circle(mask, point, 1, (0, 0, 255), thickness=-1) 
    printImage('overlay', mask)
    
    # Iterating through all frames 
    for i in range(n_frames-1):
        for j in range(len(gridPoints)): 
            curr_frame = cv2.circle(curr_frame, gridPoints[j], 3, (0, 0, 255), thickness=-1)
        
        out.write(curr_frame) 
        cv2.imshow('Overlayed', curr_frame) 
        cv2.waitKey(waitTime) 

        ret, curr_frame = cap.read() 
        if ret == False: 
            print("Yikes, hit the end of the videostream") 
            break 
    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 
    
def resizeImage(src, scale): 

    try: 
        if len(src.shape) == 3: 
            height, width, dimensions = src.shape 
        elif len(src.shape) == 2: 
            height, width = src.shape 
        else: 
            print("The image shape is not 3 or 2. resizeImage will terminate.")
            return None 
    except AttributeError: 
        print("Empty image thrown to function resizeImage(). This image will be skipped.") 
        return None 
    x_new = int(np.around(width*scale) )
    y_new = int(np.around(height*scale) )

    return_image = cv2.resize(src, (x_new, y_new))

    return return_image

def normalizeImage(src): 
    normalizedImage = np.zeros_like(src, dtype=np.uint8) 
    normalizedImage = ( cv2.normalize(src, normalizedImage, 0, 255, cv2.NORM_MINMAX) ).astype(np.uint8)

    return normalizedImage   

def orderFrames(folderDir, frameName='Frames', imgType=".tif", startNum=1): 
    '''
    Returns list of frame directories ready to be read by opencv. 
    '''
    counter = startNum
    files = os.listdir(folderDir) 

    dirList = []
    for i in range(len(files)): 
        dirList.append(folderDir + "/" + frameName + str(counter) + imgType) 
        counter += 1 
    
    return dirList 

def orderPairs(folderDir, prevName, nextName, imgType='.jpg', startNum=1, endNum=50): 
    '''
    Returns list of tuple frame directories ready to be read by opencv 
    '''
    counter = startNum
    files = os.listdir(folderDir) 

    pairs = [] 
    for i in range(endNum-startNum+1): 
        curr_prev = "{}\\{}{}{}".format(folderDir, counter, prevName, imgType) 
        curr_next = "{}\\{}{}{}".format(folderDir, counter, nextName, imgType)
        
        pairs.append( [curr_prev, curr_next] )
        counter += 1 

    return pairs 

def orderFramesAtlas(folderDir): 
    '''
    Returns list of frame directories in order according to FIBICS Atlas client export naming syntax
    '''
    files = os.listdir(folderDir) 

    dirList = [] 
    for i in range(len(files)): 
        dirList.append("{}\\{}".format(folderDir, files[i]))

    return dirList 

def downsampleFrame(img, scale, colour=False): 
    if colour == False: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape 
        newHeight, newWidth = height*scale, width*scale 
        newImg = cv2.resize(img, ( int(np.around(newWidth)), int(np.around(newHeight)) ))
        return newImg 


def loadFolderDir(description=None): 
    root = tkinter.Tk() 
    if description==None: 
        root.filename = askdirectory(parent=root, title="Choose folder to assign path:") 
    else: 
        root.filename = askdirectory(parent=root, title="Choose " + description + " path:") 

    root.destroy() 
    return root.filename 


def loadFileDir(description=None): 
    root = tkinter.Tk() 
    if description == None:
        root.filename = askopenfilename(parent=root, title="Choose folder to assign path:") 
    else: 
        root.filename = askopenfilename(parent=root, title="Choose " + description + " path:") 
    root.destroy() 
    return root.filename 

def printImage(windowName, img,): 
    cv2.imshow(windowName, img) 
    cv2.waitKey()
    cv2.destroyAllWindows()

def saveImage(src, saveDir, imageName, imageType=".tif"): 
    cv2.imwrite("{}\\{}{}".format(saveDir, imageName, imageType), src)

def zoomImage(src_path_OR_src, noSave=True): 
    
    if noSave == False: 
        app = ic.MainWindow( tk.Tk(), path=src_path_OR_src) 
        app.mainloop() 

    tempSaveDir = r"C:\Users\Irenaeus Wong\Downloads\zoomImage.png"
    cv2.imwrite(tempSaveDir, src_path_OR_src) 
    app = ic.MainWindow( tk.Tk(), path=tempSaveDir) 
    app.mainloop() 

def compare2(img1, img2):
    '''
    This function takes 2 images as NumPy nd.arrays and computes a simple
    mean difference of pixel intensities of pixels of corresponding coordinates. 
    tldr; How similar are the two photos in a metric of light intensity (0-255). 

    Args: 
        param1: 
            (np.ndarray): First image to be compared. 
        param2: 
            (np.ndarray): Second image to be compared. 
    Returns: 
        (float): Mean difference of all pixels rounded to 2 decimal places.  
        (list): List of tuples representing coordinates of variant pixels. 
    '''
    #Making sure that the images are grayscale...
    #Not necessary for now, as maybe later can apply to Color images. 
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

    #Make sure images are of same shape. 
    if not (img1.shape == img2.shape): 
        print("The images input are invalid as they are not the same shape!")
        return None
    #If same shape, continue:  
    else: 
        totalVariance = 0 
        differentPixels = []

        for i in range(img1.shape[0]): 
            for j in range(img1.shape[1]): 
                singleVariance = abs(int(img2[i,j]) - int(img1[i,j]))
                if singleVariance != 0:
                    totalVariance += singleVariance
                    differentPixels.append((i,j,singleVariance))
        
        meanVariance = totalVariance/(img1.shape[0]*img1.shape[1]) 
    return round(meanVariance,2), differentPixels

def estimateGaussNoise(I): 
    '''
    This function will estimate variance of gaussian noise of gray scale
    images represented as NumPy dtype uint8 arrays. The algorithm is 
    referenced from 'Fast noise Variance Estimation', Computer Vision and 
    Image Understanding, Vol 64, No. 2, pp.300-302, Sep. 1996.

    Args: 
        param1: 
            (np.ndarray): The image for noise to be estimated
    Returns: 
        (float): The final sigma value representing estimated 
        standard deviation of the image.
    '''

    H,W = I.shape

    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    sigma = np.sum(np.sum(np.absolute(convolve2d(I,M))))
    sigma = sigma*math.sqrt(0.5*math.pi)/(6*(W-2)*(H-2))

    return sigma

def readFolder(folderPath, imageType, index, imgNum=None ):
    '''
    Method that will read all of the files of certain type in a folder starting from 
    the indicated index, and returns a list of directory addresses for each 
    object of specified imageType in the file.

    Args:
        param1 
            (str): Path of the folder to be read. 
        param2
            (str): Type of file to be read. Ex: "tif", "png", "jpg" etc... 
        param3
            (int): Index of first image to be read in the folder. 
        param4
            (int): If inputted, will only read the number of images specified, otherwise, 
            the function will read all images in the folderPath. 
    Return: 
        (list): a list of directory addresses for the images in specified folderPath. 
    '''
    dirList = os.listdir(folderPath) 
    imageList = []
    if imgNum == None:   #If no number of images specified, function iterates through all functions 
        for i in range(len(dirList)): 
            try:
                if dirList[i + index].endswith(imageType): 
                    imgDir = folderPath + "/" + dirList[i+index]
                    imageList.append(imgDir) 
            except IndexError:    #Throws indexError if folderPath runs out of images to read 
                break             #This happens if the folderPath contains other file types other than the images we want to read
    else: 
        for i in range(imgNum): #If number of images specified, will iterate through number of imgNum
            try:
                if dirList[i + index].endswith(imageType): 
                    imgDir = folderPath + "/" + dirList[i+index]
                    imageList.append(imgDir) 
            except IndexError:    #Throws indexError if folderPath runs out of images to read 
                break             #This happens if the folderPath contains other file types other than the images we want to read

    return imageList 


#Needs more testing and adjustment if consecutive folders of same name are 
#attempted to be made in the same directory.
def createFolder(directory, folderName): 
    '''
    Creates a folder at a specified directory with specified name.

    Args:
        param1 (str): Directory location 
        param2 (str): What the new folder's name is. 
    
    Returns:
        (str): Full named location of new directory.  


    '''
    newFolder = directory + "/" + folderName
    try: 
        if not os.path.exists(newFolder): 
            os.makedirs(newFolder) 
            return str(newFolder)
        else: 
            nameCounter = 1
            newFolder = newFolder + "(" + str(nameCounter) + ")"
            while os.path.exists(newFolder):
                nameCounter += 1
                newFolder = "{}/{}({})".format(directory, folderName, nameCounter)
            os.makedirs(newFolder) 
            return str(newFolder) 

            
    except OSError: 
        print('Error: Unable to create new folder --> ' + newFolder) 

def writeImage(img, fileName, fileType, writeDir): 
    fDir = writeDir + "/"

# --------------------------------------------------- IMAGE QUALITY METRICS --------------------------------------- # 

def getMeanIntensity(img): 
    '''
    This function takes an image represented as a NumPy nd.array and returns 
    the mean intensity value of all the pixels in the image. 

    Args:
        param1:
            (np.ndarray): Image to be processed
    Returns: 
        (float): Mean intensity of all pixels. 
    '''

    height, width = img.shape 
    totalPixels = height*width

    totalIntensities = 0 
    
    #Iterating through all pixels to increment 
    #totalIntensities accordingly 
    for i in range(height): 
        for j in range(width): 
            totalIntensities += img[i, j]

    return (totalIntensities/totalPixels) 


def getSharpness(img): 
    '''
    This function will compute a sharpness value from a normalized
    variance sharpness function. 

    Args
        param1: 
            (np.ndarray): Image to be processed
        Returns: 
            (float): Sharpness value of the image 
    '''
    start = timeit.default_timer()

    height, width = img.shape 
    totalPixels = height*width 
    meanIntensity = getMeanIntensity(img)

    sharpnessFactor = 0  

    #Iterating through all pixels to perform appropriate operations
    for i in range(height):
        for j in range(width): 
            sharpnessFactor += (img[i,j] - meanIntensity)**2
    
    stop = timeit.default_timer() 

    #Debug: Testing line for efficiency. 
    # print("Finding sharpness took: " + str(round((stop - start),2)) + " seconds")

    return (sharpnessFactor/(totalPixels*meanIntensity))

def nonLMFilter(img, hVal): 
    '''
    This function takes an image and 
    '''
    start = timeit.default_timer() 

    dst = cv2.fastNlMeansDenoising(img, dst=None, h=hVal, templateWindowSize=7, searchWindowSize=21)

    stop = timeit.default_timer() 

    #Testing line to track efficiency
    #print("Filtering took: " + str(round((stop - start),2)) + " seconds")

    return dst

def calibrateNLMFilter(img, hbounds):
    '''
    This function takes an image and a tuple of length 2 and 
    iterates through a nonLocalMeansFilter with varying 
    thresholds. It will then compute a series of metrics such 
    as sharpness and gaussian noise in order to determine 
    the best threshold to filter the image through. 

    Args:
        param1:
            (np.ndarray): Image to be processed. 
        param2: 
            (tuple): Lower and Upper bounds of the 
            threshold of the nonLocalMeans filter to be applied. 
    Returns: 
        (tuple): The best suitable h value for sharpness, then gaussian noise reduction.
    '''
    start = timeit.default_timer()

    #Initialize metrics lists 
    sharpnessList = []
    hList = []
    gaussNoiseList = [] 

    #Fills hList wih hVals within the bounds given 
    if (hbounds[0] < hbounds[1]):
        hVals = hbounds[0]
        hList.append(hVals)
        hVals += 1 
        for i in range(hbounds[1]-hbounds[0]):
            hList.append(hVals) 
            hVals += 1 
    else: 
        print("The hbounds must start with the lower value, and end with the upper value")
        return None 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    for i in range(len(hList)):

        #Local method to compute each nonLocalMeans filtered image for each hVal
        dst = nonLMFilter(gray, hList[i])

        #Debug: Show each iteration with their "hVal" in the window screen
        # cv2.imshow("h = " + str(hList[i]), dst)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        sharpnessList.append(round(getSharpness(dst),2))
        gaussNoiseList.append(round(estimateGaussNoise(dst),2))

    print("Sharpness: ") 
    print(sharpnessList) 
    print("\nGaussian Noise: ")
    print(gaussNoiseList)

    #These two variables act as placeholders for what is to be returned.

    sharpH = 0 
    gaussH = 0 

    #Note: Using built in sorting function 
    #Can maybe optimize and use bubble sorting algorithm... 
    sharpnessSorted = sorted(sharpnessList, key=float) 
    gaussSorted = sorted(gaussNoiseList, key=float) 

    #Add +1 because the index is one less than corresponding
    #h value. Also, we want largest sharpness value, and 
    #smallest gaussian noise estimate value. 
    sharpH = sharpnessList.index(sharpnessSorted[-1]) + hbounds[0]
    gaussH = gaussNoiseList.index(gaussSorted[0]) + hbounds[0]

    return sharpH, gaussH

# --------------------------------------------------- ^ IMAGE QUALITY METRICS ^ --------------------------------------- # 

def findLines(img, rho, theta, threshold, minLineLength, maxLineGap): 
    #Convert image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # #NonLocalMeans filter
    # filteredEdges = nonLMFilter(gray, 40)
    # cv2.imshow("NonLocalMeans", filteredEdges)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    kernel = np.ones((5,5),np.float32)/30
    dst = cv2.filter2D(gray, -1, kernel)

    cv2.imshow("blurred", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    edges = cv2.Canny(dst, 10, 200) 
    cv2.imshow("Edges", edges)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #Detect points that form a line 
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

 
    #Draw lines onto the image (img) from lines drawn from (edges) 
    try:
        for line in lines: 
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3) 
    except TypeError: 
        print("The hough transform detected no lines. Try a lower threshold value!")
        return None 


    cv2.imshow("resulting image", img) 
    cv2.waitKey()
    cv2.destroyAllWindows() 

    return edges 


def DFTEstimate(img, c, threshold=None): 
    ''' 
    This function computes and displays an estimation of the 2D fourier transform 
    of the input image. It is important to note this is only an ESTIMATION, saved 
    as a NumPy dtype uint8. 

    This function has been modified to use an optimized version with OpenCV and NumPy. 
    
    Args: 

        param1: 
            (np.ndarray): Numpy array of dtype uint8 image to be converted from the 
            2D spatial space to the fourier space. 
        param2: 
            (float): Fourier transform scaling factor. 10-20 is usually good. 
        param3: 
            (int): Threshold value to threshold out weak edges. 90-150 is usually good. 
    Returns: 
    
        return1: 
            (np.ndarray): Numpy array of dtype uint8 image representing the approximation
            of the fourier transform. NOT TO BE CONVERTED BACK. THIS IS AN APPROXIMATION
            SO ATTEMPTING TO CONVERT IT BACK WILL NOT YIELD DESIRABLE RESULT. 
    '''
    assert len(img.shape) == 2 
    rows, cols = img.shape

    # --------------------------- Optimization Comparison START------------------------------ # 

    # # Optimize FFT to up to 4 times faster 
    # nrows = cv2.getOptimalDFTSize(rows) 
    # ncols = cv2.getOptimalDFTSize(cols) 

    # right = ncols - cols 
    # bottom = nrows - rows 
    # bordertype = cv2.BORDER_CONSTANT # Just to avoid line breakup in PDF file 
    # nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, bordertype, value=0) 

    # dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT) 

    # Discrete FFT without optimization: 
    dft = cv2.dft(np.float32(img), flags= cv2.DFT_COMPLEX_OUTPUT)

    # --------------------------- Optimization Comparison END------------------------------ # 

    dft_shift = np.fft.fftshift(dft) 

    magnitude_spectrum = c*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) 

    tfImage = np.uint8(np.around(magnitude_spectrum)) 

    # If a threshold is specified, do thresholding here: 
    if threshold != None: 
        points = [] 

        assert len(tfImage.shape) == 2
        height, width = tfImage.shape 

        for i in range(height): 
            for j in range(width): 
                if tfImage[i,j] < threshold: 
                    tfImage[i,j] = 0 
                else: 
                    tfImage[i,j] = 255
                    points.append( (i,j) ) 
    
    if threshold == None: 
        return tfImage 
    else: 
        return tfImage, points

# def DFTEstimate(img, c, threshold=None): 
#     ''' 
#     This function computes and displays an estimation of the 2D fourier transform 
#     of the input image. It is important to note this is only an ESTIMATION, saved 
#     as a NumPy dtype uint8. 
    
#     Args: 

#         param1: 
#             (np.ndarray): Numpy array of dtype uint8 image to be converted from the 
#             2D spatial space to the fourier space. 
#         param2: 
#             (float): Fourier transform scaling factor. 10-20 is usually good. 
#         param3: 
#             (int): Threshold value to threshold out weak edges. 90-150 is usually good. 
#     Returns: 

#         return1: 
#             (np.ndarray): Numpy array of dtype uint8 image representing the approximation
#             of the fourier transform. NOT TO BE CONVERTED BACK. THIS IS AN APPROXIMATION
#             SO ATTEMPTING TO CONVERT IT BACK WILL NOT YIELD DESIRABLE RESULT. 
#     '''
#     assert len(img.shape) == 2 
#     rows, cols = img.shape

#     dft = cv2.dft(np.float32(img), flags= cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft) 

#     magnitude_spectrum = c*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) 

#     tfImage = np.uint8(np.around(magnitude_spectrum)) 

#     # If a threshold is specified, do thresholding here: 
#     if threshold != None: 
#         points = [] 

#         assert len(tfImage.shape) == 2
#         height, width = tfImage.shape 

#         for i in range(height): 
#             for j in range(width): 
#                 if tfImage[i,j] < threshold: 
#                     tfImage[i,j] = 0 
#                 else: 
#                     tfImage[i,j] = 255
#                     points.append( (i,j) ) 
    
#     if threshold == None: 
#         return tfImage 
#     else: 
#         return tfImage, points 

def getLFLines(img, pixelThreshold, edgeThreshold): 
    ''' 
    This function takes in a discrete fourier transform estimation image and computes 
    the lines representing low frequency occurances. 
    Args: 
        param1:
            (np.ndarray): DFT estimation image to be processed. 
        param2: 
            (int): Pixel threshold to filter out any pixels below this value. 
        param3: 
            (int): Canny edge detection threshold value. 
    Returns: 
        (list): List of tuples of coordinates of lines with format (x1,y1,x2,y2). 
    '''
    rows, cols = img.shape

    for i in range(rows): 
        for j in range(cols): 
            if img[i][j] < np.uint8(pixelThreshold): 
                img[i][j] = 0

    edges = cv2.Canny(img, edgeThreshold, 200) 

    #TEMPORARY DEBUG BLOCK: 
    cv2.imshow("thresholded", img) 
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(r"C:\Users\Irenaeus Wong\Desktop\DFT\Test Images\transformed\Rot0deg Second Run\Individual Pixels/thresholded.tif", img)

    cv2.imshow("edgesfound", edges) 
    cv2.waitKey()
    cv2.destroyAllWindows()


    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength = 50, maxLineGap=50) 

    return lines 
    
def drawHoughLines(img, lines): 
    for line in lines: 
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2,y2), (255, 255, 255), 3)

# Bad function as snrTool.normalizeImage() does this more efficiently 
def floatToImage(img):
    '''
    This function takes in a NumPy nd.array of type floats and scales it down to 
    appropriate gray scale NumPy nd.array uint8 values from 0 to 255. Useful for 
    converting discrete fourier transform magnitude spectrums to a processable image.

    Args: 
        param1: 
            (np.ndarray): Image as a float to be converted
    Returns: 
        return1: 
            (np.ndarray): Image scaled and converted to uint8. 
    '''

    maxElement = np.amax(img) 
    conversionFactor = maxElement/255
    convertedImage = np.uint8(np.around(img/conversionFactor)) 
    return convertedImage

def fourierSquareFilter(img, squareSideLength, exportDir): 
    '''
    This function is for demos. Do not use otherwise. 
    '''
    rows, cols = img.shape
    tfImage = DFTEstimate(img) 
    m = int(rows/2) 
    print(m)
    n = int(cols/2) 
    print(n)

    lines = getLFLines(tfImage, 210, 10)    #CHANGE THE 218 FOR DIFF IMAGES

    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) 

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])) 

    #Show original magnitude spectra
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Raw Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.savefig(exportDir + "/InputAndSpectrum.tif")
    plt.show()

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[(m- int(squareSideLength/2)):(m+int(squareSideLength/2)), (n-int(squareSideLength/2)):(n+int(squareSideLength/2))] = 1 


    # mask = np.ones((rows, cols, 2), np.uint8)
    # mask[55,86] = 0 
    # mask[83,399] = 0 
    # mask[217,200] = 0 
    # mask[295,312] = 0 
    # mask[429,113] = 0 
    # mask[457,426] = 0 


    fshift = dft_shift*mask 
    magnitude_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1])) 

    plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum Masked'), plt.xticks([]), plt.yticks([])
    plt.savefig(exportDir + "/Magnitude Spectrum Masked.tif")
    plt.show()

    #Invert back to spatial domain
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift) 

    imgConverted = np.uint8(18*np.log(img_back[:,:,0]))
    cv2.imshow("No Magnitude Scaling", imgConverted)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(exportDir + "/No Magnitude Scaling.tif", imgConverted)

    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    print(img_back) 
    # folderDir = createFolder(r"C:\Users\Irenaeus Wong\Desktop\DFT\Test Images\transformed", "/filteredImage3")

    plt.imshow(img_back, cmap= 'gray') 
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Inverse Transformed'), plt.xticks([]), plt.yticks([])
    # plt.savefig(r"C:\Users\Irenaeus Wong\Desktop\DFT\Test Images\transformed\Full Run/Inverse Transformed123.tif", bbox_inches='tight')
    plt.show()


    convertedImage = floatToImage(img_back)
    cv2.imshow("finalTransformed", convertedImage) 
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(exportDir + "/InvertedTransformConverted.tif", convertedImage)

def fourierHoughTransform(img, exportDir): 
    '''
    Method is for demo. Do not use otherwise. 
    '''
    rows, cols = img.shape 

    tfImage = DFTEstimate(img) 

    lines = getLFLines(tfImage, 210, 10)    #CHANGE THE 218 FOR DIFF IMAGES

    #CODE BLOCK TO SHOW LINES DRAWN 
    imageCopy = copy.copy(img) 
    drawHoughLines(imageCopy, lines)
    cv2.imshow("Lines found:", imageCopy)
    cv2.waitKey()
    cv2.destroyAllWindows()

    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) 

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])) 

    #Show original magnitude spectra
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Raw Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.savefig(exportDir + "/InputAndSpectrum.tif")
    plt.show()

    mask = np.zeros((rows, cols, 2), np.uint8)

    drawHoughLines(mask, lines) 

    fshift = dft_shift*mask 
    magnitude_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1])) 

    plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum Masked'), plt.xticks([]), plt.yticks([])
    plt.savefig(exportDir + "/Magnitude Spectrum Masked.tif")
    plt.show()

    #Invert back to spatial domain
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift) 

    imgConverted = np.uint8(18*np.log(img_back[:,:,0]))
    print(imgConverted)
    cv2.imshow("asdf", imgConverted)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(exportDir + "/NoMagnitudeAdjustment.tif", imgConverted)

    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    print(img_back) 
    # folderDir = createFolder(r"C:\Users\Irenaeus Wong\Desktop\DFT\Test Images\transformed", "/filteredImage3")

    plt.imshow(img_back, cmap= 'gray') 
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Inverse Transformed'), plt.xticks([]), plt.yticks([])
    # plt.savefig(r"C:\Users\Irenaeus Wong\Desktop\DFT\Test Images\transformed\Full Run/Inverse Transformed123.tif", bbox_inches='tight')
    plt.show()

    convertedImage = floatToImage(img_back)
    cv2.imshow("finalTransformed", convertedImage) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(exportDir + "/invertedFourierTransformed.tif", convertedImage)


# ------------------------- END OF FOURIER: BEGIN OPTICAL FLOW ------------------------- #

def videoToFrames(videoDir, frameWriteDir, resizeScale=None, imgType=".tif"):
    cap = cv2.VideoCapture(videoDir)
    frameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    counter = 1 

    for i in range(frameNum): 
        ret, currentFrame = cap.read() 
        if resizeScale != None: 
            currentFrame = resizeImage(currentFrame, resizeScale)
        cv2.imwrite(frameWriteDir + "\\Frames" + str(counter) + imgType, currentFrame)
        counter += 1 
        
def framesToVideo(framesDir, saveVideoDir_includeName, fps=15, start_num = 1, fileName="Frame", imgType=".tif", orderAtlas=False): 
    '''
    This function takes in a folder of images to convert to a .avi video. The video is then saved at a specified 
    directory. This function also returns the directory of the video. 

    Args: 

        (param1): 
            (str): Directory of folder containing images to be processed .
        (param2): 
            (str): Name of save directory, INCLUDING the video title and '.avi'. 
                   Example: "C:\\Users\\Irenaeus Wong\\Desktop\\test\\testVideo.avi" 
        (param3): 
            (int): Frames per second of output video. 
        (param4): 
            (int): Start index of the ending of image file names. 
        (param5): 
            (str): Precursor of image naming convention. Example: If images are named "thinSliceSESI_001" to "thinSliceSESI_0020", 
                   the fileName would be input as "thinSliceSESI_00". 
        (param6): 
            (str): Type of images to be processed. Examples: ".tif", ".tif", ".bmp" etc... 

    Returns: 

        (return1): 
            (str): Save directory of video. This return value is redundant as it is the same as parameter 2.  
    '''
    if orderAtlas == False: 
        frameList = orderFrames(framesDir, frameName=fileName, imgType=imgType, startNum=start_num)
    else: 
        frameList = orderFramesAtlas(framesDir)

    firstFrame = cv2.imread(frameList[0],0)  

    height, width = firstFrame.shape
    size = (width, height) 
    print(size)

    # Initialize video writer 
    out = cv2.VideoWriter(saveVideoDir_includeName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size) 
    for i in range(len(frameList)): 
        currentFrame = cv2.imread(frameList[i]) 
        out.write(currentFrame) 
    out.release() 

    return saveVideoDir_includeName


def drawVectorField(textPath, overlayedImage, vectorScale=1, finalImagesPath=None, finalImagesName=None, colour=VECTORFIELD_ARROWS): 
    '''
    Takes in one frame and text file containing vector information and a vector scaling factor to draw
	the vector field on a black background. 

	Args:
		param1: 
			(str): Path of text file containing vector data. 
		param2: 
			(np.ndarray): Image to be overlayed. 
		param3: 
			(int): Vector magnitudes to be scaled. Defaulted to 1. 
		param4: 
			(bool): whether to overlay vector field on top of image or not. Defaulted to not. 
		param5: 
			(str): Path of images to be saved. Defaulted to none. 
    
    Returns: 
        return1: 
            (np.ndarray): Vector Field Image (Overlayed if specified). 
    '''
    if colour==0:
        mask = np.zeros(overlayedImage.shape, dtype=np.uint8)
        vectorData = np.loadtxt(textPath)
        y = vectorData[:, 0] 
        x = vectorData[:, 1] 
        dy = vectorData[:, 2] 
        dx = vectorData[:, 3] 
        status = vectorData[:, 4]

        for i in range(len(y)):    
            if int(status[i]) == 0: 
                continue      
            mask = cv2.arrowedLine(mask, (int(y[i]), int(x[i])), ( int((y[i] + vectorScale*dy[i])) , int((x[i] + vectorScale*dx[i])) ), (0,0,255), thickness=1, tipLength=0.25)
    
    elif colour==1: 
        mask = np.zeros(overlayedImage.shape, dtype=np.uint8)
        vectorData = np.loadtxt(textPath)
        y = vectorData[:, 0] 
        x = vectorData[:, 1] 
        dy = vectorData[:, 2] 
        dx = vectorData[:, 3]
        status = vectorData[:, 4] 

        # Find absolute magnitudes of all vectors 
        mag = []
        ratio = []
        for i in range(len(dy)): 
            mag.append(np.sqrt(dy[i]**2 + dx[i]**2))

        # Append ratio values so range 0-255 
        for i in range(len(mag)): 
            ratio.append(255*(mag[i]/max(mag)))
        
        #Draw points as the mask
        for i in range(len(ratio)): 
            # print(int(np.around(255-ratio[i])))
            if int(status[i]) == 0: 
                continue
            mask = cv2.circle(mask, (int(y[i]), int(x[i]) ), 1, (0, int(np.around(255-ratio[i])), 255), -1)


    if finalImagesPath != None: 
        cv2.imwrite(finalImagesPath + "/" + finalImagesName + ".tif", mask)

    return mask
        
    # For now, no overlay capability. Just the vector map is enough. 
    # else: 
    #     image = overlayedImage
    #     vectorData = np.loadtxt(textPath)
    #     y = vectorData[:, 0] 
    #     x = vectorData[:, 1] 

    #     dy = vectorData[:, 2] 
    #     dx = vectorData[:, 3] 

    #     for i in range(len(y)):
    #         mask = cv2.arrowedLine(image, (int(y[i]), int(x[i])), ( int((y[i] + vectorScale*dy[i])) , int((x[i] + vectorScale*dx[i])) ), (0,0,255), thickness=1, tipLength=0.25)
    #     displayFrame = cv2.add(image, mask)

    #     # printImage("overlayed", displayFrame)

    #     return displayFrame


def opticalFlowFrames(framesPath, overlayMethod, fps=30, fileNames='Frames', imgType='.tif', vectorScale=20, harrisCorners=400, harrisQuality=0.01, harrisMinDistance=7, harrisBlockSize=7, lkWinSize=(15,15), lkMaxLevel=2, \
    lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), textDir=None, imgDir=None ): 
    '''
    Calculates and records coordinates and vectors of optical flow of key features found in all frames 
    found at specified framesPath. This function utilizes the ShiTomasi corner detection algorithm to find 
    key features to track using the Lucas Kanade Method of optical flow quantification.
    The Running this method displays the original video with the original video and vector fields overlayed. 
    '''

    feature_params = dict( maxCorners = harrisCorners,
                       qualityLevel = harrisQuality,
                       minDistance = harrisMinDistance,
                       blockSize = harrisBlockSize )
    lk_params = dict( winSize  = lkWinSize,
                    maxLevel = lkMaxLevel,
                    criteria = lkCriteria )

    # Initialize colors when overlayMethod == 0 or overlayMethod == 'linetrail' 
    # Need to include harrisCorners as number of colors generated. 
    color = np.random.randint(0,255,(harrisCorners,3))

    # This block is 'primitive' way to ensure the frames are ordered correctly. 
    # Will NEED to change later to make program more robust...
    framesOS = os.listdir(framesPath) 
    framesDir = [] 

    for i in range(len(framesOS)): 
        framesDir.append(framesPath + "/" + fileNames + str(i+1) + imgType) 
    
    # Take first frame and find corners of it (ShiTomasi corner detection)
    firstFrame = cv2.imread(framesDir[0]) 
    # Grayscale to find the corners. 
    oldGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    #Find points using goodFeaturesToTrack method of OpenCV 
    p0 = cv2.goodFeaturesToTrack(oldGray, mask=None, **feature_params) 

    #Initialize counter outside of the loop. 
    counter = 1  #This counts number of frames processed. 

    # Initialize 'mask' here to draw continuous line of paths 
    mask = np.zeros_like(firstFrame) 

    while(True):
        # Initialize data points/vectors to be collected.
        # Refreshes for each new frame.  
        points = [] 
        vectors = [] 

        # Exception caught if no more frames to load and process. 
        try:
            # Read current frame to process Optical Flow LK
            currentFrame = cv2.imread(framesDir[counter])
            currentGray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY) 
        except cv2.error:
            break 
        except IndexError: 
            break
        
        # Calculate Optical Flow: 
        p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, currentGray, p0, None, **lk_params)

        # Select and save the good points: 
        newPoints = p1[st==1] 
        oldPoints = p0[st==1] 

        # Find and save vector quantities of good points: 
        for i,(new, old) in enumerate(zip(newPoints, oldPoints)): 
            y2,x2 = new.ravel() 
            y1,x1 = old.ravel() 
            # Append point values
            points.append((y1, x1)) 
            # Calculate and append vector values
            vectors.append((y2-y1, x2-x1))  

            # Overlay mask drawing method: 'linetrail': 
            if overlayMethod == 0 or overlayMethod == 'linetrail':
                mask = cv2.line(mask, (y2,x2),(y1,x1), color[i].tolist(), 2)
                currentFrame = cv2.circle(currentFrame,(y2,x2),2,color[i].tolist(),-1)

            elif overlayMethod == 1 or overlayMethod == 'vectormap': 
                # Block to retrieve 
                mask = cv2.arrowedLine(mask, ( int(y1), int(x1) ), ( int(y1) + int(np.around(vectorScale*(y2-y1))), int(x1) + int(np.around(vectorScale*(x2-x1))) ), (0,0,255), thickness=1, tipLength=0.25)
                # currentFrame = cv2.circle(currentFrame, (y2,x2), 1, color[i].tolist(), -1) 
        
        # Add current frame and mask to create the final display frame. 
        displayFrame = cv2.add(currentFrame, mask) 
        # Remove old vectors drawn from previous frames.
        if overlayMethod == 1 or overlayMethod == 'vectormap': 
            mask = np.zeros_like(firstFrame)

        # Update previous frame and previous points: 
        oldGray = currentGray.copy() 
        p0 = newPoints.reshape(-1, 1, 2) 

        # Initialize textfile writing procedures: 
        if textDir != None: 
            textFilePath = textDir + "/Frame" + str(counter) + ".txt" 
            vectorFile = open(textFilePath, 'w') 

            # Write point and vector information to textfile 
            writeLines = []
            for i in range(len(points)): 
                writeLines.append(str(points[i][0]) + "\t" + str(points[i][1]) + "\t" + str(vectors[i][0]) + "\t" + str(vectors[i][1]) + "\t" + "0.000" + "\n") 
            vectorFile.writelines(writeLines)  
            vectorFile.close()
        
        if imgDir != None: 
            imgFilePath = imgDir + "/Frame" + str(counter) + ".tif"
            cv2.imwrite(imgFilePath, displayFrame) 

        counter += 1 

        waitTime = int(np.around(1000/fps)) 

        # Done writing textfiles. Display animation here: 
        if overlayMethod == 0 or overlayMethod == 'linetrail': 
            cv2.imshow('Frame with Line Trail Overlay', displayFrame)
            k = cv2.waitKey(waitTime) & 0xff 
            if k == 27: 
                break
        if overlayMethod == 1 or overlayMethod == 'vectormap': 
            cv2.imshow('Frame with Line Trail Overlay', displayFrame)
            k = cv2.waitKey(waitTime) & 0xff 
            if k == 27: 
                break
    
    cv2.destroyAllWindows()    
    

def opticalFlowVideo(videoPath, saveDir=None, vectorMaps=True, vectorScale=15, fps=5, harrisCorners=400, harrisQuality=0.01, harrisMinDistance=7, harrisBlockSize=7, lkWinSize=(15,15), lkMaxLevel=2, \
    lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), 
    vector_filtering=False, magnitude_threshold=0.1, median_threshold=1.5, 
    grid=False, grid_dimensions=40, initial_offset=10, select_points=False, select_lines=False, stabilizeDiagnostics=False): 
    '''
    Calculates and records coordinates and vectors of optical flow of key features found in the
    first frame of the video found at the videoPath. This function utilizes the ShiTomasi corner detection algorithm to find 
    key features to track using the Lucas Kanade Method of optical flow quantification.
    Running this method displays the original 
    video with the vector fields overlayed. 

    If saveDir=True, will create folders in saveDir for image frames, text files, and vectormap frames.

    Args:
        param1: 
            (str): Path to video to be processed. 
        param2: 
            (int): Number of corner features to track.
        
    '''
    if saveDir != None: 
        linetrail_dir = createFolder(saveDir, 'linetrail frames')
        vectorOverlay_dir = createFolder(saveDir, 'vector overlay frames')
        text_dir = createFolder(saveDir, 'text files') 
        if vector_filtering == True: 
            text_dir_filtered = createFolder(saveDir, 'text files filtered Mag={}_Med={}'.format(magnitude_threshold, median_threshold)) 

    cap = cv2.VideoCapture(videoPath)

    feature_params = dict( maxCorners = harrisCorners,
                       qualityLevel = harrisQuality,
                       minDistance = harrisMinDistance,
                       blockSize = harrisBlockSize )
    lk_params = dict( winSize  = lkWinSize,
                    maxLevel = lkMaxLevel,
                    criteria = lkCriteria )

    # Take first frame and find corners of it (ShiTomasi corner detection algorithm)
    ret, oldFrame = cap.read() 
    # Grayscale to find corners
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

    # Using ShiTomasi Corner Detection algorithm to select points 
    if ( (grid == False) and (select_points == False) and (select_lines == False) ): 
        p0 = cv2.goodFeaturesToTrack(oldGray, mask=None, **feature_params)
        print("Selecting ShiTomasi features...")

    # Selecting points with GUI: 
    elif ((grid == False) and (select_points == True) and (select_lines == False)):
        print("Selecting points through GUI...")
        ret, firstFrame = cap.read() 
        p0 = chooseMultiplePoints(firstFrame, drag_lines=False, wrap_for_cv2=True) 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Select lines with GUI: 
    elif ( (grid == False) and (select_points == False) and (select_lines == True) ):
        print("Selecting lines through GUI...")
        ret, firstFrame = cap.read() 
        p0 = chooseMultiplePoints(firstFrame, drag_lines=True, wrap_for_cv2=True) 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Creating grid 
    else: 
        p0 = getGridPoints(oldGray, grid_dimensions, initial_offset, evenGrid=False, wrap_for_cv2=True)


    new_p0 = np.zeros((len(p0), 1, 5), dtype=np.float32)

    color = np.random.randint(0,255,(len(p0),3))

    for i in range(len(p0)):
        new_p0[i][0] = np.append(p0[i][0], color[i]) 

    #Initialize counter outside of loop 
    counter = 1 

    #initialize 'mask' here to draw continuous line of paths 
    linetrail_mask = np.zeros_like(oldFrame)
    vector_mask = np.zeros_like(oldFrame) 

    print("Prerequisites initialized. Drawing Overlayed Vectormaps:")

    while(True): 
        # Data to be collected for each frame
        points = []  
        vectors = []

        # Frame read to draw linetrail overlay 
        ret, linetrail_frame = cap.read() 
        # Copy frame to draw vector overlay
        try: 
            vector_frame = linetrail_frame.copy() 
        except AttributeError: 
            pass

        # Create fresh mask every iteration --> only for vector overlay: 
        vector_mask = np.zeros_like(linetrail_frame) 

        # Breaks when no more frames to write, frame==None 
        try:
            linetrail_frameGray = cv2.cvtColor(linetrail_frame, cv2.COLOR_BGR2GRAY)
            vector_frameGray = cv2.cvtColor(vector_frame, cv2.COLOR_BGR2GRAY)            
        except cv2.error: 
            break 

        # Initialize proper formatting of new_p0 to throw into calcOpticalFlowPyrLK
        pythonList = [] 
        for i in range(len(new_p0)): 
            pythonList.append([])
            pythonList[i].append([]) 
            for j in range(2): 
                pythonList[i][0].append(new_p0[i][0][j])
        
        opticalFlow_p0 = np.asarray(pythonList) 
        
        # Calc opticalflow once because linetrail_frameGray and vector_frameGray are identical as of now...
        if stabilizeDiagnostics == False: 
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, linetrail_frameGray, opticalFlow_p0, None, **lk_params) 
        else: 
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, linetrail_frameGray, opticalFlow_p0, None, maxLevel=lkMaxLevel) 

        # Add colour values to p1 from new_p0 
        pythonList = [] 
        for i in range(len(p1)): 
            pythonList.append([])
            pythonList[i].append([]) 
            for j in range(2): 
                pythonList[i][0].append(p1[i,0,j])
            for k in range(3): 
                    pythonList[i][0].append(new_p0[i, 0, k+2]) 
        new_p1 = np.asarray(pythonList) 

        #Select and save good points 
        newPoints = new_p1[st==1] 
        try: 
            oldPoints = new_p0[st==1] 
        except IndexError: 
            print("Index Error in oldPoints assignment.")
            pass 
            
        #Find and save vector quantities of good points: 
        for i,(new, old) in enumerate(zip(newPoints, oldPoints)):
            y2,x2, B, G, R = new.ravel() 
            y1,x1, B, G, R = old.ravel() 
            points.append((y1, x1)) 
            #calculate and append vector values...
            vectors.append((y2-y1, x2-x1))  

            # Data processing and acquisition complete. 
            # Time to visualize by drawing arrows on mask 
            # mask = cv2.arrowedLine(mask, (y1, x1), (y2, x2), (0,0,255), thickness=3)

            # Draw points moving per frame in color. 
            linetrail_mask = cv2.line(linetrail_mask, (y2,x2),(y1,x1), (int(B), int(G), int(R)) , 2)
            linetrail_frame = cv2.circle(linetrail_frame,(y2,x2),2, (int(B), int(G), int(R)) ,-1)

            vector_mask = cv2.arrowedLine(vector_mask, ( int(y1), int(x1) ), ( int(y1) + int(np.around(vectorScale*(y2-y1))), int(x1) + int(np.around(vectorScale*(x2-x1))) ), (0,0,255), thickness=1, tipLength=0.25)

            
        linetrail_frame = cv2.add(linetrail_frame, linetrail_mask) 
        vector_frame = cv2.add(vector_frame, vector_mask)


        # Now update the previous frame and previous points
        oldGray = linetrail_frameGray.copy()
        # -1 in reshape() needed to update number of valid points! 
        new_p0 = newPoints.reshape(-1,1,5)
            
        # Initialize textfile writing procedures
        #  
        if saveDir != None: 
            textFilePath = text_dir + "/Frame" + str(counter) + ".txt" 
            vectorFile = open(textFilePath, 'w') 

            # Write point and vector information to textfile 
            writeLines = []
            for i in range(len(points)): 
                writeLines.append(str(points[i][0]) + "\t" + str(points[i][1]) + "\t" + str(vectors[i][0]) + "\t" + str(vectors[i][1]) + "\t" + "1.000" + "\n") 
            vectorFile.writelines(writeLines)  
            vectorFile.close()
        
        # Write image frames here: 
        if saveDir != None: 
            # Write linetrail images here
            imgFilePath = linetrail_dir + "/Frame" + str(counter) + ".tif"
            cv2.imwrite(imgFilePath, linetrail_frame) 
            # Write vectormap overlayed images here 
            imgFilePath = vectorOverlay_dir + "/Frame" + str(counter) + ".tif" 
            cv2.imwrite(imgFilePath, vector_frame) 

        counter += 1 

        #Done writing text and image frames stuff. Display animation now: 
        cv2.imshow('frame', linetrail_frame) 
        k = cv2.waitKey(30) & 0xff 
        if k == 27: 
            break 
        
        print("Frame Pairs {}/{} processed...".format(counter-1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)) )

    cv2.destroyAllWindows() 

    # Apply magnitude filtering and median filtering before drawing vectorfields 
    if vector_filtering == True: 
        textFiles = orderFrames(text_dir, frameName='Frame', imgType='.txt') 
        # Iterate through all textfiles 
        filtered_vectors = []   # For now, this variable is not used. Can delete. It stores vector data as 2D array. 
        for i in range(len(textFiles)): 
            filtered_vectors.append(filterVectors(textFiles[i], saveDir_withName="{}/Frame{}.txt".format(text_dir_filtered, i+1), window_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
                , window_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), magnitude_threshold=magnitude_threshold, median_threshold=median_threshold )) 


    print("\n\n\nInitializing Vectormap Drawing and Saving:")
    # Draw vectorfield here and save to assigned directories 
    if vectorMaps == True: 
        vectormap_dir = createFolder(saveDir, 'vectormap frames')
        # dotmap_dir = createFolder(saveDir, 'dotmap frames')
        
        if vector_filtering == False: 
            textFiles = orderFrames(text_dir, frameName='Frame', imgType='.txt') 
        else: 
            textFiles = orderFrames(text_dir_filtered, frameName='Frame', imgType='.txt')
        firstFrame = cv2.imread(linetrail_dir + "/Frame1.tif") 

        print(textFiles)
        for i in range(len(textFiles)): 
            # Draw and save arrow vector map 
            arrow_frame = drawVectorField(textFiles[i], firstFrame, vectorScale=vectorScale, colour=VECTORFIELD_ARROWS)
            cv2.imshow('Vector Field', arrow_frame) 
            cv2.imwrite(vectormap_dir + "/Frame" + str(i+1) + ".tif", np.uint8(np.around(arrow_frame))) 
            k = cv2.waitKey(30) & 0xff
            if k == 27: 
                break 
            # Draw and save dots vector map 
            # dot_frame = drawVectorField(textFiles[i], firstFrame, vectorScale=vectorScale, colour=VECTORFIELD_DOTS)
            # cv2.imwrite(dotmap_dir + "/Frame" + str(i+1) + ".tif", np.uint8(np.around(dot_frame))) 
            print("Frame Pairs {}/{} processed...".format((i+1), len(textFiles)) )
        cv2.destroyAllWindows()
    
        # Write all frame data to videos... 
        print("Frames all saved. Saving as videos...")
        video_dir = createFolder(saveDir, 'Videos')
        framesToVideo(linetrail_dir, saveVideoDir_includeName="{}/linetrail.avi".format(video_dir), fps=fps ) 
        print("Linetrail Video saved.")
        framesToVideo(vectorOverlay_dir, saveVideoDir_includeName="{}/vectorOverlay.avi".format(video_dir), fps=fps ) 
        print("Vectors Overlayed Video saved.")
        framesToVideo(vectormap_dir, saveVideoDir_includeName="{}/vectormap.avi".format(video_dir), fps=fps) 
        print("Vectormap Video saved.")
        # framesToVideo(dotmap_dir, saveVideoDir_includeName="{}/dotmap.avi".format(video_dir), fps=fps) 
        # print("Dotmap Video saved.")
        print("BYE BYEEE JOOJOOOOOOO")


def filterVectors(textDir, saveDir_withName, window_height, window_width, magnitude_threshold, median_threshold=None): 
    '''
    Opens one textfile with vector format "x \t y \t dx \t dy \t status \n" and applies a 
    magnitude filter with specified threshold. Threshold is relative to window size. If 
    threshold is 0.1, only dx dy magnitudes within 10% of window size. If median_threshold 
    is initialized, a median filter will be applied to change the status of the vectors. 
    '''
    textFile = np.loadtxt(textDir) 

    y_threshold = window_height*magnitude_threshold 
    x_threshold = window_width*magnitude_threshold 

    # Apply magnitude_threshold filter 
    for i in range(len(textFile)): 
        if (np.abs(textFile[i,2]) < np.abs(x_threshold)) and (np.abs(textFile[i,3]) < np.abs(y_threshold)): 
            textFile[i, 4] = 1 
        else: 
            textFile[i, 4] = 0 
    
    # Apply median_threshold filter if threshold value is available 
    if median_threshold != None: 
        abs_vectors = [] 
        all_medians = np.median(np.absolute(textFile), axis=0) 
        x_median = all_medians[2] 
        print("x_median: {}".format(x_median))
        y_median = all_medians[3]
        print("y_median: {}".format(y_median)) 

        x_threshold = x_median*median_threshold 
        y_threshold = y_median*median_threshold 

        for i in range(len(textFile)): 
            if (np.abs(textFile[i,2]) < np.abs(x_threshold)) and (np.abs(textFile[i,3]) < np.abs(y_threshold)): 
                textFile[i, 4] = 1 
            else: 
                textFile[i, 4] = 0 
    
    # Write data and return data: 
    writeVectors = open(saveDir_withName, 'w') 
    for i in range(len(textFile)): 
        writeVectors.writelines("{}\t{}\t{}\t{}\t{}\n".format(textFile[i,0],textFile[i,1], textFile[i,2], textFile[i,3], textFile[i,4]) )
        
    writeVectors.close() 
    
    # Return the array of textFiles with filtered status markers 
    return textFile 

def vectorFieldAnimation(folderPath, originalImage, fileNames, fps=30, vectorScale=15, imgWriteDir=None, overlayMethod=VECTORFIELD_ARROWS): 
    '''
    This function takes in a directory of text files generated from the opticalFlowVideo method, and 
    displays an animation of a vector field generated by the Lucas Kanade optical flow algorithm with 
    Harris corner detections. This method can also save textfiles and images with vector field data to user-specified
    directories.
    
    Args: 
        param1: 
            (str): Path of folder containing text files of vector data to be read. 
        param2: 
            (np.ndarray): Original Image
        param3: 
            (str): Directory to save vector field frames. 
    '''
    textFiles = os.listdir(folderPath) 
    textFileNames = orderFrames(folderPath, frameName='Frame', imgType='.txt')

    waitTime = int(np.around(1000/fps))
    # for i in range(len(textFiles)): 
    #     textFileNames.append(folderPath + "/" + fileNames + str(i+1) + ".txt") 

    if overlayMethod==0: 
        for i in range(len(textFileNames)): 
            drawFrame = drawVectorField(textFileNames[i], originalImage, vectorScale, colour=VECTORFIELD_ARROWS)
            cv2.imshow('Vector Field', drawFrame) 
            if imgWriteDir != None: 
                cv2.imwrite(imgWriteDir + "/Frame" + str(i+1) + ".tif", np.uint8(np.around(drawFrame)))
            k = cv2.waitKey(waitTime) & 0xff
            if k == 27: 
                break 

    if overlayMethod==1: 
        for i in range(len(textFileNames)): 
            drawFrame = drawVectorField(textFileNames[i], originalImage, vectorScale, colour=VECTORFIELD_DOTS)
            cv2.imshow('Vector Field', drawFrame) 
            if imgWriteDir != None: 
                cv2.imwrite(imgWriteDir + "/Frame" + str(i+1) + ".tif", np.uint8(np.around(drawFrame)))
            k = cv2.waitKey(waitTime) & 0xff
            if k == 27: 
                break 
    # else: 
    #     for i in range(len(textFileNames)): 
    #         drawFrame = drawVectorField(textFileNames[i], originalImage, vectorScale=20, overlay=True)
    #         cv2.imshow('Vector Field', drawFrame) 
    #         k = cv2.waitKey(30) & 0xff
    #         if k == 27: 
    #             break 
            

def playFrames(folderDir, fps, fileNames='Frames', imgType=".tif"): 
    '''
    This method takes in a folder directory which contains a series of images following the naming 
    convention "Frame[i]", and displays them at a specified frames per second (fps). 

    Args: 
        param1: 
            (str): Directory path for images to be played. 
        param2:
            (int): Frames Per Second 
        param3: 
            (str): Image file type, defaulted to ".tif" images. 
    '''
    wait = int(np.around(1000/fps))

    images = os.listdir(folderDir) 
    
    imageFileNames = [] 

    for i in range(len(images)): 
        imageFileNames.append(folderDir + "/" + fileNames + str(i+1) + imgType) 
    
    for i in range(len(imageFileNames)): 
        try:
            cv2.imshow('Frames at ' + str(fps) + ' frames per second', cv2.imread(imageFileNames[i]) )
        except cv2.error: 
            break
        k = cv2.waitKey(wait) & 0xff 
        if k == 27: 
            break

def movingAverage(curve, radius): 
    window_size = 2 * radius + 1
    # Define the filter 
    f = np.ones(window_size)/window_size 
    # Add padding to the boundaries 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    # Remove padding 
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory, smoothing_radius=40): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=smoothing_radius)
 
  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

def stabilizeVideoSmoothControl(videoDir, outputFileNames, saveDir=None, graphFrames=(None, None), comparison=False, smoothing=False, smoothing_radius=40, fps=15 
    , lkWinSize=(15,15), lkMaxLevel=3, lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), tomasiCorners=2000, RANSACThreshold=3
    , show=False, saveDiagnosticData=True, diagnosticOverlay=True, overlayRightShift=True, affinePartial=False, frameMethod=FRAMECOMBINATION 
    , minEigenValue=None, maxRefinement=0):

    '''
    July 31st 2019
    This function is written to test the effects of kalman filtering, or more specifically, 
    the effects of stabilizing an image stack without the use of kalman filtering or smoothening. 
    Instead of estimating the affine transform matrices of a frame pair, smoothening, and applying 
    to the first frame of the pair, the affine transform matrix will be estimated, inversed and 
    applied to the second frame.

    '''

    # Create folder for optical flow diagnostic data 
    if saveDiagnosticData == True: 
        diagnostics = createFolder(saveDir, 'diagnostics') 
        optical_flow_dir = createFolder(diagnostics, 'optical flow') 
        estimated_affine = open("{}\\estimated_affine.txt".format(diagnostics), "w") 
        inversed_affine = open("{}\\inverse_affine.txt".format(diagnostics), "w")  
        confidence_text = open("{}\\confidence measure.txt".format(diagnostics), "w") 
        optical_flow_error_text = open("{}\\optical_flow_error.txt".format(diagnostics), "w") 
    
    frames_location = createFolder(saveDir, 'frames') 

    # Save matrices anyways for simplicity 
    estimated_affine_matrices = [] 
    inversed_affine_matrices = []

    # Initialize confidence levels per frame pair optical flow calc
    confidence = []  
    confidence_good = [] 
    confidence_all = [] 
    error = [] 

    waitTime = int(np.around(1000/fps))
    # if saveDir != None: 
        # outfile = open(saveDir + outputFileNames + ".txt", 'w') 

    # Read input video
    cap = cv2.VideoCapture(videoDir)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # Set up videowriter object with appropriate widths 
    if (comparison==False) and (saveDir != None): 
    # Set up output video
        out = cv2.VideoWriter("{}\\{}.avi".format(saveDir, outputFileNames), fourcc, fps, (w,h))
    elif saveDir != None: 
        write_w = int(np.around(w/2)) 
        write_h = int(np.around(h/2)) 
        out = cv2.VideoWriter("{}\\{}.avi".format(saveDir, outputFileNames), fourcc, fps, (int(write_w*2), write_h)) 
    if frameMethod == FRAMEBYFIRST: 
        # Save first image
        _, first_image = cap.read()

        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY) 

        # Calculate good points to track in first frame 
        first_pts = cv2.goodFeaturesToTrack(first_image,
                                        maxCorners=tomasiCorners,
                                        qualityLevel=0.01,
                                        minDistance=20,
                                        blockSize=3)


    elif frameMethod == FRAMEBYFRAME: 
        _, prev = cap.read()
        # Convert frame to grayscale
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    elif frameMethod == FRAMECOMBINATION: 
        _, prev = cap.read() 
        # Convert frame to grayscale
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

        # Use this to store accumulated transform matrices to save computing time 
        accumulated = np.eye(3)  

    else: 
        print("u messed up bad") 
        return None 


    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 6), np.float32) 

    # Pre-define inversed transforms for second frame
    transforms_inverse = np.zeros( (n_frames-1, 6), np.float32) 
    
    for i in range(n_frames-1):

        if frameMethod == FRAMEBYFRAME: 
            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                        maxCorners=tomasiCorners,
                                        qualityLevel=0.01,
                                        minDistance=20,
                                        blockSize=3)
        
            # Read next frame
            success, curr = cap.read() 
            if not success: 
                break

            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

            # Calculate optical flow (i.e. track feature points)
            if minEigenValue == None: 
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS) #, **lk_params)
            else:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=minEigenValue)  
            # Sanity check
            # Check that number of previous points is equal to current points 
            assert prev_pts.shape == curr_pts.shape 

            confidence_all.append(len(curr_pts)) 

            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            confidence_good.append(len(curr_pts)) 
        
            #Find transformation matrix
            if affinePartial == True: 
                m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            else:
                # Get Shear, Scale, and translation 
                m = cv2.estimateAffine2D(prev_pts, curr_pts) 

            # Save each estimated affine matrix. 
            # Index [0] needed due to function formatting( we only want the numpy array, nothing else ) 
            estimated_affine_matrices.append(m[0]) 

            #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

            #Extract affine transform matrix members 
            zero_zero = m[0][0,0]
            zero_one = m[0][0,1] 
            one_zero = m[0][1,0] 
            one_one = m[0][1,1] 
            dx = m[0][0,2] 
            dy = m[0][1,2] 

            # Store Shear, Scale and Translation transformation matrix 
            # Subtract 1 from scaling terms so scaling factors accurately represented 
            # Note: Subtracting 1 isn't useful in this function. The subtracting of the 
            # identity matrix serves only to throw to a matrix value filter (kalman filter...) 
            transforms[i] = [zero_zero-1, zero_one, dx, one_zero, one_one-1, dy]

            temporary_matrix = np.empty( (3,3), dtype=np.float32) 

            # Reconstruct current matrix as a 3x3 homogenous affine matrix 
            # Note: we don't want to subtract from zero_zero and one_one so we can 
            # compute the inverse matrix properly 
            temporary_matrix[0] = [zero_zero, zero_one, dx] 
            temporary_matrix[1] = [one_zero, one_one, dy] 
            temporary_matrix[2] = [0, 0, 1] 

            inverse = np.linalg.inv(temporary_matrix) 

            transforms_inverse[i] = [inverse[0,0], inverse[0,1], inverse[0,2], inverse[1,0], inverse[1,1], inverse[1,2]] 

            # Move to next frame
            prev_gray = curr_gray
        
            print("Frame Pairs: " + str(i+1) +  "/" + str(n_frames-1) + " -  Tracked points : " + str(len(prev_pts)))
        
        elif frameMethod == FRAMEBYFIRST: 
            
            # No need to detect features of previous frames...already saved in first_points variable 
            # Read next frame to get optical flow from 
            success, curr = cap.read() 
            if not success: 
                break 
            
            # Convert to grayscale 
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(first_image, curr_gray, first_pts, None, maxLevel=lkMaxLevel) #, **lk_params) 
            # Sanity check
            # Check that number of previous points is equal to current points 
            assert first_pts.shape == curr_pts.shape 

            # Filter only valid points
            idx = np.where(status==1)[0]
            first_pts = first_pts[idx]
            curr_pts = curr_pts[idx]
        
            #Find transformation matrix
            if affinePartial == True: 
                m = cv2.estimateAffinePartial2D(first_pts, curr_pts)
            else:
                # Get Shear, Scale, and translation 
                m = cv2.estimateAffine2D(first_pts, curr_pts) 

            # Save each estimated affine matrix. 
            # Index [0] needed due to function formatting( we only want the numpy array, nothing else ) 
            estimated_affine_matrices.append(m[0]) 

            #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

            #Extract affine transform matrix members 
            zero_zero = m[0][0,0]
            zero_one = m[0][0,1] 
            one_zero = m[0][1,0] 
            one_one = m[0][1,1] 
            dx = m[0][0,2] 
            dy = m[0][1,2] 

            temporary_matrix = np.empty( (3,3), dtype=np.float32) 

            # Reconstruct current matrix as a 3x3 homogenous affine matrix 
            # Note: we don't want to subtract from zero_zero and one_one so we can 
            # compute the inverse matrix properly 
            temporary_matrix[0] = [zero_zero, zero_one, dx] 
            temporary_matrix[1] = [one_zero, one_one, dy] 
            temporary_matrix[2] = [0, 0, 1] 

            inverse = np.linalg.inv(temporary_matrix) 

            transforms_inverse[i] = [inverse[0,0], inverse[0,1], inverse[0,2], inverse[1,0], inverse[1,1], inverse[1,2]] 

            # Move to next frame
            prev_gray = curr_gray
        
            print("Frame Pairs: " + str(i+1) +  "/" + str(n_frames-1) + " -  Tracked points : " + str(len(first_pts)))

        elif frameMethod == FRAMECOMBINATION: 

            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                        maxCorners=tomasiCorners,
                                        qualityLevel=0.01,
                                        minDistance=20,
                                        blockSize=3)
        
            # Read next frame
            success, curr = cap.read() 
            if not success: 
                break

            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

            if minEigenValue == None: 
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS) #, **lk_params)
            else:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=minEigenValue)  
            # Sanity check

            avg_error = np.sum(err[:,0])/err.shape[0]
            error.append(avg_error)  
            # Sanity check
            # Check that number of previous points is equal to current points 
            assert prev_pts.shape == curr_pts.shape 

            confidence_all.append(len(curr_pts)) 

            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            confidence_good.append(len(curr_pts)) 

            confidence.append(confidence_good[i]/confidence_all[i])

            #Find transformation matrix
            if affinePartial == True: 
                m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            else:
                # Get Shear, Scale, and translation 
                m = cv2.estimateAffine2D(prev_pts, curr_pts, ransacReprojThreshold=RANSACThreshold, refineIters=maxRefinement)  

            # Save each estimated affine matrix. 
            # Index [0] needed due to function formatting( we only want the numpy array, nothing else ) 
            estimated_affine_matrices.append(m[0]) 

            #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

            #Extract affine transform matrix members 
            zero_zero = m[0][0,0]
            zero_one = m[0][0,1] 
            one_zero = m[0][1,0] 
            one_one = m[0][1,1] 
            dx = m[0][0,2] 
            dy = m[0][1,2] 

            # Store Shear, Scale and Translation transformation matrix 
            # Subtract 1 from scaling terms so scaling factors accurately represented 
            # Note: Subtracting 1 isn't useful in this function. The subtracting of the 
            # identity matrix serves only to throw to a matrix value filter (kalman filter...) 
            transforms[i] = [zero_zero-1, zero_one, dx, one_zero, one_one-1, dy]

            temporary_matrix = np.empty( (3,3), dtype=np.float32) 

            # Reconstruct current matrix as a 3x3 homogenous affine matrix 
            # Note: we don't want to subtract from zero_zero and one_one so we can 
            # compute the inverse matrix properly 
            temporary_matrix[0] = [zero_zero, zero_one, dx] 
            temporary_matrix[1] = [one_zero, one_one, dy] 
            temporary_matrix[2] = [0, 0, 1] 

            inverse = np.linalg.inv(temporary_matrix) 

            # Compute the accumulation transform matrix 
            accumulated = np.dot(inverse, accumulated) 

            transforms_inverse[i] = [accumulated[0,0], accumulated[0,1], accumulated[0,2], accumulated[1,0], accumulated[1,1], accumulated[1,2]] 

            # Move to next frame
            prev_gray = curr_gray
        
            print("Frame Pairs: " + str(i+1) +  "/" + str(n_frames-1) + " -  Tracked points : " + str(len(prev_pts)))

        
    if affinePartial == False: 
        # print("Trajectories, x, y and a: {}".format(trajectory)) 
        # Add plot for confidence level 
        if graphFrames[0] == None: 
            # Start subplot of dx, dy, and da's 
            fig, ax = plt.subplots(2,8, figsize=(60,20))  
            # plt.setp(ax, xticks=[])
            ax[0, 0].plot(transforms[:,0])
            ax[0, 0].set_title('transforms[0,0]') 
            # ax[0, 0].set_ylabel('pixels/dt') 
            ax[0, 0].set_xlabel('frames') 

            ax[0, 1].plot(transforms[:,1])
            ax[0, 1].set_title('transforms[0,1]') 
            # ax[0, 1].set_ylabel('pixels/dt') 
            ax[0, 1].set_xlabel('frames')

            ax[0, 2].plot(transforms[:,2])
            ax[0, 2].set_title('original dx') 
            ax[0, 2].set_ylabel('pixels/dt') 
            ax[0, 2].set_xlabel('frames') 

            ax[0, 3].plot(transforms[:,3])
            ax[0, 3].set_title('transforms[1,0]') 
            # ax[0, 3].set_ylabel('pixels/dt') 
            ax[0, 3].set_xlabel('frames')

            ax[0, 4].plot(transforms[:,4])
            ax[0, 4].set_title('transforms[1,1]') 
            # ax[0, 4].set_ylabel('pixels/dt') 
            ax[0, 4].set_xlabel('frames')

            ax[0, 5].plot(transforms[:,5])
            ax[0, 5].set_title('original dy') 
            ax[0, 5].set_ylabel('pixels/dt') 
            ax[0, 5].set_xlabel('frames')

            ax[0, 6].plot(confidence_good, label="Number of Good Points", color='red') 
            ax[0, 6].plot(confidence_all, label="Total Number of Points", color='green') 
            ax[0, 6].legend() 
            ax[0, 6].set_title("Confidence Metric Point Count") 
            ax[0, 6].set_xlabel("frame pairs") 
            ax[0, 6].set_ylabel("Number of Points") 

            ax[0,7].plot(error) 
            ax[0,7].set_title("Error Measure of Optical Flow") 
            ax[0,7].set_xlabel("frame pairs") 
            ax[0,7].set_ylabel("average O-F error per frame") 


            # Draw inversed transforms
            ax[1,0].plot(transforms_inverse[:,0], color='purple')
            ax[1,0].set_title('inversed transforms[0,0]') 
            # ax[1,0].set_ylabel('pixels') 
            ax[1,0].set_xlabel('frames')

            ax[1,1].plot(transforms_inverse[:,1], color='purple')
            ax[1,1].set_title('inversed transforms[0,1]') 
            # ax[1,1].set_ylabel('pixels') 
            ax[1,1].set_xlabel('frames')

            ax[1,2].plot(transforms_inverse[:,2], color='purple')
            ax[1,2].set_title('inversed translation x')
            ax[1,2].set_ylabel('pixels') 
            ax[1,2].set_xlabel('frames')

            ax[1,3].plot(transforms_inverse[:,3], color='purple')
            ax[1,3].set_title('inversed transforms[1,0]')
            # ax[1,3].set_ylabel('radians') 
            ax[1,3].set_xlabel('frames')

            ax[1,4].plot(transforms_inverse[:,4], color='purple')
            ax[1,4].set_title('inversed transforms[1,1]')
            # ax[1,4].set_ylabel('radians') 
            ax[1,4].set_xlabel('frames')

            ax[1,5].plot(transforms_inverse[:,5], color='purple')
            ax[1,5].set_title('inversed translation y')
            ax[1,5].set_ylabel('pixels') 
            ax[1,5].set_xlabel('frames')

            ax[1,6].plot(confidence, color='cyan') 
            ax[1,6].set_title("Confidence Metric Per Frame Pair")
            ax[1,6].set_xlabel("frame pairs") 
            ax[1,6].set_ylabel("Confidence Metric (larger is better)") 

            # Save figure into save directory specified in parameters. 
            fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + '.png')

            plt.close(fig) 

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    success, first_frame = cap.read() 
    out.write(first_frame) 
    cv2.imwrite("{}\\Frame{}.tif".format(frames_location, 1), first_frame) 
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-1):
        # Read next frame
        success, frame = cap.read() 
        if not success:
            break
        

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = transforms_inverse[i,0]      
        m[0,1] = transforms_inverse[i,1] 
        m[0,2] = transforms_inverse[i,2] 
        m[1,0] = transforms_inverse[i,3]
        m[1,1] = transforms_inverse[i,4]   
        m[1,2] = transforms_inverse[i,5] 
        
        # Save inversed affine matrices 
        inversed_affine_matrices.append(m) 

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h) )
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            frame_out = frame_stabilized 
            # Writing diagnostic data on top 
            if diagnosticOverlay == True: 
                height, width, dim = frame_out.shape
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

                # Right shift the matrix overlay
                if overlayRightShift == True: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (width-1500, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (width-1500, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[inversed m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (width-750, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (width-750, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                # Left shift the matrix overlay
                else: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (200, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (200, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[inversed m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (950, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (950, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

            # Save overlayed frame
            cv2.imwrite("{}\\Frame{}.tif".format(frames_location, i+2), frame_out) # i + 2 bc not including first frame
            if show == True: 
                cv2.imshow("Stabilized Frame", frame_out)
                cv2.waitKey(waitTime) 
            out.write(frame_out)
            
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            # if (frame_out.shape[1] > 2980): 
            frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) )) 

            # Writing diagnostic data on top 
            if diagnosticOverlay == True: 
                height, width, dim = frame_out.shape
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

                # Right shift the matrix overlay
                if overlayRightShift == True: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (width-1500, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (width-1500, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[inversed m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (width-750, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (width-750, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                # Left shift the matrix overlay
                else: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (200, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (200, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[inversed m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (950, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (950, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

            # Save overlayed frame
            cv2.imwrite("{}\\Frame{}.tif".format(frames_location, i+2), frame_out)  # i+2 bc not including first frame 

            if show == True: 
                cv2.imshow("Stabilized Frame", frame_out)
                cv2.waitKey(waitTime) 
            out.write(frame_out)

    # Save diagnostic data 
    if saveDiagnosticData == True: 

        confidence_lines = [] 
        for i in range(len(confidence)): 
            confidence_lines.append("FramePair: {}...\t{}\t{}\t{}\n".format(i+1, confidence_good[i], confidence_all[i], confidence[i])) 
        confidence_text.writelines(confidence_lines) 
        confidence_text.close() 

        error_lines = [] 
        for i in range(len(error)): 
            error_lines.append("FramePair: {}...\t{}\n".format(i+1, error[i]))
        optical_flow_error_text.writelines(error_lines)

        for i,curr_matrix in enumerate(estimated_affine_matrices): 
            # Write matrix data in following format 
            estimated_affine.write( "FramePair: {}...\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1, curr_matrix[0,0], curr_matrix[0,1], curr_matrix[0,2]
                , curr_matrix[1,0], curr_matrix[1,1], curr_matrix[1,2]) ) 
        estimated_affine.close()

        for i,curr_matrix in enumerate(inversed_affine_matrices): 
            # Write matrix data in following format 
            inversed_affine.write( "FramePair: {}...\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1, curr_matrix[0,0], curr_matrix[0,1], curr_matrix[0,2]
                , curr_matrix[1,0], curr_matrix[1,1], curr_matrix[1,2]) ) 
        inversed_affine.close() 

        # Block to calculate optical flow with same parameters as acquisition... 
        opticalFlowVideo(videoDir, optical_flow_dir, harrisCorners=tomasiCorners, lkWinSize=lkWinSize, lkMaxLevel=lkMaxLevel, lkCriteria=lkCriteria
            , stabilizeDiagnostics=True, vectorScale=1)
        

    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 


def stabilizeVideo(videoDir, outputFileNames, saveDir=None, graphFrames=(None, None), comparison=False, smoothing_radius=40, fps=15 \
    , lkWinSize=(15,15), lkMaxLevel=2, lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), tomasiCorners=1000):

    waitTime = int(np.around(1000/fps))
    # if saveDir != None: 
        # outfile = open(saveDir + outputFileNames + ".txt", 'w') 
    

    # Read input video
    cap = cv2.VideoCapture(videoDir)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # Set up videowriter object with appropriate widths 
    if (comparison==False) and (saveDir != None): 
    # Set up output video
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames + "_Win=" + str(lkWinSize[0]) + "_R=" + str(smoothing_radius) + ".avi", fourcc, fps, (w, h))
    elif saveDir != None: 
        write_w = int(np.around(w/2)) 
        write_h = int(np.around(h/2)) 
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames + "_Comparison_Win=" + str(lkWinSize[0]) + "_R=" + str(smoothing_radius) + ".avi", fourcc, fps, (int(write_w*2), write_h))

    _, prev = cap.read() 
    
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 

    for i in range(n_frames-1):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=tomasiCorners,
                                     qualityLevel=0.01,
                                     minDistance=20,
                                     blockSize=3)
        # lk_params = dict( winSize  = lkWinSize,
        #             maxLevel = lkMaxLevel,
        #             criteria = lkCriteria )
    
        # Read next frame
        success, curr = cap.read() 
        if not success: 
            break
 
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
    
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) #, **lk_params) 
    
        # Sanity check
        # Check that number of previous points is equal to current points 
        assert prev_pts.shape == curr_pts.shape 
    
        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
    
        #Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

        # Extract traslation
        dx = m[0][0,2] 
        dy = m[0][1,2] 

    
        # Extract rotation angle
        da = np.arctan2(m[0][1,0], m[0][0,0])
    
        # Store transformation
        transforms[i] = [dx,dy,da]
        
        # Move to next frame
        prev_gray = curr_gray
    
        print("Frame Pairs: " + str(i+2) +  "/" + str(n_frames-2) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 

    if graphFrames[0] == None: 
        # Start subplot of dx, dy, and da's 
        fig, ax = plt.subplots(2,3, figsize=(40,15), sharex=True)  
        plt.setp(ax, xticks=[])
        ax[0, 0].plot(transforms[:,0])
        ax[0, 0].set_title('original dx') 

        ax[0, 1].plot(transforms[:,1])
        ax[0, 1].set_title('original dy') 

        ax[0, 2].plot(transforms[:,2])
        ax[0, 2].set_title('original da') 

        ax[1,0].plot(trajectory[:,0])
        ax[1,0].set_title('cum.sum trajectory dx') 

        ax[1,1].plot(trajectory[:,1])
        ax[1,1].set_title('cum.sum trajectory dy') 

        ax[1,2].plot(trajectory[:,2])
        ax[1,2].set_title('cum.sumtrajectory da') 

        # Save figure into save directory specified in parameters. 
        fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform W=' + str(lkWinSize[0]) + '_R=' + str(smoothing_radius) + '.png')
    # When graphing frame boundaries are set: 
    else: 
        # Start subplot of dx, dy, and da's 
        fig, ax = plt.subplots(2,3, figsize=(40,15), sharex=True )  
        ax[0, 0].plot(transforms[:,0][graphFrames[0]:graphFrames[1]])
        ax[0, 0].set_title('original dx') 

        ax[0, 1].plot(transforms[:,1][graphFrames[0]:graphFrames[1]])
        ax[0, 1].set_title('original dy') 

        ax[0, 2].plot(transforms[:,2][graphFrames[0]:graphFrames[1]])
        ax[0, 2].set_title('original da') 

        ax[1,0].plot(trajectory[:,0][graphFrames[0]:graphFrames[1]])
        ax[1,0].set_title('cum.sum trajectory dx') 

        ax[1,1].plot(trajectory[:,1][graphFrames[0]:graphFrames[1]])
        ax[1,1].set_title('cum.sum trajectory dy') 

        ax[1,2].plot(trajectory[:,2][graphFrames[0]:graphFrames[1]])
        ax[1,2].set_title('cum.sumtrajectory da') 

        # Save figure into save directory specified in parameters. 
        fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_W=' + str(lkWinSize[0]) + '_R=' + str(smoothing_radius) + '.png')

    # Calculate smoothed trajectories using smoothing_radius 
    smoothed_trajectory = smooth(trajectory, smoothing_radius=smoothing_radius) 

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    
    # Calculate newer transformation array
    transforms_smooth = transforms + difference      

    if graphFrames[0] == None: 
        fig, ax = plt.subplots(2, 3, figsize=(40,15)) 
        ax[0,0].plot(smoothed_trajectory[:,0])
        ax[0,0].set_title('cum.sum Smoothed trajectory dx')

        ax[0,1].plot(smoothed_trajectory[:,1])
        ax[0,1].set_title('cum.sum Smoothed trajectory dy')

        ax[0,2].plot(smoothed_trajectory[:,2])
        ax[0,2].set_title('cum.sum Smoothed trajectory da')

        ax[1,0].plot(transforms_smooth[:,0])
        ax[1,0].set_title('Smoothed transforms dx')

        ax[1,1].plot(transforms_smooth[:,1])
        ax[1,1].set_title('Smoothed transforms dy')

        ax[1,2].plot(transforms_smooth[:,2])
        ax[1,2].set_title('Smoothed transforms da')

        fig.savefig(saveDir + "/" + outputFileNames + 'Post-Transform' + '_W=' + str(lkWinSize[0]) + '_R=' + str(smoothing_radius) + '.png')

    else: 
        fig, ax = plt.subplots(2, 3, figsize=(40,15)) 
        ax[0,0].plot(smoothed_trajectory[:,0][graphFrames[0]:graphFrames[1]])
        ax[0,0].set_title('cum.sum smoothed trajectory dx')

        ax[0,1].plot(smoothed_trajectory[:,1][graphFrames[0]:graphFrames[1]])
        ax[0,1].set_title('cum.sum smoothed trajectory dy')

        ax[0,2].plot(smoothed_trajectory[:,2][graphFrames[0]:graphFrames[1]])
        ax[0,2].set_title('cum.sum smoothed trajectory da')

        # Calculate difference in smoothed_trajectory and trajectory
        difference = smoothed_trajectory - trajectory
        
        # Calculate newer transformation array
        transforms_smooth = transforms + difference   
        ax[1,0].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]])
        ax[1,0].set_title('Smoothed Transforms dx')

        ax[1,1].plot(transforms_smooth[:,1][graphFrames[0]:graphFrames[1]])
        ax[1,1].set_title('Smoothed Transforms dy')

        ax[1,2].plot(transforms_smooth[:,2][graphFrames[0]:graphFrames[1]])
        ax[1,2].set_title('Smoothed Transforms da')

        fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_W=' + str(lkWinSize[0]) + '_R=' + str(smoothing_radius) + '.png')


    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        # Read next frame
        success, frame = cap.read() 
        if not success:
            break
    
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]
        
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            # Write ONLY new stabilized frames
            frame_out = frame_stabilized 
            cv2.imshow("Stabilized Frame", frame_out)
            out.write(frame_out)
            cv2.waitKey(waitTime) 
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            if (frame_out.shape[1] > 1980): 
                frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) ))

            cv2.imshow("Before and After", frame_out)
            out.write(frame_out)
            cv2.waitKey(waitTime)

    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 


# ---------------------------------KALMAN FILTER METHODS---------------------------------------------------------#  
class MovingAverageFilter(object):

    def __init__(self, window):
        self.window = window
        self.data = []

    def step(self, measurement):
        self.data.append(measurement)
        if len(self.data) > self.window:
            self.data.pop(0)

    def current_state(self):
        return sum(self.data) / len(self.data)

class SingleStateKalmanFilter(object):
    '''
    This class implements a simple derivative of the kalman 
    filter method used to smoothen out a noisy digital signal
    while preserving original information as much as possible. 
    '''
    def __init__(self, A, B, C, x, P, Q, R):
        self.A = A  # Process dynamics (process=prediction)
        self.B = B  # Control dynamics
        self.C = C  # Measurement dynamics
        self.current_state_estimate = x  # Current state estimate
        self.current_prob_estimate = P  # Current probability of state estimate
        self.Q = Q  # Process covariance (process=prediction)
        self.R = R  # Measurement covariance

    def current_state(self):
        return self.current_state_estimate

    def step(self, control_input, measurement):
        # Prediction step
        predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_input
        predicted_prob_estimate = (self.A * self.current_prob_estimate) * self.A + self.Q

        # Observation step
        innovation = measurement - self.C * predicted_state_estimate
        innovation_covariance = self.C * predicted_prob_estimate * self.C + self.R

        # Update step
        kalman_gain = predicted_prob_estimate * self.C * 1 / float(innovation_covariance)
        self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation

        # eye(n) = nxn identity matrix.
        self.current_prob_estimate = (1 - kalman_gain * self.C) * predicted_prob_estimate

def stabilizeVideoKalman(videoDir, outputFileNames, kalman=True, saveDir=None, graphFrames=(None, None), comparison=False, smoothing_radius=40, fps=15 \
    , lkWinSize=(15,15), lkMaxLevel=2, lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), tomasiCorners=2000, affinePartial=True
    , saveDiagnosticData=True, diagnosticOverlay=True, overlayRightShift=True):
    '''
    This function stabilizes a video utilizing the Kalman method of signal stabilization to smoothen out the detected affine transform
    matrices in order to achieve a smooth result for stack alignment. The method reads a video and processes each frame by detecting 
    key features to track, tracks those features over the current and next frame, extracts translation and rotation data to estimate 
    an affine transform matrix and stores it as 3 distinct digital signals of dx, dy and da. This signal is integrated to the pure 
    translation and rotation of x, y and a so that they can be smoothened out using the Kalman filter ("I'VE NEVER SEEN IT FAIL REEEE"). 
    The dx dy and da's for each frame are derived and thrown into a generated partial Affine 2D transformation matrix to be aligned. 

    Args: (hella params) 

        (param1 - videoDir): 
            (str): Path of video to be stabilized, or video of image stack to be aligned. 
        (param2 - outputFileNames): 
            (str): Name of output video file and accompanying diagnostic graphical data. 
        (param3 - kalman): 
            (bool): To use the kalman method or rolling average method of parameter transform smoothening. 
        (param4 - saveDir): 
            (str): Save directory to store stabilized video and diagnostic data. 
        (param5 - graphFrames): 
            (tuple(int, int) ): (Start frame, End Frame) to be graphed by diagnostic data. 
                                Example: graphFrames=(1, 10) --> Frames 1 to 10 will be considered for graphing affine transform data.
        (param6 - comparison): 
            (bool): Whether or not final video output is a comparison video with original unaligned stack and aligned stack 
                    side by side. 
        (param7 - smoothing_radius): 
            (int): Smoothing radius to be used if Kalman smoothening method not chosen. 40 is a good value to use. 
        (param8 - fps): 
            (int): Frames per second of stabilized video. 5-20 is a range of values. 
        (param9 - lkWinSize): 
            (tuple(int, int)): Lucas Kanade optical flow algorithm window size. (15, 15) is good. 
        (param10 - lkMaxLevel): 
            (int): Number of dimensions Lucas Kanada optical flow algorithm uses for pyramid scheme. 
        (param11 - lkCriteria): 
            (dict): More Lucas Kanade optical flow algorithm parameters to be initialized. 
        (param12 - tomasiCorners): 
            (int): Numbers of corners to be detected by the ShiTomasi corner detection algorithm 
                   in each frame to be tracked by the lucas kanade optical flow algorithm. 1000-2000 is usually good. 
    
    Returns: 

        No return value. Function stabilizes the video and saves it along with diagnostic graphical
        data in user specified save directory. 

    '''
    # Create folder for optical flow diagnostic data 
    if saveDiagnosticData == True: 
        diagnostics = createFolder(saveDir, 'diagnostics') 
        estimated_affine = open("{}\\estimated_affine.txt".format(diagnostics), "w") 
        smoothened_affine = open("{}\\smoothened_affine.txt".format(diagnostics), "w") 
    if diagnosticOverlay == True: 
        frames_location = createFolder(saveDir, 'frames') 

    # Save matrices anyways for simplicity     
    estimated_affine_matrices = [] 
    smoothened_affine_matrices = [] 
    
    waitTime = int(np.around(1000/fps))

    # if saveDir != None: 
    #     outfile = open(saveDir + outputFileNames + ".txt", 'w') 
    

    # Read input video
    cap = cv2.VideoCapture(videoDir)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # Set up videowriter object with appropriate widths 
    if (comparison==False) and (saveDir != None): 
    # Set up output video
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames +  ".avi", fourcc, fps, (w, h))
    elif saveDir != None: 
        # Check if total concatenated is less than 1980 (990 for now because not concatenated yet) 
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 990:
            write_w = int(np.around(w/2)) 
            write_h = int(np.around(h/2))
        else: 
            write_w = w 
            write_h = h 
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames + "_Comparison.avi", fourcc, fps, (int(np.around(write_w*2)), write_h))
        # print("VideoWriter WxH Dimensions: {} x {}".format(int(np.around(w*2)), h))
    _, prev = cap.read() 
    
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 

    for i in range(n_frames-1):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                    maxCorners=tomasiCorners,
                                    qualityLevel=0.01,
                                    minDistance=20,
                                    blockSize=3)
        # lk_params = dict( winSize  = lkWinSize,
        #             maxLevel = lkMaxLevel,
        #             criteria = lkCriteria )
    
        # Read next frame
        success, curr = cap.read() 
        if not success: 
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel) #, **lk_params) 
        # Sanity check                      
        # Check that number of previous points is equal to current points 
        assert prev_pts.shape == curr_pts.shape 

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
    
        #Find transformation matrix
        if affinePartial == True: 
            m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        else:
            m = cv2.estimateAffine2D(prev_pts, curr_pts) 
        #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

        # Save each estimated affine matrix. 
        # Index [0] needed due to function formatting( we only want the numpy array, nothing else ) 
        estimated_affine_matrices.append(m[0]) 

        # Extract traslation
        dx = m[0][0,2] 
        dy = m[0][1,2] 

    
        # Extract rotation angle
        da = np.arctan2(m[0][1,0], m[0][0,0])
    
        # Store transformation
        transforms[i] = [dx,dy,da]
        
        # Move to next frame
        prev_gray = curr_gray
    
        print("Frame Pairs: " + str(i+1) +  "/" + str(n_frames-2) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 

    if affinePartial == True: 
        # print("Trajectories, x, y and a: {}".format(trajectory)) 
        if graphFrames[0] == None: 
            # Start subplot of dx, dy, and da's 
            fig, ax = plt.subplots(2,3, figsize=(40,15))  
            # plt.setp(ax, xticks=[])
            ax[0, 0].plot(transforms[:,0])
            ax[0, 0].set_title('original dx') 
            ax[0, 0].set_ylabel('pixels/dt') 
            ax[0, 0].set_xlabel('frames') 

            ax[0, 1].plot(transforms[:,1])
            ax[0, 1].set_title('original dy') 
            ax[0, 1].set_ylabel('pixels/dt') 
            ax[0, 1].set_xlabel('frames')

            ax[0, 2].plot(transforms[:,2])
            ax[0, 2].set_title('original da') 
            ax[0, 2].set_ylabel('radians/dt') 
            ax[0, 2].set_xlabel('frames') 

            ax[1,0].plot(trajectory[:,0], color='purple')
            ax[1,0].set_title('integrated translation x') 
            ax[1,0].set_ylabel('pixels') 
            ax[1,0].set_xlabel('frames')

            ax[1,1].plot(trajectory[:,1], color='purple')
            ax[1,1].set_title('integrated translation y') 
            ax[1,1].set_ylabel('pixels') 
            ax[1,1].set_xlabel('frames')

            ax[1,2].plot(trajectory[:,2], color='purple')
            ax[1,2].set_title('integrated rotation a')
            ax[1,2].set_ylabel('radians') 
            ax[1,2].set_xlabel('frames')

            # Save figure into save directory specified in parameters. 
            fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + '.png')
        # When graphing frame boundaries are set: 
        else: 
            # Start subplot of dx, dy, and da's 
            fig, ax = plt.subplots(2,3, figsize=(40,15))  
            ax[0, 0].plot(transforms[:,0][graphFrames[0]:graphFrames[1]])
            ax[0, 0].set_title('original dx') 

            ax[0, 1].plot(transforms[:,1][graphFrames[0]:graphFrames[1]])
            ax[0, 1].set_title('original dy') 

            ax[0, 2].plot(transforms[:,2][graphFrames[0]:graphFrames[1]])
            ax[0, 2].set_title('original da') 

            ax[1,0].plot(trajectory[:,0][graphFrames[0]:graphFrames[1]])
            ax[1,0].set_title('integrated translation x') 

            ax[1,1].plot(trajectory[:,1][graphFrames[0]:graphFrames[1]])
            ax[1,1].set_title('integrated translation y') 

            ax[1,2].plot(trajectory[:,2][graphFrames[0]:graphFrames[1]])
            ax[1,2].set_title('integrated rotation a') 

            # Save figure into save directory specified in parameters. 
            fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) +  '.png')

    if kalman == False: 
        # Calculate smoothed trajectories using smoothing_radius 
        smoothed_trajectory = smooth(trajectory) 
    else: 
        x_trajectory = [] 
        y_trajectory = [] 
        a_trajectory = [] 
        for i in range(len(trajectory)): 
            x_trajectory.append(trajectory[i][0])
            y_trajectory.append(trajectory[i][1]) 
            a_trajectory.append(trajectory[i][2])  

        A = 1 # No process innovation 
        C = 1 # Measurement 
        B = 0 # No control input 
        Q = 0.0001 # Process noise covariance 
        R = 1 # Measurement covariance 
        # X = 0 # Initial estimate 
        P = 1 # Initial covariance 

        # Dividing initial trajectories by 2 to provide smoother first guess for Kalman
        x_kalman_filter = SingleStateKalmanFilter(A, B, C, trajectory[0][0]/4, P, Q, R) 
        y_kalman_filter = SingleStateKalmanFilter(A, B, C, trajectory[0][1]/4, P, Q, R) 
        a_kalman_filter = SingleStateKalmanFilter(A, B, C, 0, P, Q, R) 

        # Empty list for capturing filter estimates 
        x_kalman_filter_estimates = [] 
        y_kalman_filter_estimates = [] 
        a_kalman_filter_estimates = [] 

        # Simulate data arriving sequentially -- dx 
        for data in x_trajectory: 
            x_kalman_filter.step(0, data)
            x_kalman_filter_estimates.append(x_kalman_filter.current_state())
        # Simulate data arriving sequentially -- dy 
        for data in y_trajectory: 
            y_kalman_filter.step(0, data)
            y_kalman_filter_estimates.append(y_kalman_filter.current_state())
        # Simulate data arriving sequentially -- da
        for data in a_trajectory: 
            a_kalman_filter.step(0, data)
            a_kalman_filter_estimates.append(a_kalman_filter.current_state())
        
        # Store filtered trajectories in appropriate format 
        smoothed_trajectory = np.empty( (len(trajectory), 3))
        for i in range(len(trajectory)): 
            smoothed_trajectory[i] = [ x_kalman_filter_estimates[i], y_kalman_filter_estimates[i], a_kalman_filter_estimates[i] ] 
            # smoothed_trajectory.append( [x_kalman_filter_estimates[i], y_kalman_filter_estimates[i], a_kalman_filter_estimates[i]] )               # potential error here formatting wise 
        # print("Smoothed Trajectories, x, y and a: {}".format(smoothed_trajectory)) 

    # Calculate difference in smoothed_trajectory and trajectory
    # Essentially same as taking a discrete derivative to find dx, dy and da 
    difference = smoothed_trajectory - trajectory
    
    # Calculate newer transformation array
    transforms_smooth = transforms + difference      

    if affinePartial == True: 
        if graphFrames[0] == None: 

            # Initialize subplot figure 
            fig, ax = plt.subplots(2, 3, figsize=(40,15)) 

            # Plot x smoothed trajectory and original trajectory
            ax[0,0].plot(smoothed_trajectory[:,0], label='Smoothed Trajectory (x)', color='red')
            ax[0,0].plot(trajectory[:, 0], label="Original Trajectory (x)", color='purple') 
            ax[0,0].plot(difference[:,0], label="Difference = Smoothed Trajectory - Original Trajectory", color='green')
            ax[0,0].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,0].set_title('Kalman filtered trajectory x')
                ax[0,0].set_ylabel('pixels') 
                ax[0,0].set_xlabel('frames')
            else: 
                ax[0,0].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory x')

            # Plot y smoothed trajectory and original trajectory 
            ax[0,1].plot(smoothed_trajectory[:,1], label='Smoothed Trajectory (y)', color='red') 
            ax[0,1].plot(trajectory[:, 1], label="Original Trajectory (y)", color='purple') 
            ax[0,1].plot(difference[:,1], label="Difference = Smoothed Trajectory - Original Trajectory", color='green')
            ax[0,1].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,1].set_title('Kalman filtered trajectory y')
                ax[0,1].set_ylabel('pixels') 
                ax[0,1].set_xlabel('frames')
            else: 
                ax[0,1].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory y')

            # Plot a smoothed trajectory and original trajectory 
            ax[0,2].plot(smoothed_trajectory[:,2], label="Smoothed Trajectory (a)", color='red') 
            ax[0,2].plot(trajectory[:,2], label="Original Trajectory (a)", color='purple') 
            ax[0,2].plot(difference[:,2], label="Difference = Smoothed Trajectory - Original Trajectory", color='green')
            ax[0,2].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,2].set_title('Kalman filtered trajectory a')
                ax[0,2].set_ylabel('radians') 
                ax[0,2].set_xlabel('frames')
            else: 
                ax[0,2].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory a')

            ax[1,0].plot(transforms_smooth[:,0], label='Smoothed Transforms (dx) = Original Transforms + Difference', linewidth=1.2) 
            ax[1,0].plot(transforms[:,0], label='Original Transforms (dx)', linewidth = 0.5, alpha=0.75) 
            ax[1,0].plot(difference[:,0], label="Difference", linewidth = 0.5, alpha=0.75)
            ax[1,0].set_title('Smoothed transforms dx')
            ax[1,0].set_ylabel('pixels/dt') 
            ax[1,0].set_xlabel('frames')
            ax[1,0].legend() 

            ax[1,1].plot(transforms_smooth[:,1], label="Smoothed Transforms (dy) = Original Transforms + Difference", linewidth=1.2) 
            ax[1,1].plot(transforms[:,1], label="Original Transforms (dy)", linewidth = 0.5, alpha=0.75)
            ax[1,1].plot(difference[:,1], label="Difference", linewidth = 0.5, alpha=0.75)
            ax[1,1].set_title('Smoothed transforms dy')
            ax[1,1].set_ylabel('pixels/dt') 
            ax[1,1].set_xlabel('frames')
            ax[1,1].legend() 

            ax[1,2].plot(transforms_smooth[:,2], label="Smoothed Transforms (da) = Original Transforms + Difference", linewidth=1.2) 
            ax[1,2].plot(transforms[:,2], label="Original Transforms (da)", linewidth = 0.5, alpha=0.75)
            ax[1,2].plot(difference[:,2], label="Difference", linewidth = 0.5, alpha=0.75)
            ax[1,2].set_title('Smoothed transforms da')
            ax[1,2].set_ylabel('radians/dt') 
            ax[1,2].set_xlabel('frames')
            ax[1,2].legend() 

            

            if kalman==True: 
                fig.savefig(saveDir + "/" + outputFileNames + '_Kalman_Post-Transform' +  '.png')
            else: 
                fig.savefig(saveDir + "/" + outputFileNames + '_Roll_Radius=' + str(smoothing_radius) + '_Post-Transform' +  '.png')

        else: 
            fig, ax = plt.subplots(2, 3, figsize=(40,15)) 
            ax[0,0].plot(smoothed_trajectory[:,0][graphFrames[0]:graphFrames[1]])
            ax[0,0].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]]) 
            ax[0,0].set_title('Kalman filtered trajectory x')
            ax[0,0].legend() 

            ax[0,1].plot(smoothed_trajectory[:,1][graphFrames[0]:graphFrames[1]])
            ax[0,1].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]]) 
            ax[0,1].set_title('Kalman filtered trajectory y')
            ax[0,1].legend() 

            ax[0,2].plot(smoothed_trajectory[:,2][graphFrames[0]:graphFrames[1]])
            ax[0,2].plot(transforms_smooth[:,2][graphFrames[0]:graphFrames[1]]) 
            ax[0,2].set_title('Kalman filtered trajectory a')
            ax[0,2].legend() 

            # Calculate difference in smoothed_trajectory and trajectory
            difference = smoothed_trajectory - trajectory
            
            # Calculate newer transformation array
            transforms_smooth = transforms + difference   
            ax[1,0].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]])
            ax[1,0].plot(transforms[:,0][graphFrames[0]:graphFrames[1]]) 
            ax[1,0].set_title('Smoothed Transforms dx')
            ax[1,0].legend() 

            ax[1,1].plot(transforms_smooth[:,1][graphFrames[0]:graphFrames[1]])
            ax[1,1].plot(transforms[:,1][graphFrames[0]:graphFrames[1]]) 
            ax[1,1].set_title('Smoothed Transforms dy')
            ax[1,1].legend() 

            ax[1,2].plot(transforms_smooth[:,2][graphFrames[0]:graphFrames[1]])
            ax[1,2].plot(transforms[:,2][graphFrames[0]:graphFrames[1]]) 
            ax[1,2].set_title('Smoothed Transforms da')
            ax[1,2].legend() 

            if kalman==True: 
                fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_Kalman' +  '.png')
            else: 
                fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_Roll_Radius=' + str(smoothing_radius)  +  '.png')

    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        # Read next frame
        success, frame = cap.read() 
        if not success:
            break
    
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]
        
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            frame_out = frame_stabilized 
            cv2.imshow("Stabilized Frame", frame_out)
            out.write(frame_out)
            cv2.waitKey(waitTime) 
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            if (frame_out.shape[1] > 1980): 
                frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) ))
            # print("Final Frame Out Shape: {}".format(frame_out.shape))
            cv2.imshow("Before and After", frame_out)
            out.write(frame_out)
            cv2.waitKey(waitTime)

    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 

def concatenateVideo(videoDir1, videoDir2, saveDir, outputFileName, fps=5, resize=False, saveFrames=True): 

    # Save concatenated frames
    framesDir = createFolder(saveDir, "{} - Frames".format(outputFileName)) 

    # Read input of video1 and video2 
    cap1 = cv2.VideoCapture(videoDir1) 
    cap2 = cv2.VideoCapture(videoDir2) 

    # Get total frames count 
    n_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) 

    assert n_frames1 == n_frames2 


    # If both are same, set number of frames
    n_frames = n_frames1

    # Get Width and Height of video streams 
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try: 
        assert h1 == h2 
    except AssertionError: 
        print("Video frames not same size. Largest will be resized to fit.") 
        resize = True 
        pass

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    write_w = w1 + w2 
    write_h = h1 

    # Identify smallest w to see if we need to downsize 
    smaller_w = w1 
    if w1 > w2: 
        smaller_w =  w2 
    
    if resize == True: 
        write_w = smaller_w * 2 

    if write_w > 2880: 
        write_w = int(np.around(write_w/2)) 
        write_h = int(np.around(write_h/2))

    out = cv2.VideoWriter("{}\\{}.avi".format(saveDir, outputFileName), fourcc, fps, (write_w, write_h) ) 

    for i in range(n_frames): 

        _, f_1 = cap1.read() 
        _, f_2 = cap2.read() 

        if resize == True: 

            width1 = f_1.shape[1] 
            width2 = f_2.shape[1] 

            if width1 > width2: 

                f_1 = cv2.resize(f_1, (f_2.shape[1], f_2.shape[0]) ) 

            elif width1 < width2: 

                f_2 = cv2.resize(f_2, (f_1.shape[1], f_1.shape[0]) ) 

        f_write = cv2.hconcat( [f_1, f_2] ) 

        # # Convert to grayscale 
        # f_write = cv2.cvtColor(f_write, cv2.COLOR_BGR2GRAY)

        # Read height and width only 
        height, width, dimensions = f_write.shape 
 
        if width > 2880: 

            if f_1.shape[1] != f_write.shape[1]:
                f_write = cv2.resize(f_write, (int(np.around(f_write.shape[1]/2)), int(np.around(f_write.shape[0]/2)) ))
                # f_write = resizeImage(f_write, 0.5) 

        
        out.write(f_write) 
        cv2.imwrite("{}\\Frame{}.tif".format(framesDir, i+1), f_write)

    cap1.release() 
    cap2.release() 
    cv2.destroyAllWindows() 



def stabilizeVideoKalmanAffineControl(videoDir, outputFileNames, kalman=True, saveDir=None, graphFrames=(None, None), comparison=False, smoothing_radius=40, fps=15 \
    , lkWinSize=(15,15), lkMaxLevel=3, lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), tomasiCorners=2000, affinePartial=False, show=True
    , saveDiagnosticData=False, diagnosticOverlay=True, overlayRightShift=True):
    '''
    This function stabilizes a video utilizing the Kalman method of signal stabilization to smoothen out the detected affine transform
    matrices in order to achieve a smooth result for stack alignment. The method reads a video and processes each frame by detecting 
    key features to track, tracks those features over the current and next frame, extracts translation and rotation data to estimate 
    an affine transform matrix and stores it as 3 distinct digital signals of dx, dy and da. This signal is integrated to the pure 
    translation and rotation of x, y and a so that they can be smoothened out using the Kalman filter ("I'VE NEVER SEEN IT FAIL REEEE"). 
    The dx dy and da's for each frame are derived and thrown into a generated partial Affine 2D transformation matrix to be aligned. 

    Args: (hella params) 

        (param1 - videoDir): 
            (str): Path of video to be stabilized, or video of image stack to be aligned. 
        (param2 - outputFileNames): 
            (str): Name of output video file and accompanying diagnostic graphical data. 
        (param3 - kalman): 
            (bool): To use the kalman method or rolling average method of parameter transform smoothening. 
        (param4 - saveDir): 
            (str): Save directory to store stabilized video and diagnostic data. 
        (param5 - graphFrames): 
            (tuple(int, int) ): (Start frame, End Frame) to be graphed by diagnostic data. 
                                Example: graphFrames=(1, 10) --> Frames 1 to 10 will be considered for graphing affine transform data.
        (param6 - comparison): 
            (bool): Whether or not final video output is a comparison video with original unaligned stack and aligned stack 
                    side by side. 
        (param7 - smoothing_radius): 
            (int): Smoothing radius to be used if Kalman smoothening method not chosen. 40 is a good value to use. 
        (param8 - fps): 
            (int): Frames per second of stabilized video. 5-20 is a range of values. 
        (param9 - lkWinSize): 
            (tuple(int, int)): Lucas Kanade optical flow algorithm window size. (15, 15) is good. 
        (param10 - lkMaxLevel): 
            (int): Number of dimensions Lucas Kanada optical flow algorithm uses for pyramid scheme. 
        (param11 - lkCriteria): 
            (dict): More Lucas Kanade optical flow algorithm parameters to be initialized. 
        (param12 - tomasiCorners): 
            (int): Numbers of corners to be detected by the ShiTomasi corner detection algorithm 
                   in each frame to be tracked by the lucas kanade optical flow algorithm. 1000-2000 is usually good. 
    
    Returns: 

        No return value. Function stabilizes the video and saves it along with diagnostic graphical
        data in user specified save directory. 

    '''
    # Create folder for optical flow diagnostic data 
    if saveDiagnosticData == True: 
        diagnostics = createFolder(saveDir, 'diagnostics') 
        estimated_affine = open("{}\\estimated_affine.txt".format(diagnostics), "w") 
        smoothened_affine = open("{}\\smoothened_affine.txt".format(diagnostics), "w") 
    if diagnosticOverlay == True: 
        frames_location = createFolder(saveDir, 'frames') 

    # Save matrices anyways for simplicity     
    estimated_affine_matrices = [] 
    smoothened_affine_matrices = [] 

    waitTime = int(np.around(1000/fps))

    # if saveDir != None: 
    #     outfile = open(saveDir + outputFileNames + ".txt", 'w') 
    

    # Read input video
    cap = cv2.VideoCapture(videoDir)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # Set up videowriter object with appropriate widths 
    if (comparison==False) and (saveDir != None): 
    # Set up output video
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames +  ".avi", fourcc, fps, (w, h))
    elif saveDir != None: 
        # Check if total concatenated is less than 1980 (990 for now because not concatenated yet) 
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 990:
            write_w = int(np.around(w/2)) 
            write_h = int(np.around(h/2))
        else: 
            write_w = w 
            write_h = h 
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames + "_Comparison.avi", fourcc, fps, (int(np.around(write_w*2)), write_h))
        # print("VideoWriter WxH Dimensions: {} x {}".format(int(np.around(w*2)), h))
    _, prev = cap.read() 
    
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define affine transformation 2x3 array 
    transforms = np.zeros((n_frames-1, 6), np.float32) 

    for i in range(n_frames-1):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                    maxCorners=tomasiCorners,
                                    qualityLevel=0.01,
                                    minDistance=20,
                                    blockSize=3)
        # lk_params = dict( winSize  = lkWinSize,
        #             maxLevel = lkMaxLevel,
        #             criteria = lkCriteria )
    
        # Read next frame
        success, curr = cap.read() 
        if not success: 
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel) #, **lk_params) 
        # Sanity check
        # Check that number of previous points is equal to current points 
        assert prev_pts.shape == curr_pts.shape 

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
    
        #Find transformation matrix
        if affinePartial == True: 
            m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        else:
            # Get Shear, Scale, and translation 
            m = cv2.estimateAffine2D(prev_pts, curr_pts) 

        # Save each estimated affine matrix. 
        # Index [0] needed due to function formatting( we only want the numpy array, nothing else ) 
        estimated_affine_matrices.append(m[0]) 

        #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

        #Extract affine transform matrix members 
        zero_zero = m[0][0,0]
        zero_one = m[0][0,1] 
        one_zero = m[0][1,0] 
        one_one = m[0][1,1] 
        dx = m[0][0,2] 
        dy = m[0][1,2] 

        # Store Shear, Scale and Translation transformation matrix 
        # Subtract 1 from scaling terms so scaling factors accurately represented 
        transforms[i] = [zero_zero-1, zero_one, dx, one_zero, one_one-1, dy] 


        # Move to next frame
        prev_gray = curr_gray
    
        print("Frame Pairs: " + str(i+1) +  "/" + str(n_frames-1) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 

    if affinePartial == False: 
        # print("Trajectories, x, y and a: {}".format(trajectory)) 
        if graphFrames[0] == None: 
            # Start subplot of dx, dy, and da's 
            fig, ax = plt.subplots(2,6, figsize=(60,20))  
            # plt.setp(ax, xticks=[])
            ax[0, 0].plot(transforms[:,0])
            ax[0, 0].set_title('transforms[0,0]') 
            # ax[0, 0].set_ylabel('pixels/dt') 
            ax[0, 0].set_xlabel('frames') 

            ax[0, 1].plot(transforms[:,1])
            ax[0, 1].set_title('transforms[0,1]') 
            # ax[0, 1].set_ylabel('pixels/dt') 
            ax[0, 1].set_xlabel('frames')

            ax[0, 2].plot(transforms[:,2])
            ax[0, 2].set_title('original dx') 
            ax[0, 2].set_ylabel('pixels/dt') 
            ax[0, 2].set_xlabel('frames') 

            ax[0, 3].plot(transforms[:,3])
            ax[0, 3].set_title('transforms[1,0]') 
            # ax[0, 3].set_ylabel('pixels/dt') 
            ax[0, 3].set_xlabel('frames')

            ax[0, 4].plot(transforms[:,4])
            ax[0, 4].set_title('transforms[1,1]') 
            # ax[0, 4].set_ylabel('pixels/dt') 
            ax[0, 4].set_xlabel('frames')

            ax[0, 5].plot(transforms[:,5])
            ax[0, 5].set_title('original dy') 
            ax[0, 5].set_ylabel('pixels/dt') 
            ax[0, 5].set_xlabel('frames')


            # Draw integrated transforms --> trajectories 
            ax[1,0].plot(trajectory[:,0], color='purple')
            ax[1,0].set_title('integrated transforms[0,0]') 
            ax[1,0].set_ylabel('pixels') 
            ax[1,0].set_xlabel('frames')

            ax[1,1].plot(trajectory[:,1], color='purple')
            ax[1,1].set_title('integrated transforms[0,1]') 
            ax[1,1].set_ylabel('pixels') 
            ax[1,1].set_xlabel('frames')

            ax[1,2].plot(trajectory[:,2], color='purple')
            ax[1,2].set_title('integrated translation x')
            ax[1,2].set_ylabel('radians') 
            ax[1,2].set_xlabel('frames')

            ax[1,3].plot(trajectory[:,3], color='purple')
            ax[1,3].set_title('integrated transforms[1,0]')
            ax[1,3].set_ylabel('radians') 
            ax[1,3].set_xlabel('frames')

            ax[1,4].plot(trajectory[:,4], color='purple')
            ax[1,4].set_title('integrated transforms[1,1]')
            ax[1,4].set_ylabel('radians') 
            ax[1,4].set_xlabel('frames')

            ax[1,5].plot(trajectory[:,5], color='purple')
            ax[1,5].set_title('integrated translation y')
            ax[1,5].set_ylabel('radians') 
            ax[1,5].set_xlabel('frames')


            # Save figure into save directory specified in parameters. 
            fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + '.png')

            plt.close(fig) 
        # When graphing frame boundaries are set: 
        # else: 
        #     # Start subplot of dx, dy, and da's 
        #     fig, ax = plt.subplots(2,3, figsize=(40,15))  
        #     ax[0, 0].plot(transforms[:,0][graphFrames[0]:graphFrames[1]])
        #     ax[0, 0].set_title('original dx') 

        #     ax[0, 1].plot(transforms[:,1][graphFrames[0]:graphFrames[1]])
        #     ax[0, 1].set_title('original dy') 

        #     ax[0, 2].plot(transforms[:,2][graphFrames[0]:graphFrames[1]])
        #     ax[0, 2].set_title('original da') 

        #     ax[1,0].plot(trajectory[:,0][graphFrames[0]:graphFrames[1]])
        #     ax[1,0].set_title('integrated translation x') 

        #     ax[1,1].plot(trajectory[:,1][graphFrames[0]:graphFrames[1]])
        #     ax[1,1].set_title('integrated translation y') 

        #     ax[1,2].plot(trajectory[:,2][graphFrames[0]:graphFrames[1]])
        #     ax[1,2].set_title('integrated rotation a') 

        #     # Save figure into save directory specified in parameters. 
        #     fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) +  '.png')

        #     plt.close(fig) 

    if kalman == False: 
        # Calculate smoothed trajectories using smoothing_radius 
        smoothed_trajectory = smooth(trajectory) 
    else: 
        zero_zero_trajectory = [] 
        zero_one_trajectory = [] 
        x_trajectory = [] 
        one_zero_trajectory = [] 
        one_one_trajectory = [] 
        y_trajectory = [] 
        a_trajectory = [] 

        for i in range(len(trajectory)): 
            zero_zero_trajectory.append(trajectory[i][0])
            zero_one_trajectory.append(trajectory[i][1]) 
            x_trajectory.append(trajectory[i][2])
            one_zero_trajectory.append(trajectory[i][3])  
            one_one_trajectory.append(trajectory[i][4])  
            y_trajectory.append(trajectory[i][5])  



        A = 1 # No process innovation 
        C = 1 # Measurement 
        B = 0 # No control input 
        Q = 0.0001 # Process noise covariance 
        R = 1 # Measurement covariance 
        # X = 0 # Initial estimate 
        P = 1 # Initial covariance 

        # Dividing initial trajectories by 4 to provide smoother first guess for Kalman
        zero_zero_kalman_filter = SingleStateKalmanFilter(A, B, C, zero_zero_trajectory[0]/4, P, Q, R) 
        zero_one_kalman_filter = SingleStateKalmanFilter(A, B, C, zero_one_trajectory[0]/4, P, Q, R) 
        x_kalman_filter = SingleStateKalmanFilter(A, B, C, x_trajectory[0]/4, P, Q, R) 
        one_zero_kalman_filter = SingleStateKalmanFilter(A, B, C, one_zero_trajectory[0]/4, P, Q, R) 
        one_one_kalman_filter = SingleStateKalmanFilter(A, B, C, one_one_trajectory[0]/4, P, Q, R) 
        y_kalman_filter = SingleStateKalmanFilter(A, B, C, y_trajectory[0]/4, P, Q, R) 



        # Empty list for capturing filter estimates 
        zero_zero_kalman_filter_estimates = [] 
        zero_one_kalman_filter_estimates = [] 
        x_kalman_filter_estimates = [] 
        one_zero_kalman_filter_estimates = [] 
        one_one_kalman_filter_estimates = [] 
        y_kalman_filter_estimates = [] 


        # Simulate data arriving sequentially -- zero_zero
        for data in zero_zero_trajectory: 
            zero_zero_kalman_filter.step(0, data)
            zero_zero_kalman_filter_estimates.append(zero_zero_kalman_filter.current_state())
        # Simulate data arriving sequentially -- zero_one
        for data in zero_one_trajectory: 
            zero_one_kalman_filter.step(0, data)
            zero_one_kalman_filter_estimates.append(zero_one_kalman_filter.current_state())
        # Simulate data arriving sequentially -- dx 
        for data in x_trajectory: 
            x_kalman_filter.step(0, data)
            x_kalman_filter_estimates.append(x_kalman_filter.current_state())

        # Simulate data arriving sequentially -- one_zero
        for data in one_zero_trajectory: 
            one_zero_kalman_filter.step(0, data)
            one_zero_kalman_filter_estimates.append(one_zero_kalman_filter.current_state())
        # Simulate data arriving sequentially -- one_one
        for data in zero_zero_trajectory: 
            one_one_kalman_filter.step(0, data)
            one_one_kalman_filter_estimates.append(one_one_kalman_filter.current_state())
        # Simulate data arriving sequentially -- dy
        for data in y_trajectory: 
            y_kalman_filter.step(0, data)
            y_kalman_filter_estimates.append(y_kalman_filter.current_state())
        

        
        # Store filtered trajectories in appropriate format 
        smoothed_trajectory = np.empty( (len(trajectory), 6))
        smoothed_a_trajectory = np.empty( (len(trajectory), 1)) 
        for i in range(len(trajectory)): 
            smoothed_trajectory[i] = [ zero_zero_kalman_filter_estimates[i], zero_one_kalman_filter_estimates[i], x_kalman_filter_estimates[i],
                                       one_zero_kalman_filter_estimates[i], one_one_kalman_filter_estimates[i], y_kalman_filter_estimates[i] ] 


    # Calculate difference in smoothed_trajectory and trajectory
    # Essentially same as taking a discrete derivative to find dx, dy and da 
    difference = smoothed_trajectory - trajectory

    
    # Calculate newer transformation array
    transforms_smooth = transforms + difference 
   

    if affinePartial == False: 
        if graphFrames[0] == None: 

            # Initialize subplot figure 
            fig, ax = plt.subplots(2, 6, figsize=(60,20)) 

            # Plot smoothed integrated transforms [0,0]
            ax[0,0].plot(smoothed_trajectory[:,0], label='Smoothed Integrated Transforms[0,0]', color='red')
            ax[0,0].plot(trajectory[:, 0], label="Original Integrated Transforms[0,0]", color='purple') 
            ax[0,0].plot(difference[:,0], label="Difference[0,0] = Smoothed Integrated Transforms[0,0] - Original Integrated Transforms[0,0]", color='green')
            ax[0,0].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,0].set_title('Kalman Filtered Integrated Transforms[0,0]')
                # ax[0,0].set_ylabel('pixels') 
                ax[0,0].set_xlabel('frames')
            else: 
                ax[0,0].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory x')

            # Plot smoothed integrated transforms[0,1]
            ax[0,1].plot(smoothed_trajectory[:,1], label='Smoothed Integrated Transforms[0,1]', color='red')
            ax[0,1].plot(trajectory[:, 1], label="Original Integrated Transforms[0,1]", color='purple') 
            ax[0,1].plot(difference[:,1], label="Difference[0,1] = Smoothed Integrated Transforms[0,1] - Original Integrated Transforms[0,1]", color='green')
            ax[0,1].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,1].set_title('Kalman Filtered Integrated Transforms[0,1]')
                # ax[0,0].set_ylabel('pixels') 
                ax[0,1].set_xlabel('frames')
            else: 
                ax[0,1].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory x')
            

            # Plot x smoothed trajectory and original trajectory 
            ax[0,2].plot(smoothed_trajectory[:,2], label='Smoothed Trajectory (x)', color='red') 
            ax[0,2].plot(trajectory[:, 2], label="Original Trajectory (x)", color='purple') 
            ax[0,2].plot(difference[:,2], label="Difference = Smoothed Trajectory - Original Trajectory", color='green')
            ax[0,2].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,2].set_title('Kalman Filtered Translation Trajectory x')
                ax[0,2].set_ylabel('pixels') 
                ax[0,2].set_xlabel('frames')
            else: 
                ax[0,2].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory x')

            # Plot smoothed integrated transforms[1,0]
            ax[0,3].plot(smoothed_trajectory[:,3], label='Smoothed Integrated Transforms[1,0]', color='red')
            ax[0,3].plot(trajectory[:, 3], label="Original Integrated Transforms[1,0]", color='purple') 
            ax[0,3].plot(difference[:,3], label="Difference[1,0] = Smoothed Integrated Transforms[1,0] - Original Integrated Transforms[1,0]", color='green')
            ax[0,3].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,3].set_title('Kalman Filtered Integrated Transforms[1,0]')
                # ax[0,0].set_ylabel('pixels') 
                ax[0,3].set_xlabel('frames')
            else: 
                ax[0,3].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered Integrated Transforms[1,0]')

            # Plot smoothed integrated transforms[1,1]
            ax[0,4].plot(smoothed_trajectory[:,4], label='Smoothed Integrated Transforms[1,1]', color='red')
            ax[0,4].plot(trajectory[:, 4], label="Original Integrated Transforms[1,1]", color='purple') 
            ax[0,4].plot(difference[:,4], label="Difference[1,1] = Smoothed Integrated Transforms[1,1] - Original Integrated Transforms[1,1]", color='green')
            ax[0,4].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,4].set_title('Kalman Filtered Integrated Transforms[1,1]')
                # ax[0,0].set_ylabel('pixels') 
                ax[0,4].set_xlabel('frames')
            else: 
                ax[0,4].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered Integrated Transforms[1,1]')

            # Plot a smoothed trajectory and original trajectory 
            ax[0,5].plot(smoothed_trajectory[:,5], label="Smoothed Trajectory (y)", color='red') 
            ax[0,5].plot(trajectory[:,5], label="Original Trajectory (y)", color='purple') 
            ax[0,5].plot(difference[:,5], label="Difference = Smoothed Trajectory - Original Trajectory", color='green')
            ax[0,5].legend() 

            # Label axis 
            if kalman==True: 
                ax[0,5].set_title('Kalman Filtered Translation Trajectory y')
                ax[0,5].set_ylabel('pixels') 
                ax[0,5].set_xlabel('frames')
            else: 
                ax[0,5].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory y')


            ax[1,0].plot(transforms_smooth[:,0], label='Smoothed Transforms[0,0] = Original Transforms[0,0] + Difference[0,0]', linewidth=1.2) 
            ax[1,0].plot(transforms[:,0], label='Original Transforms[0,0]', linewidth = 0.5, alpha=0.75) 
            ax[1,0].plot(difference[:,0], label="Difference[0,0]", linewidth = 0.5, alpha=0.75)
            ax[1,0].set_title('Smoothed Transforms[1,0]')
            # ax[1,0].set_ylabel('pixels/dt') 
            ax[1,0].set_xlabel('frames')
            ax[1,0].legend() 

            ax[1,1].plot(transforms_smooth[:,1], label='Smoothed Transforms[0,1] = Original Transforms[0,1] + Difference[0,1]', linewidth=1.2) 
            ax[1,1].plot(transforms[:,1], label='Original Transforms[0,1]', linewidth = 0.5, alpha=0.75) 
            ax[1,1].plot(difference[:,1], label="Difference[0,1]", linewidth = 0.5, alpha=0.75)
            ax[1,1].set_title('Smoothed Transforms[0,1]')
            # ax[1,0].set_ylabel('pixels/dt') 
            ax[1,1].set_xlabel('frames')
            ax[1,1].legend() 

            ax[1,2].plot(transforms_smooth[:,2], label="Smoothed Transforms (dx) = Original Transforms + Difference", linewidth=1.2) 
            ax[1,2].plot(transforms[:,2], label="Original Transforms (dx)", linewidth = 0.5, alpha=0.75)
            ax[1,2].plot(difference[:,2], label="Difference", linewidth = 0.5, alpha=0.75)
            ax[1,2].set_title('Smoothed transforms dx')
            ax[1,2].set_ylabel('pixels/dt') 
            ax[1,2].set_xlabel('frames')
            ax[1,2].legend() 

            ax[1,3].plot(transforms_smooth[:,3], label='Smoothed Transforms[1,0] = Original Transforms[1,0] + Difference[1,0]', linewidth=1.2) 
            ax[1,3].plot(transforms[:,3], label='Original Transforms[1,0]', linewidth = 0.5, alpha=0.75) 
            ax[1,3].plot(difference[:,3], label="Difference[1,0]", linewidth = 0.5, alpha=0.75)
            ax[1,3].set_title('Smoothed Transforms[1,0]')
            # ax[1,0].set_ylabel('pixels/dt') 
            ax[1,3].set_xlabel('frames')
            ax[1,3].legend() 

            ax[1,4].plot(transforms_smooth[:,4], label='Smoothed Transforms[1,1] = Original Transforms[1,1] + Difference[1,1]', linewidth=1.2) 
            ax[1,4].plot(transforms[:,4], label='Original Transforms[1,1]', linewidth = 0.5, alpha=0.75) 
            ax[1,4].plot(difference[:,4], label="Difference[1,1]", linewidth = 0.5, alpha=0.75)
            ax[1,4].set_title('Smoothed Transforms[1,1]')
            # ax[1,0].set_ylabel('pixels/dt') 
            ax[1,4].set_xlabel('frames')
            ax[1,4].legend() 

            ax[1,5].plot(transforms_smooth[:,5], label="Smoothed Transforms (dy) = Original Transforms + Difference", linewidth=1.2) 
            ax[1,5].plot(transforms[:,5], label="Original Transforms (dy)", linewidth = 0.5, alpha=0.75)
            ax[1,5].plot(difference[:,5], label="Difference", linewidth = 0.5, alpha=0.75)
            ax[1,5].set_title('Smoothed transforms dy')
            ax[1,5].set_ylabel('pixels/dt') 
            ax[1,5].set_xlabel('frames')
            ax[1,5].legend()


            

            if kalman==True: 
                fig.savefig(saveDir + "/" + outputFileNames + '_Kalman_Post-Transform' +  '.png')
            else: 
                fig.savefig(saveDir + "/" + outputFileNames + '_Roll_Radius=' + str(smoothing_radius) + '_Post-Transform' +  '.png')

            plt.close(fig) 

        # else: 
        #     fig, ax = plt.subplots(2, 3, figsize=(40,15)) 
        #     ax[0,0].plot(smoothed_trajectory[:,0][graphFrames[0]:graphFrames[1]])
        #     ax[0,0].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]]) 
        #     ax[0,0].set_title('Kalman filtered trajectory x')
        #     ax[0,0].legend() 

        #     ax[0,1].plot(smoothed_trajectory[:,1][graphFrames[0]:graphFrames[1]])
        #     ax[0,1].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]]) 
        #     ax[0,1].set_title('Kalman filtered trajectory y')
        #     ax[0,1].legend() 

        #     ax[0,2].plot(smoothed_trajectory[:,2][graphFrames[0]:graphFrames[1]])
        #     ax[0,2].plot(transforms_smooth[:,2][graphFrames[0]:graphFrames[1]]) 
        #     ax[0,2].set_title('Kalman filtered trajectory a')
        #     ax[0,2].legend() 

        #     # Calculate difference in smoothed_trajectory and trajectory
        #     difference = smoothed_trajectory - trajectory
            
        #     # Calculate newer transformation array
        #     transforms_smooth = transforms + difference   
        #     ax[1,0].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]])
        #     ax[1,0].plot(transforms[:,0][graphFrames[0]:graphFrames[1]]) 
        #     ax[1,0].set_title('Smoothed Transforms dx')
        #     ax[1,0].legend() 

        #     ax[1,1].plot(transforms_smooth[:,1][graphFrames[0]:graphFrames[1]])
        #     ax[1,1].plot(transforms[:,1][graphFrames[0]:graphFrames[1]]) 
        #     ax[1,1].set_title('Smoothed Transforms dy')
        #     ax[1,1].legend() 

        #     ax[1,2].plot(transforms_smooth[:,2][graphFrames[0]:graphFrames[1]])
        #     ax[1,2].plot(transforms[:,2][graphFrames[0]:graphFrames[1]]) 
        #     ax[1,2].set_title('Smoothed Transforms da')
        #     ax[1,2].legend() 

        #     if kalman==True: 
        #         fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_Kalman' +  '.png')
        #     else: 
        #         fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_Roll_Radius=' + str(smoothing_radius)  +  '.png')

        #     plt.close(fig) 

    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-1):
        # Read next frame
        success, frame = cap.read() 
        if not success:
            break
        

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = transforms_smooth[i,0] + 1     # Add 1 for scale x 
        m[0,1] = transforms_smooth[i,1] 
        m[0,2] = transforms_smooth[i,2] 
        m[1,0] = transforms_smooth[i,3]
        m[1,1] = transforms_smooth[i,4] + 1     # Add 1 for scale y  
        m[1,2] = transforms_smooth[i,5] 
        
        # Save each smoothened affine matrix 
        smoothened_affine_matrices.append(m) 


        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h) )
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            frame_out = frame_stabilized 
            # Writing diagnostic data on top 
            if diagnosticOverlay == True: 
                height, width, dim = frame_out.shape
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

                # Right shift the matrix overlay
                if overlayRightShift == True: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (width-1500, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (width-1500, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[smoothened m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (width-750, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (width-750, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                # Left shift the matrix overlay
                else: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (200, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (200, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[smoothened m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (950, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (950, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                # Save overlayed frame
                cv2.imwrite("{}\\Frame{}.tif".format(frames_location, i+1), frame_out) 
            if show == True: 
                cv2.imshow("Stabilized Frame", frame_out)
                cv2.waitKey(waitTime) 
            out.write(frame_out)
            
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            if (frame_out.shape[1] > 1980): 
                frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) )) 

            # Writing diagnostic data on top 
            if diagnosticOverlay == True: 
                height, width, dim = frame_out.shape
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

                # Right shift the matrix overlay
                if overlayRightShift == True: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (width-1500, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (width-1500, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[smoothened m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (width-750, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (width-750, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                # Left shift the matrix overlay
                else: 
                    cv2.putText(frame_out, "estimated m = [ {:.3f}, {:.3f}, {:.3f},".format( 
                        estimated_affine_matrices[i][0,0],estimated_affine_matrices[i][0,1],estimated_affine_matrices[i][0,2])
                        , (200, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        estimated_affine_matrices[i][1,0],estimated_affine_matrices[i][1,1],estimated_affine_matrices[i][1,2] )
                        , (200, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                    cv2.putText(frame_out, "[smoothened m = {:.3f}, {:.3f}, {:.3f},".format( 
                        m[0,0],m[0,1],m[0,2] )
                        , (950, height-150), font, 1, (0, 255, 0), 2, cv2.LINE_AA )
                    cv2.putText(frame_out, "               {:.3f}, {:.3f}, {:.3f}]".format( 
                        m[1,0],m[1,1],m[1,2] )
                        , (950, height-85), font, 1, (0, 255, 0), 2, cv2.LINE_AA )

                # Save overlayed frame
                cv2.imwrite("{}\\Frame{}.tif".format(frames_location, i+1), frame_out) 

            if show == True: 
                cv2.imshow("Stabilized Frame", frame_out)
                cv2.waitKey(waitTime) 
            out.write(frame_out)

            # print("Final Frame Out Shape: {}".format(frame_out.shape))
            if show == True: 
                cv2.imshow("Before and After", frame_out)
                cv2.waitKey(waitTime)
            out.write(frame_out)

    # Save diagnostic data 
    if saveDiagnosticData == True: 

        for i,curr_matrix in enumerate(estimated_affine_matrices): 
            # Write matrix data in following format 
            estimated_affine.write( "FramePair: {}...\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1, curr_matrix[0,0], curr_matrix[0,1], curr_matrix[0,2]
                , curr_matrix[1,0], curr_matrix[1,1], curr_matrix[1,2]) ) 

        for i,curr_matrix in enumerate(smoothened_affine_matrices): 
            # Write matrix data in following format 
            smoothened_affine.write( "FramePair: {}...\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1, curr_matrix[0,0], curr_matrix[0,1], curr_matrix[0,2]
                , curr_matrix[1,0], curr_matrix[1,1], curr_matrix[1,2]) ) 
        

    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 

def compareFullAffineAndRigidAffine(videoPath, saveDir, outputFileNames): 

    print("Initializing parameters and save directories...")
    # Create 2 folders for each dataset, and one folder holding comparison videos/frames 
    fullAffineData = createFolder(saveDir, "Full Affine Data - {}".format(outputFileNames))
    rigidAffineData = createFolder(saveDir, "Rigid Affine Data - {}".format(outputFileNames)) 

    comparisonVideos = createFolder(saveDir, "Comparisons - {}".format(outputFileNames))
    destabilizedComparison = createFolder(comparisonVideos, "Misaligned and Aligned") 

    comparisonFull = createFolder(destabilizedComparison, "Full Affine Compare")
    comparisonRigid = createFolder(destabilizedComparison, "Rigid Affine Compare")
    print("Done.\n")

    print("Aligning with full affine...")
    # Run full affine and partial
    stabilizeVideoKalmanAffineControl(videoPath, outputFileNames, saveDir=fullAffineData, 
        saveDiagnosticData=True, show=False, comparison=False, affinePartial=False, overlayRightShift=True) 
    print("Done.\n")
    print("Aligning with rigid affine...") 
    stabilizeVideoKalmanAffineControl(videoPath, outputFileNames, saveDir=rigidAffineData, 
        saveDiagnosticData=True, show=False, comparison=False, affinePartial=True, overlayRightShift=False) 
    print("Done.\n")
    
    print("Generating comparisons with full affine...")
    # Run again but save comparison videos/frames
    stabilizeVideoKalmanAffineControl(videoPath, outputFileNames, saveDir=comparisonFull, 
        saveDiagnosticData=True, show=False, comparison=True, affinePartial=False, overlayRightShift=True)
    print("Done.\n")
     
    print("Generating comparisons with rigid affine...")
    stabilizeVideoKalmanAffineControl(videoPath, outputFileNames, saveDir=comparisonRigid, 
        saveDiagnosticData=True, show=False, comparison=True, affinePartial=True, overlayRightShift=False)
    print("Done.")

    print("Generating comparison frames between full and rigid affine...")
    videoDir1 = "{}\\{}.avi".format(fullAffineData, outputFileNames) 
    videoDir2 = "{}\\{}.avi".format(rigidAffineData, outputFileNames) 

    concatenateVideo(videoDir1, videoDir2, comparisonVideos, "Full VS Rigid Affine", saveFrames=True)
    print("Done. Function has run through all code :)") 


def getTransforms(images, lkWinSize=(15,15), lkMaxLevel=2, tomasiCorners=2000): 
    
    '''
    This function takes in 2 grayscale images and computes the optical flow of key points in the image and computes an estimated 
    affine transform to align the 2 images as if they were in a stack. In more detail, the ShiTomasi corner detection 
    algorithm is used to find key features to track, where they are tracked using the Lucas Kanade optical flow method, 
    where the translation and rotation data is extracted to estimate an affine transformation matrix where 3 distinct 
    signals of dx, dy and da are saved. The signal is then integrated to the pure translation and rotation of x, y and a 
    so that they can be smoothened out using the Kalman filter ("I'VE NEVER SEEN IT FAILL JOJOOOOOO"). The dx, dy and da's for
    each frome are derived and thrown into a generated partial Affine 2D transformation matrix to be aligned. 

    This function accepts separate x, y and a kalman filters to use as a basis for the next integrated x, y and a signal 
    smoothening. 

    Args: 

        (param1): 
            (list[ np.ndarray,... ]): List of images to be transformed. 

    Returns: 


    '''

    # Save each time taken for each iteration to calculate average time and total time taken for execution. 
    iteration_time = [] 
    
    # Make sure that there are at least 2 image s
    try: 
        assert len(images) >= 2 
    except AssertionError: 
        print("getTransforms() takes in at minimum 2 images. You entered: {} images. This function will now return None".format(len(images)) )
        return None 

    # Make sure that all shapes are the same 
    try: 
        for i in range(len(images)): 
            assert images[0].shape == images[i].shape 
    except AssertionError: 
        print("This function requires that all images are of the same size. This function will now return None")
        return None 

    # # Calculate each frame wait time for saved video 
    # waitTime = int(np.around(1000/fps)) 
    # Get frame count
    n_frames = len(images) 

    # Initialize height and width even if not read in video
    height, width = images[0].shape 

    # Read first comparison frame: 
    prev_gray = images[0] 

    
    # Pre-define transformation-storing array 
    transforms = np.zeros( (n_frames-1, 3), np.float32) 

    for i in range(n_frames-1): 

        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                    maxCorners=tomasiCorners,
                                    qualityLevel=0.01,
                                    minDistance=20,
                                    blockSize=3)

        # Read second comparison frame: 
        curr_gray = images[i+1] 

        # Calculate optical flow from features tracked from prev_gray 
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel) #, **lk_params) 
        # Sanity check
        # Check that number of previous points is equal to current points 
        assert prev_pts.shape == curr_pts.shape 

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        #Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        #   m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

        # Extract traslation
        dx = m[0][0,2] 
        dy = m[0][1,2] 


        # Extract rotation angle
        da = np.arctan2(m[0][1,0], m[0][0,0])

        transforms[i] = [dx, dy, da] 

    '''
    Cannot compute trajectory if function is only called for 2 images. Need at least 2 transform 
    matrices to discretely integrate into a trajectory matrix. 
    '''
    # Compute trajectory using cumulative sum of transformations 
    # trajectory = np.cumsum(transforms, axis=0)

    return transforms, len(prev_pts) 


def smoothTransformsKalman(transforms): 
    '''
    Reformats a list of transforms as a python list object to a numpy array of proper size. 
    Trajectories are integrated from the transforms matrix by taking the discrete cummulative
    sum of the transforms dx dy and da into linear trajectories x y and a. 
    '''
    
    # Process transforms 
    original_transforms = np.zeros( (len(transforms), 3), dtype=np.float32) 
    for i in range(len(transforms)): 
        original_transforms[i, 0] = transforms[i, 0] 
        original_transforms[i, 1] = transforms[i, 1] 
        original_transforms[i, 2] = transforms[i, 2]

    trajectories = np.cumsum(original_transforms, axis=0) 

    x_trajectories = [] 
    y_trajectories = [] 
    a_trajectories = [] 

    for i in range(len(trajectories)):
        x_trajectories.append(trajectories[i, 0]) 
        y_trajectories.append(trajectories[i, 1]) 
        a_trajectories.append(trajectories[i, 2]) 

    # Initialize original trajectory formatting to proceed with aligning image stack 
    trajectory = np.zeros( (len(trajectories), 3), dtype=np.float32) 
    for i in range(len(trajectories)): 
        trajectory[i, 0] = x_trajectories[i]
        trajectory[i, 1] = y_trajectories[i] 
        trajectory[i, 2] = a_trajectories[i] 

    A = 1 # No process innovation 
    C = 1 # Measurement 
    B = 0 # No control input 
    Q = 0.0001 # Process noise covariance 
    R = 1 # Measurement covariance 
    # X = 0 # Initial estimate 
    P = 1 # Initial covariance 

    # Dividing initial trajectories by 2 to provide smoother first guess for Kalman
    x_kalman_filter = SingleStateKalmanFilter(A, B, C, trajectory[0][0]//4, P, Q, R) 
    y_kalman_filter = SingleStateKalmanFilter(A, B, C, trajectory[0][1]//4, P, Q, R) 
    a_kalman_filter = SingleStateKalmanFilter(A, B, C, 0, P, Q, R) 

    # Empty list for capturing filter estimates 
    x_kalman_filter_estimates = [] 
    y_kalman_filter_estimates = [] 
    a_kalman_filter_estimates = [] 

    # Simulate data arriving sequentially -- dx 
    for data in x_trajectories: 
        x_kalman_filter.step(0, data)
        x_kalman_filter_estimates.append(x_kalman_filter.current_state())
    # Simulate data arriving sequentially -- dy 
    for data in y_trajectories: 
        y_kalman_filter.step(0, data)
        y_kalman_filter_estimates.append(y_kalman_filter.current_state())
    # Simulate data arriving sequentially -- da
    for data in a_trajectories: 
        a_kalman_filter.step(0, data)
        a_kalman_filter_estimates.append(a_kalman_filter.current_state())
    
    # Store filtered trajectories in appropriate format 
    smoothed_trajectory = np.empty( (len(trajectory), 3))
    for i in range(len(trajectory)): 
        smoothed_trajectory[i] = [ x_kalman_filter_estimates[i], y_kalman_filter_estimates[i], a_kalman_filter_estimates[i] ] 

    # Calculate difference in smoothed_trajectory and trajectory
    # Essentially same as taking a discrete derivative to find dx, dy and da 
    difference = smoothed_trajectory - trajectory
    
    # Calculate newer transformation array
    transforms_smooth = original_transforms + difference

    return transforms_smooth       

def alignImageStack(images, outputFileNames, smoothed_transforms, saveDir=None, video=True, fps=7, comparison=False, show_frame=False): 
    '''
    This function aligns any number of images given a list of transforms of length len(images). 
    '''
    # Calculate each frame wait time for saved video 
    waitTime = int(np.around(1000/fps)) 

    # Get frame count
    n_frames = len(images) 

    # Initialize height and width even if not read in video
    height, width = images[0].shape 

    # Initialize video writer and video parameters if specified: 
    if video == True: 

        # Define codec for output video: 
        fourcc = cv2.VideoWriter_fourcc(*'DIVX') 

        # Find shape of each image 
        height, width = images[0].shape 
        
        # No comparison, just output video 
        if comparison == False: 
            write_w = width 
            write_h = height 

            out = cv2.VideoWriter("{}\\{}.avi".format(saveDir, outputFileNames), fourcc, fps, (write_w, write_h)) 

        # Comparison, need to change width and height of video writer 
        else: 
            write_w = int(np.around(width*2)) 
            write_h = int(np.around(height))

            out = cv2.VideoWriter("{}\\{}.avi".format(saveDir, outputFileNames), fourcc, fps, (write_w, write_h)) 
 
    # Initialize height and width 
    for i in range(len(images)):

        # Read image/frame to be transformed   
        frame = images[i] 
        
        # Extract transformations from the smoothed transforms array 
        if len(images) == 1: 
            dx = smoothed_transforms[0]
            dy = smoothed_transforms[1]
            da = smoothed_transforms[2]
        else: 
            dx = smoothed_transforms[i, 0] 
            dy = smoothed_transforms[i, 1] 
            da = smoothed_transforms[i, 2] 
        
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            frame_out = frame_stabilized 

            if show_frame == True: 
                cv2.imshow("Stabilized Frame", frame_out)
                cv2.waitKey(waitTime) 
            cv2.imwrite("{}\\{}.tif".format(saveDir, outputFileNames), frame_out) 
            
            if video == True: 
                out.write(frame_out)
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            if (frame_out.shape[1] > 1980): 
                frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) ))
            if show_frame == True: 
                cv2.imshow("Before and After", frame_out)
                cv2.waitKey(waitTime)
            cv2.imwrite("{}\\{}.tif".format(saveDir, outputFileNames), frame_out)
            if video == True: 
                out.write(frame_out)

    if video == True: 
        out.release() 

    cv2.destroyAllWindows() 

def alignImagePair(src1, src2, outputFileNames, smoothed_transforms, saveDir=None, comparison=False): 
    '''
    Does same thing as alignImageStack() except with only 2 images at a time. Reads 2 images, and applies
    a specified affine transform that should be obtained from function smoothTransforms(). 
    '''

    # Make sure first and second image are of same size
    try: 
        assert src1.shape == src2.shape 
    except AssertionError: 
        print("Images to be aligned need to have same dimensions. Function will now return None") 
        return None 

    # Initialize height and width even if not read in video
    height, width = src1.shape 
 
    frames = [src1, src2] 
    # Initialize height and width 
    for i in range(2):

        # Read image/frame to be transformed   
        frame = frames[i]
        
        # Extract transformations from the smoothed transforms array 
        dx = smoothed_transforms[0]
        dy = smoothed_transforms[1]
        da = smoothed_transforms[2]
        
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            frame_out = frame_stabilized 
            cv2.imwrite("{}\\{}.tif".format(saveDir, outputFileNames), frame_out) 
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            if (frame_out.shape[1] > 1980): 
                frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) ))
            cv2.imwrite("{}\\{}.tif".format(saveDir, outputFileNames), frame_out) 

    cv2.destroyAllWindows() 


def stabilizeVideoGrid(videoDir, outputFileNames, kalman=True, saveDir=None, graphFrames=(None, None), comparison=False, smoothing_radius=40, fps=5 
    , lkWinSize=(15,15), lkMaxLevel=2, lkCriteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), tomasiCorners=2000, grid=True 
    , grid_dimensions=40, initial_offset=10, magnitude_threshold=0.1, median_threshold=1.5): 

    '''
    Sets grid coordinates to be used to calculate RESIDUAL FLOW of an already stabilized video from snrTool.stabilizeVideoKalman(). 

    '''
    waitTime = int(np.around(1000/fps))

    # Read input video
    cap = cv2.VideoCapture(videoDir)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # Set up videowriter object with appropriate widths 
    if (comparison==False) and (saveDir != None): 
    # Set up output video
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames + ".avi", fourcc, fps, (w, h))
    elif saveDir != None: 
        # Check if total concatenated is less than 1980 (990 for now because not concatenated yet) 
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 990:
            write_w = int(np.around(w/2)) 
            write_h = int(np.around(h/2))
        else: 
            write_w = w 
            write_h = h 
        out = cv2.VideoWriter(saveDir + "/" + outputFileNames + "_Comparison.avi", fourcc, fps, (int(np.around(write_w*2)), write_h))
        # print("VideoWriter WxH Dimensions: {} x {}".format(int(np.around(w*2)), h))
    _, prev = cap.read() 
    
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 
    
    for i in range(n_frames-2):
        # Get grid coordinates to track with LK Optical Flow 
        pointsToTrack = getGridPoints(prev, grid_dimensions, initial_offset, evenGrid=True, wrap_for_cv2=True) 
        prev_pts = np.asarray(pointsToTrack,dtype=np.float32) 

        # Read next frame
        success, curr = cap.read() 
        if not success: 
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, maxLevel=lkMaxLevel) #, **lk_params) 
        # Sanity check
        # Check that number of previous points is equal to current points 
        assert prev_pts.shape == curr_pts.shape 

        # -------------------------- Apply magnitude and median filters -------------------------- #
        window_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        window_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        y_threshold = window_height*magnitude_threshold
        x_threshold = window_height*magnitude_threshold 

        dx_list = [] 
        dy_list = [] 
        for j in range(len(prev_pts)): 
            curr_dx = curr_pts[j][0][0] - prev_pts[j][0][0] 
            curr_dy = curr_pts[j][0][1] - prev_pts[j][0][1] 

            dx_list.append(curr_dx) 
            dy_list.append(curr_dy) 

        # Apply magnitude_threshold filter 
        for k in range(len(prev_pts)): 
            if (np.abs(dx_list[k]) < np.abs(x_threshold)) and (np.abs(dy_list[k]) < np.abs(y_threshold)): 
                status[k] = 1 
            else: 
                status[k] = 0 

        # Apply median_threshold filter if threshold value is available 
        x_median = np.median(dx_list)
        y_median = np.median(dy_list) 

        x_threshold = x_median*median_threshold
        y_threshold = y_median*median_threshold 

        for l in range(len(prev_pts)): 
            if (np.abs(dx_list[l]) < np.abs(x_threshold)) and (np.abs(dy_list[l]) < np.abs(y_threshold)): 
                status[l] = 1 
            else: 
                status[l] = 0
        
        # Filter only valid points from LK optical flow 
        idx = np.where(status==1)[0] 
        prev_pts = prev_pts[idx] 
        curr_pts = curr_pts[idx] 

        # -------------------------- Points filtered finished -------------------------- #
        #Find transformation matrix
        try: 
            m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        except cv2.error: 
            pass
        # Extract traslation
        dx = m[0][0,2] 
        dy = m[0][1,2] 

        # Extract rotation angle
        da = np.arctan2(m[0][1,0], m[0][0,0])

        # Store transformation
        transforms[i] = [dx,dy,da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame Pairs: " + str(i+2) +  "/" + str(n_frames-2) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # print("Trajectories, x, y and a: {}".format(trajectory)) 

    if graphFrames[0] == None: 
        # Start subplot of dx, dy, and da's 
        fig, ax = plt.subplots(2,3, figsize=(40,15), sharex=True)  
        plt.setp(ax, xticks=[])
        ax[0, 0].plot(transforms[:,0])
        ax[0, 0].set_title('original dx') 

        ax[0, 1].plot(transforms[:,1])
        ax[0, 1].set_title('original dy') 

        ax[0, 2].plot(transforms[:,2])
        ax[0, 2].set_title('original da') 

        ax[1,0].plot(trajectory[:,0])
        ax[1,0].set_title('integrated translation x') 

        ax[1,1].plot(trajectory[:,1])
        ax[1,1].set_title('cintegrated translation y') 

        ax[1,2].plot(trajectory[:,2])
        ax[1,2].set_title('integrated rotation a') 

        # Save figure into save directory specified in parameters. 
        fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + '.png')
    # When graphing frame boundaries are set: 
    else: 
        # Start subplot of dx, dy, and da's 
        fig, ax = plt.subplots(2,3, figsize=(40,15))  
        ax[0, 0].plot(transforms[:,0][graphFrames[0]:graphFrames[1]])
        ax[0, 0].set_title('original dx') 

        ax[0, 1].plot(transforms[:,1][graphFrames[0]:graphFrames[1]])
        ax[0, 1].set_title('original dy') 

        ax[0, 2].plot(transforms[:,2][graphFrames[0]:graphFrames[1]])
        ax[0, 2].set_title('original da') 

        ax[1,0].plot(trajectory[:,0][graphFrames[0]:graphFrames[1]])
        ax[1,0].set_title('integrated translation x') 

        ax[1,1].plot(trajectory[:,1][graphFrames[0]:graphFrames[1]])
        ax[1,1].set_title('integrated translation y') 

        ax[1,2].plot(trajectory[:,2][graphFrames[0]:graphFrames[1]])
        ax[1,2].set_title('integrated rotation a') 

        # Save figure into save directory specified in parameters. 
        fig.savefig(saveDir + "/" + outputFileNames + 'Pre-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) +  '.png')

    if kalman == False: 
        # Calculate smoothed trajectories using smoothing_radius 
        smoothed_trajectory = smooth(trajectory) 
    else: 
        x_trajectory = [] 
        y_trajectory = [] 
        a_trajectory = [] 
        for i in range(len(trajectory)): 
            x_trajectory.append(trajectory[i][0])
            y_trajectory.append(trajectory[i][1]) 
            a_trajectory.append(trajectory[i][2])  

        A = 1 # No process innovation 
        C = 1 # Measurement 
        B = 0 # No control input 
        Q = 0.0001 # Process noise covariance 
        R = 1 # Measurement covariance 
        # X = 0 # Initial estimate 
        P = 1 # Initial covariance 

        # Dividing initial trajectories by 2 to provide smoother first guess for Kalman
        x_kalman_filter = SingleStateKalmanFilter(A, B, C, trajectory[0][0]//4, P, Q, R) 
        y_kalman_filter = SingleStateKalmanFilter(A, B, C, trajectory[0][1]//4, P, Q, R) 
        a_kalman_filter = SingleStateKalmanFilter(A, B, C, 0, P, Q, R) 

        # Empty list for capturing filter estimates 
        x_kalman_filter_estimates = [] 
        y_kalman_filter_estimates = [] 
        a_kalman_filter_estimates = [] 

        # Simulate data arriving sequentially -- dx 
        for data in x_trajectory: 
            x_kalman_filter.step(0, data)
            x_kalman_filter_estimates.append(x_kalman_filter.current_state())
        # Simulate data arriving sequentially -- dy 
        for data in y_trajectory: 
            y_kalman_filter.step(0, data)
            y_kalman_filter_estimates.append(y_kalman_filter.current_state())
        # Simulate data arriving sequentially -- da
        for data in a_trajectory: 
            a_kalman_filter.step(0, data)
            a_kalman_filter_estimates.append(a_kalman_filter.current_state())
        
        # Store filtered trajectories in appropriate format 
        smoothed_trajectory = np.empty( (len(trajectory), 3))
        for i in range(len(trajectory)): 
            smoothed_trajectory[i] = [ x_kalman_filter_estimates[i], y_kalman_filter_estimates[i], a_kalman_filter_estimates[i] ] 
            # smoothed_trajectory.append( [x_kalman_filter_estimates[i], y_kalman_filter_estimates[i], a_kalman_filter_estimates[i]] )               # potential error here formatting wise 
        print("Smoothed Trajectories, x, y and a: {}".format(smoothed_trajectory)) 

    # Calculate difference in smoothed_trajectory and trajectory
    # Essentially same as taking a discrete derivative to find dx, dy and da 
    difference = smoothed_trajectory - trajectory
    
    # Calculate newer transformation array
    transforms_smooth = transforms + difference      

    if graphFrames[0] == None: 
        fig, ax = plt.subplots(2, 3, figsize=(40,15)) 
        ax[0,0].plot(smoothed_trajectory[:,0])
        if kalman==True: 
            ax[0,0].set_title('Kalman filtered trajectory x')
        else: 
            ax[0,0].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory x')

        ax[0,1].plot(smoothed_trajectory[:,1])
        if kalman==True: 
            ax[0,1].set_title('Kalman filtered trajectory y')
        else: 
            ax[0,1].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory y')

        ax[0,2].plot(smoothed_trajectory[:,2])
        if kalman==True: 
            ax[0,2].set_title('Kalman filtered trajectory a')
        else: 
            ax[0,2].set_title('Smooth_R=' + str(smoothing_radius) +  '_filtered trajectory a')

        ax[1,0].plot(transforms_smooth[:,0])
        ax[1,0].set_title('Smoothed transforms dx')

        ax[1,1].plot(transforms_smooth[:,1])
        ax[1,1].set_title('Smoothed transforms dy')

        ax[1,2].plot(transforms_smooth[:,2])
        ax[1,2].set_title('Smoothed transforms da')

        if kalman==True: 
            fig.savefig(saveDir + "/" + outputFileNames + '_Kalman_Post-Transform' +  '.png')
        else: 
            fig.savefig(saveDir + "/" + outputFileNames + '_Roll_Radius=' + str(smoothing_radius) + '_Post-Transform' +  '.png')

    else: 
        fig, ax = plt.subplots(2, 3, figsize=(40,15)) 
        ax[0,0].plot(smoothed_trajectory[:,0][graphFrames[0]:graphFrames[1]])
        ax[0,0].set_title('Kalman filtered trajectory x')

        ax[0,1].plot(smoothed_trajectory[:,1][graphFrames[0]:graphFrames[1]])
        ax[0,1].set_title('Kalman filtered trajectory y')

        ax[0,2].plot(smoothed_trajectory[:,2][graphFrames[0]:graphFrames[1]])
        ax[0,2].set_title('Kalman filtered trajectory a')

        # Calculate difference in smoothed_trajectory and trajectory
        difference = smoothed_trajectory - trajectory
        
        # Calculate newer transformation array
        transforms_smooth = transforms + difference   
        ax[1,0].plot(transforms_smooth[:,0][graphFrames[0]:graphFrames[1]])
        ax[1,0].set_title('Smoothed Transforms dx')

        ax[1,1].plot(transforms_smooth[:,1][graphFrames[0]:graphFrames[1]])
        ax[1,1].set_title('Smoothed Transforms dy')

        ax[1,2].plot(transforms_smooth[:,2][graphFrames[0]:graphFrames[1]])
        ax[1,2].set_title('Smoothed Transforms da')

        if kalman==True: 
            fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_Kalman' +  '.png')
        else: 
            fig.savefig(saveDir + "/" + outputFileNames + ' Post-Transform' + ' From Frames ' + str(graphFrames[0]) + '-' + str(graphFrames[1]) + '_Roll_Radius=' + str(smoothing_radius)  +  '.png')

    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        # Read next frame
        success, frame = cap.read() 
        if not success:
            break
    
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]
        
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        # Normalize da's by 2 because rotations already detected in original affine fitter 
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))
        
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
        
        if comparison==False: 
            frame_out = frame_stabilized 
            cv2.imshow("Stabilized Frame", frame_out)
            out.write(frame_out)
            cv2.waitKey(waitTime) 
        
        else: 
            # Create comparison window
            frame_out = cv2.hconcat([frame, frame_stabilized])
            # If the image is too big, resize it.
            if (frame_out.shape[1] > 1980): 
                frame_out = cv2.resize(frame_out, (int(np.around(frame_out.shape[1]/2)), int(np.around(frame_out.shape[0]/2)) ))
            # print("Final Frame Out Shape: {}".format(frame_out.shape))
            cv2.imshow("Before and After", frame_out)
            out.write(frame_out)
            cv2.waitKey(waitTime)

    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 

def transformImage(src, x_translation, y_translation, noiseFactor=None): 
    '''
    This function takes an image and applies a specified x and y 2D affine transformation 
    with capability to add a random Gaussian noise component. 
    
    Args: 
        param1: 
            (np.ndarray): Input image to be transformed. 
        param2: 
            (int): Amount of transform in the x direction. 
        param3: 
            (int): Amount of transform in the y direction. 
        param4: 
            (float): Scaling of noise from 0 to 1. 0 or None is no noise, 
            and 1 is noise scaled to the translation parameter inputs. 
    '''
    rows, cols = src.shape 
    if noiseFactor==None: 
        transform = np.float32([ [1, 0, x_translation], [0, 1, y_translation]])
    else: 
        transform = np.float32([ [1, 0, x_translation + x_translation*noiseFactor*np.random.randn()], [0, 1, y_translation + y_translation*noiseFactor*np.random.randn() ]])
    warped_img = cv2.warpAffine(src, transform, (cols, rows)) 
    
    return warped_img

def transformFrames(imgDir, writeDir, noiseFactor, frameName='Frames', writeVideoDir=None, shift_percent=25, warp_percent=100, subsample=None): 
    '''
    This function takes in a stack of images saved in a file directory and iterates a translation affine 
    transform throughout the stack with an added option of adding a Gaussian distribution of translation noise. 
    
    Args: 
        param1: 
            (str): Image stack file directory. 
        param2: 
            (str): File directory to write saved translated frames. 
        param3: 
            (float): Scaling of noise from 0 or None to 1.0. 
        param4: 
            (int): Percentage of image dimensions to increase border size. 
                   (shift_percent=25 will start image 25% shifted top left corner)
        param5: 
            (int): Percentage of image corner travel. 
                   (warp_percent=100 has image travelling from top left corner to bottom right corner.) 
        param6: 
            (float): If not None, will subsample all images with specified sample factor.
    
        
    '''
    read_dir = orderFrames(imgDir, frameName=frameName) 

    first_image = cv2.imread(read_dir[0], 0) 
    h, w = first_image.shape 
    
    if subsample!= None: 
        img = cv2.resize(first_image, (int(w*subsample), int(h*subsample)) )
    else: 
        img = first_image 
    
    h,w = img.shape 

    border_fill_h = int(np.around(h*(shift_percent/100))) 
    border_fill_w = int(np.around(w*(shift_percent/100))) 
    borderType = cv2.BORDER_CONSTANT 

    borderedImage = cv2.copyMakeBorder(img, border_fill_h, border_fill_h, border_fill_w, border_fill_w, borderType)

    warped = transformImage(borderedImage, int(np.around(-w*(shift_percent/100))) , int(np.around(-h*(shift_percent/100))) )

    warp_factor_w = (w*(warp_percent/100)*2*(shift_percent/100))/(len(read_dir)-1)
    warp_factor_h = (h*(warp_percent/100)*2*(shift_percent/100))/(len(read_dir)-1) 

    for i in range(len(read_dir)-1): 
        cur_img = cv2.imread(read_dir[i+1], 0)
        h, w = cur_img.shape 

        if subsample!= None: 
            cur_img = cv2.resize( cur_img, (int(np.around((w*subsample))), int(np.around(h*subsample))) )
        h,w = cur_img.shape 
        border_fill_h = int(np.around(h*(shift_percent/100))) 
        border_fill_w = int(np.around(w*(shift_percent/100)))    
        borderType = cv2.BORDER_CONSTANT          

        cur_img = cv2.copyMakeBorder(cur_img, border_fill_h, border_fill_h, border_fill_w, border_fill_w, borderType) 

        warped = transformImage(cur_img, int(np.around(-w*0.25)) + warp_factor_w*(i+1), int(np.around(-h*0.25)) + warp_factor_h*(i+1) , noiseFactor=noiseFactor)

        cv2.imwrite(writeDir + "/Frames" + str(i+1) + ".tif", warped)
    
    if writeVideoDir != None:
        framesToVideo(writeDir, saveVideoDir_includeName="{}/transformed.avi".format(writeVideoDir), fileName='Frames') 
    
def cutVideo(videoDir, saveDir, videoName, startFrame=1, numFrames=100, fps=15, resize=None): 
    waitTime = int(np.around(1000/fps)) 

    cap = cv2.VideoCapture(videoDir) 
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize==None: 
        out = cv2.VideoWriter("{}/{}.avi".format(saveDir, videoName), fourcc, fps, (w,h)) 
    else: 
        resized_w = int(np.around(resize*w ))
        resized_h = int(np.around(resize*h )) 
        print("w = {}, h = {}".format(resized_w, resized_h)) 
        out = cv2.VideoWriter("{}/{}.avi".format(saveDir, videoName), fourcc, fps, (resized_w,resized_h)) 

    # Skip to the start frame 
    for i in range(startFrame-1): 
        _, frame = cap.read() 
    
    # Read and save specified frames to output video 
    for i in range(numFrames): 
        _, frame = cap.read() 
        if resize != None: 
            newX = int(np.around(resize*w ))
            newY = int(np.around(resize*h )) 
            print("Saved w={}, Saved h={}".format(newX, newY))
            frame = cv2.resize(frame, (newX, newY) ) 
        out.write(frame) 
        print("Frame {}/{} from {} saved...".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(n_frames), videoDir ) ) 

        cv2.imshow("Saved Frames", frame) 
        cv2.waitKey(waitTime) 

    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 

def getGridPoints(src, grid_dimensions, initial_offset, evenGrid=False, wrap_for_cv2=False): 
    pointsToTrack = []
    if src.shape[-1] == 3: 
        height, width, dimensions = src.shape 
    else: 
        height, width = src.shape 

    y, x = initial_offset, initial_offset
    
    if evenGrid == False: 
        y_interval = height//grid_dimensions
        x_interval = width//grid_dimensions 
    else: 
        y_interval = height//grid_dimensions
        x_interval = height//grid_dimensions 

    if wrap_for_cv2 == False: 
        for i in range(grid_dimensions): 
            pointsToTrack.append((x,y)) 
            for j in range(grid_dimensions): 
                x += x_interval
                pointsToTrack.append((x, y)) 
            y += y_interval
            x = initial_offset 
    else: 
        counter = 0 
        for i in range(grid_dimensions): 
            pointsToTrack.append([])
            pointsToTrack[counter].append((x,y)) 
            counter += 1 
            for j in range(grid_dimensions): 
                x += x_interval
                pointsToTrack.append([]) 
                pointsToTrack[counter].append((x, y)) 
                counter += 1 
            y += y_interval
            x = initial_offset 
            
    return pointsToTrack 

# Not used at the moment. --June 11th 2019--
def filterVectorMap(textFolderDir, saveDir): 
    textDir = orderFrames(textFolderDir, frameName='Frame', imgType='.txt')

    for i in range(len(textDir)): 
        vectorFile = open("{}/Frame{}.txt".format(saveDir, i+1), 'w')

        curr_data = np.loadtxt(textDir[i]) 

        dx_list = curr_data[:,2] 
        dy_list = curr_data[:,3] 

        median_filtered_dx = medfilt(dx_list) 
        median_filtered_dy = medfilt(dy_list) 

        writeLines = [] 
        for i in range(len(median_filtered_dx)): 
            writeLines.append("{}\t{}\t{}\t{}\t{}".format(curr_data[:,0], curr_data[:,1], ))


# --------------------------------------- Haar Wavelet Transforms --------------------------------------- # 

def getBiorthonormalWavelet(src, saveDir=None,figSaveName=None, displayBool=False, normalizeReturnValues=False): 
    '''
    This function takes in a source image 'src' and computes the discrete 
    Haar wavelet transform, outputing the approximatied image (LL), Horizontal 
    detail (LH), Vertical detail (HL), and Diagonal detail (HH) of the image. 

    Args: 
        param1: 
            (np.ndarray): Source image. 
        param2: 
            (str): Figure save directory 
        param3: 
            (str): Name of figure to be saved. 
        param4: 
            (bool): whether to display the figures. 
    Returns: 
        return1: 
            (np.ndarray): Original approximation. 
        return2: 
            (np.ndarray): Horizontal Details. 
        return3: 
            (np.ndarray): Vertical Details. 
        return4: 
            (np.ndarray): Diagonal Details. 

    '''
    print(len(src.shape)) 
    if len(src.shape) > 2: 
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 

    coeffs2 = pywt.dwt2(src, 'bior1.3') 

    LL, (LH, HL, HH) = coeffs2 

    height, width = src.shape 

    if displayBool == True: 
        titles = ['Approximation', 'horizontal detail', 'Vertical detail', 'Diagonal detail'] 
        fig = plt.figure(figsize=(12,3)) 

        for i, a in enumerate([LL, LH, HL, HH]): 
            ax = fig.add_subplot(1, 4, i + 1) 
            ax.imshow(a, interpolation='nearest', cmap=plt.cm.gray) 
            ax.set_title(titles[i], fontsize=10) 
            ax.set_xticks([]) 
            ax.set_yticks([])
        
        fig.tight_layout() 
        plt.show() 

        if saveDir == True and figSaveName != None: 
            plt.savefig("{}/{}.png".format(saveDir, figSaveName) )

    if normalizeReturnValues == True: 
        normalized = [] 
        for i, a in enumerate([LL, LH, HL, HH]): 
            normalized_img = np.zeros_like(src) 
            normalized_img = ( cv2.normalize(a, normalized_img, 0, 255, cv2.NORM_MINMAX) ).astype(np.uint8)
            normalized.append(normalized_img) 

        return normalized[0], normalized[1], normalized[2], normalized[3] 
            
    return LL, LH, HL, HH 

def getHaarWavelet(src, saveDir=None,figSaveName=None, displayBool=False, normalizeReturnValues=False): 
    '''
    This function takes in a source image 'src' and computes the discrete 
    Haar wavelet transform, outputing the approximatied image (LL), Horizontal 
    detail (LH), Vertical detail (HL), and Diagonal detail (HH) of the image. 

    Args: 
        param1: 
            (np.ndarray): Source image. 
        param2: 
            (str): Figure save directory 
        param3: 
            (str): Name of figure to be saved. 
        param4: 
            (bool): whether to display the figures. 
    Returns: 
        return1: 
            (np.ndarray): Original approximation. 
        return2: 
            (np.ndarray): Horizontal Details. 
        return3: 
            (np.ndarray): Vertical Details. 
        return4: 
            (np.ndarray): Diagonal Details. 

    '''
    print(len(src.shape)) 
    if len(src.shape) > 2: 
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 

    coeffs2 = pywt.dwt2(src, 'haar') 

    LL, (LH, HL, HH) = coeffs2 

    height, width = src.shape 

    if displayBool == True: 
        titles = ['Approximation', 'horizontal detail', 'Vertical detail', 'Diagonal detail'] 
        fig = plt.figure(figsize=(12,3)) 

        for i, a in enumerate([LL, LH, HL, HH]): 
            ax = fig.add_subplot(1, 4, i + 1) 
            ax.imshow(a, interpolation='nearest', cmap=plt.cm.gray) 
            ax.set_title(titles[i], fontsize=10) 
            ax.set_xticks([]) 
            ax.set_yticks([])
        
        fig.tight_layout() 
        plt.show() 

        if saveDir == True and figSaveName != None: 
            plt.savefig("{}/{}.png".format(saveDir, figSaveName) )

    if normalizeReturnValues == True: 
        normalized = [] 
        for i, a in enumerate([LL, LH, HL, HH]): 
            normalized_img = np.zeros_like(src) 
            normalized_img = ( cv2.normalize(a, normalized_img, 0, 255, cv2.NORM_MINMAX) ).astype(np.uint8)
            normalized.append(normalized_img) 

        return normalized[0], normalized[1], normalized[2], normalized[3] 
            
    return LL, LH, HL, HH 


def partitionImage(src, windowLength): 
    '''
    Takes in source image and returns array referencing array slice coordinates of each pixel 
    pertaining to each window, depending on specified parameter windowLength. In short, this
    function returns the top left coordinates of each interrogation window of specified 
    window Length in the source image (src). 
    '''

    if (len(src.shape)) > 2: 
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) 
    
    height, width = src.shape 
    window_height_num = int( np.around(height/windowLength) ) 
    window_width_num = int( np.around(width/windowLength) ) 

    return_window = np.zeros( (window_height_num, window_width_num, 2), dtype=np.uint32 )

    curr_x = 0 
    curr_y = 0 

    # Iterating through each window 
    for i in range(window_height_num): 
        for j in range(window_width_num):
            return_window[i,j] = [curr_y, curr_x] 
            curr_x += windowLength 
        # Need to reset curr_x to 0 
        curr_x = 0 
        curr_y += windowLength 
    
    return return_window 

def getEdgeMapPartitioned(cHn, cVn, cDn, windowLength): 
    # Get image dimensions 
    assert (cHn.shape == cVn.shape) and (cHn.shape == cDn.shape)
    height, width = cHn.shape 

    edge_map = np.zeros_like(cHn) 

    for i in range(height): 
        for j in range(width): 
            edge_map[i,j] = math.sqrt( cHn[i,j]**2 + cVn[i,j]**2 + cDn[i,j]**2 )

    # Testing block creating edge_map by averaging horizontal vertical and diagonal domains 
    # for i in range(height): 
    #     for j in range(width): 
    #         edge_map[i,j] = int(np.around( (cHn[i,j] + cVn[i,j] + cDn[i,j])/3 ))


    # Now edge_map is found for every pixel, find local maxima in each partitioned window:  
    partition_coords = partitionImage(cHn, windowLength=windowLength) 
    window_height_num, window_width_num, dimensions = partition_coords.shape 

    curr_max = 0 
    # return_window is to be same size as partitioned windows (one edge_map value per grid) 
    return_window = np.zeros( (window_height_num, window_width_num), dtype=np.float64)

    # For each individual partition with coordinates referenced by partition_coords: 
    for i in range(window_height_num): 
        for j in range(window_width_num): 
            
            # For each individual pixel within a partition 
            # Initialize offsets for partition windows 
            curr_y = 0 
            curr_x = 0 

            for k in range(windowLength): 
                for l in range(windowLength): 
                    # Expecting an index error when the image dimensions are not even numbers 
                    try: 
                        if edge_map[ (partition_coords[i,j,0] + curr_y), (partition_coords[i,j, 1] + curr_x) ] >= curr_max:
                            curr_max =  edge_map[ (partition_coords[i,j,0] + curr_y), (partition_coords[i,j, 1] + curr_x) ]
                    except IndexError: 
                        # print("The image dimensions are not even so an index error is thrown in snrTool.py")
                        # print("The operation level window height, width are {},{}".format(height, width)) 
                        pass
                    # Increment x offset in partitioned window 
                    if curr_x < windowLength-1: 
                        curr_x += 1 

                # Reset x offset as the iterations in x direction are over. 
                curr_x = 0 

                # Increment y offset in partitioned window 
                if curr_y < windowLength-1:
                    curr_y += 1 
                
            # Reset offsets for next window partitioned 
            curr_y = 0 
            # Save local maxima into return_window of same size as partitioned windows 
            return_window[i,j] = curr_max 
            # Reset curr_max to 0 after checking each partition 
            curr_max = 0 

    return return_window 
    
def thresholdEdgeMap(edgeMap, threshold_value, show_thresholded=False, view_image_scale = 6): 
    '''
    This function takes in an edge map generated by getEdgeMapPartitioned()
    and returns the original edgeMap with an added dimension to the array 
    of either value 0 or 1 that identifies the point is not, or is an edge.
    It is important to note the formatting of the coordinate return values 
    of the edges are in (y, x) 
    '''
    height, width = edgeMap.shape 
    edges = edgeMap.copy() 
    edges = edges[..., np.newaxis]

    for i in range(height): 
        for j in range(width): 
            if edgeMap[i,j] >= threshold_value: 
                edges[i,j,0] = 1 
            else: 
                edges[i,j,0] = 0

    # If we want to visualize it to test thresholding... 
    if show_thresholded == True: 
        # Create black mask to display edges as white
        mask = np.zeros_like(edgeMap, dtype=np.uint8) 
        for i in range(height): 
            for j in range(width): 
                if edges[i,j,0] == 1: 
                    mask[i,j] = 255 
                elif edges[i,j,0] == 0: 
                    mask[i,j] = 0 
        mask = resizeImage(mask, view_image_scale) 
        printImage('Edges with threshold: {}'.format(threshold_value), mask) 
    
    return edges 
        
def getHaarWaveletMultiLevel(src, levels, normalize=False, displayBool=False): 
    '''
    This function takes an image and computes the discrete haar wavelet transform 
    at a specified number of decompostiion levels. ONLY 3 LEVELS SUPPORTED AT THE 
    MOMENT. Any more and it will be too computationally expensive. Any less and 
    it won't be worth the bandwidth. 

    Args: 

        (param1): 
            (np.ndarray): Input image to be transformed. 
        (param2): 
            (int): Number of wavelet decomposition levels. 
        (param3): 
            (bool): Whether to normalize final results to 0-255 range. 
                    Not recommended as this hinders wavelet analysis. 
        (param4): 
            (bool): Whether to display the results or not. 
        (param5): 
            (str): Where to save the displayed results. 
    
    Returns: 

        (return1): 
            (list[np.ndarray...]): First level of decomposition (smallest) 
        (return2): 
            (list[np.ndarray]): Second level of decomposition (middle) 
        (return3): 
            (list[np.ndarray]): Third level of decomposition (Largest, still half size of original image) 
    '''
    c = pywt.wavedec2(src, 'haar', mode='symmetric', level=levels) 

    # Save original image in case further processing necessary 
    aaa = c[0] 

    # Save return data to named levels --> Level 1 is highest livel (smallest) 
    [h1, v1, d1] = c[1] 
    [h2, v2, d2] = c[2] 
    [h3, v3, d3] = c[3] 
    
    if normalize == True: 
        b = [h1, v1, d1, h2, v2, d2, h3, v3, d3]  
        normalized = [] 
        for image in b: 
            normalized.append( normalizeImage(image) ) 
        
        [h1, v1, d1] = [normalized[0], normalized[1], normalized[2]] 
        [h2, v2, d2] = [normalized[3], normalized[4], normalized[5]] 
        [h3, v3, d3] = [normalized[6], normalized[7], normalized[8]] 

    return [h1, v1, d1], [h2, v2, d2] , [h3, v3, d3]

def getPerceptualBlur(src, sobel_threshold, detail_orientation, search_radius): 
    '''
    This function is to work in tandem with getHaarWaveletMultilevel() written above. It is to implement 
    an altered version of the perceptive blur algorithm described in the academic article "No-Reference Objective Wavelet Based
    Noise Immune Image Sharpness Metric" by R.Ferzli and Lina J.Karam. Originally, function should take in some 
    wavelet transformed image as 'src', find edges in the x and y direction, calculate and return the average
    orthogonal edge length for the image as a perceptual image sharpness metric. 

    Instead, this function will calculate an orthogonal gradient difference between the 2 extrema of each edge. 
    The gradient difference will then be normalized by 100 to output a decimal metric between 0 and 1. 

    Args:

        (param1): 
            (np.ndarray): Source array inputted. Currently, src input should be normalized haar wavelet transform. 
        (param2): 
            (int): Threshold value used to find edges from sobel thresholding algorithm. 
        (param3): 
            (str): "x" or "X" for X-direction haar wavelet detail orientation. 
                   "y" or "Y" for Y-direction haar wavelet detail orientation. 
        (param4): 
            (int): Radius of search to determine gradient coefficient for each edge detected. 
        
    Returns: 

        (return1): 
            (float): Perceptual blur coefficient normalized by 100. 

    '''
    # Initialize sobel filter parameters
    scale = 1 
    delta = 0 
    ddepth = cv2.CV_16S

    # If image is in coloured channels, change to grayscale 
    if len(src.shape) == 3: 
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 

    # Calculate x gradiants 
    if detail_orientation == 'x' or detail_orientation == 'X': 
        grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT) 

        # convert to usable scale 
        abs_grad = cv2.convertScaleAbs(grad_x) 
    
    # Calculate y gradient
    elif detail_orientation == 'y' or detail_orientation == 'Y': 

        # convert to usable scale 
        grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad = cv2.convertScaleAbs(grad_y) 
    
    else: 
        print("Invalid detail_orientation input! Try again idiot().")
        return None 
    


    # Standardize calculations as 'edges'. No need to specify orientations as dealt with above. 
    edges = [] 

    height, width = abs_grad.shape

    # Iterate through 
    for i in range(height): 
        for j in range(width): 
            if abs_grad[i,j] >= sobel_threshold: 
                edges.append( [i,j] ) 
            if abs_grad[i,j] >= sobel_threshold: 
                edges.append( [i,j] ) 

    # Store discretized gradient coefficients 
    gradients = [] 

    # No need to check detail_orientation as upper bounds and lower bounds function accounts for it 
    # Compute for x-edges 
    for i, edge in enumerate(edges): 
        
        # Find edge bounds to calculate gradient coefficient. 
        curr_upperBounds, curr_lowerBounds = findEdgeBounds(src, edge, detail_orientation, search_radius)

        # Sum gradients of each bound 
        sum_lowerBound = 0
        sum_upperBound = 0 

        # Need to except index error as after iterating through last point of each bound it will try to look for the next point 
        # that does not exist. 
        try: 
            for j,point in enumerate(curr_lowerBounds): 
                sum_lowerBound += np.abs( int(src[curr_lowerBounds[j][0], curr_lowerBounds[j][1] ]) - int(src[curr_lowerBounds[j+1][0], curr_lowerBounds[j+1][1]] ) )   
        except IndexError: 
            pass 
        try: 
            for j, point in enumerate(curr_upperBounds): 
                sum_upperBound += np.abs( int(src[curr_upperBounds[j][0], curr_upperBounds[j][1] ]) - int(src[curr_upperBounds[j+1][0], curr_upperBounds[j+1][1]]) )   
        except IndexError: 
            pass 

        # Calculate gradient metric (smaller gradient means sharper image) 
        curr_gradient = np.abs(sum_lowerBound - sum_upperBound) 
        gradients.append(curr_gradient) 

    avg_gradient = sum(gradients) / (len(gradients) * 100 ) 

    return avg_gradient 

# ----------------------------------- BRISQUE METHODS -------------------------------- # 
def AGGDfit(structdis): 
    # Variables to count positive pixels / negative pixels and their squared sum 
    poscount = 0 
    negcount = 0 
    possqsum = 0 
    negsqsum = 0 
    abssum = 0

    poscount = len(structdis[structdis > 0]) # Number of positive pixels
    negcount = len(structdis[structdis < 0]) # Number of negative pixels 

    # Calculate squared sum of positive pixels and negative pixels 
    possqsum = np.sum(np.power(structdis[structdis > 0], 2)) 
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))
    
    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum/negcount))
    rsigma_best = np.sqrt((possqsum/poscount))

    gammahat = lsigma_best/rsigma_best
    
    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum/totalcount, 2)/((negsqsum + possqsum)/totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1)/(m.pow(m.pow(gammahat, 2) + 1, 2))
    
    prevgamma = 0
    prevdiff  = 1e10
    sampling  = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes = [np.float], cache = False)
    
    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best] 

def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while(gam < 10):
        r_gam = tgamma(2/gam) * tgamma(2/gam) / (tgamma(1/gam) * tgamma(3/gam))
        diff = abs(r_gam - rhatnorm)
        if(diff > prevdiff): break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best

def compute_features(img):
    scalenum = 2
    feat = []
    # make a copy of the image 
    im_original = img.copy()

    # scale the images twice 
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im*im, (7, 7), 1.166)
        sigma = (sigma - mu_sq)**0.5
        
        # structdis is the MSCN image
        structdis = im - mu
        structdis /= (sigma + 1.0/255) # Add 1.0/255 to avoid ZeroDivisionError
        
        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)

        # unwrap the best fit parameters 
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best  = best_fit_params[2]
        
        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2)

        # shifting indices for creating pair-wise products
        shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift-1] # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))
            
            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product 
            # best fit the pairwise product 
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best  = best_fit_params[2]

            constant = m.pow(tgamma(1/gamma_best), 0.5)/m.pow(tgamma(3/gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2/gamma_best)/tgamma(1/gamma_best)) * constant

            # append the best fit calculated parameters            
            feat.append(gamma_best) # gamma best
            feat.append(meanparam) # mean shape
            feat.append(m.pow(lsigma_best, 2)) # left variance square
            feat.append(m.pow(rsigma_best, 2)) # right variance square
        
        # resize the image on next iteration
        im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    
    return feat

# function to calculate BRISQUE quality score 
# takes input of the image
def test_measure_BRISQUE(img):

    dis = img 
    # compute feature vectors of the image
    features = compute_features(dis)

    # rescale the brisqueFeatures vector from -1 to 1
    x = [0]
    
    # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
    min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
    
    max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

    # append the rescaled vector to x 
    for i in range(0, 36):
        min = min_[i]
        max = max_[i] 
        x.append(-1 + (2.0/(max - min) * (features[i] - min)))
    
    # load model 
    model = svmutil.svm_load_model('brisque_svm.txt')

    # create svm node array from python list
    x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
    x[36].index = -1 # set last index to -1 to indicate the end.
	
	# get important parameters from model
    svm_type = model.get_svm_type()
    is_prob_model = model.is_probability_model()
    nr_class = model.get_nr_class()
    
    # Checking type of SVM model loaded... 
    if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
        # here svm_type is EPSILON_SVR as it's regression problem
        nr_classifier = 1
    dec_values = (c_double * nr_classifier)()
    
    # calculate the quality score of the image using the model and svm_node_array
    qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)

    return qualityscore

    

def getImageQualityBRISQUE(src, diagnostic=False, SVM_model='brisque_svm.txt'):
    '''
    This function takes in one image and computes the image quality from the 
    proposed Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE). 

    Args: 
        (param1): 
            (np.ndarray): Input image to be processed. 
        (param2): 
            (bool): Whether or not to print diagnostic data. 
        (param3): 
            (str): Directory path to SVM trained characterization model. 

    Returns: 
        (return1): 
            (float): Image quality metric, 0 - 100. 0 is best, 100 is worst. 
    '''

    image_quality = test_measure_BRISQUE(src) 

    return image_quality 


def getSharpnessMetricHaarGradient(src, diagnostic=False, bilateral_filter=None): 
    
    level1, level2, level3 = getHaarWaveletMultiLevel(src, 3, True) 

    # assign more intuitive names
    # All horizontal details level1 to 3  
    horizontal_detail = [level1[0], level2[0], level3[0]] 
    
    # All vertical details level1 to 3 
    vertical_detail = [level1[1], level2[1], level3[1]] 

    gradients_list = [] 
    for i, transform in enumerate(horizontal_detail):

        # Get dimensions of current transform level to determine search radius  
        height, width = transform.shape 
        # Approximately 1/15th of the mean distance of height and width 
        search_radius = int( (height + width) / (2*15) )

        # Compute the gradient coefficient for this level...
        # Want to find vertical edges to contrast the horizontal details of transform 
        gradients_list.append( getPerceptualBlur(transform, 220, 'y', search_radius))

    for i, transform in enumerate(vertical_detail): 

        height, width = transform.shape 
        search_radius = int( (height + width) / (2*15) ) 

        gradients_list.append( getPerceptualBlur(transform, 220, "x", search_radius))

    avg_gradient = sum(gradients_list) / len(gradients_list) 

    return avg_gradient 
        





def findEdgeBounds(src, edge, edge_direction, search_radius): 
    '''
    This function takes in a source image, a single edge, and edge parameters to find
    local edges in a specific direction and radius for comparing and further processing. 

    Args: 

        (param1): 
            (np.ndarray): Input image array. 
        (param2): 
            (tuple): Edge to found edge bounds for in format (y,x) 
        (param3): 
            (str): "x" or "X" for X-direction haar wavelet detail orientation. 
                   "y" or "Y" for Y-direction haar wavelet detail orientation. 
        (param4): 
            (int): Radius of search to determine gradient coefficient for each edge detected. 
    
    Returns: 

        (return1): 
            ([ list[ (y1, x1),...], [ list[ (y1, x1),...]] ): 2 Nested lists of upper and lower bounds. 
    '''
    # Initialize src characteristics to check bounds 
    height, width = src.shape 
    
    # When edge direction is in x 
    if edge_direction == "x" or edge_direction == "X": 

        # set list of local points in correct direction(x or y) 
        upperBounds = [] 
        lowerBounds = [] 

        # Find lower bounds 
        for i in range( search_radius ): 
            # Incase a boundary is hit. Only need to pass so it can just be ignored. 
            try: 
                if  (edge[1] - search_radius + i) >= 0:
                    lowerBounds.append( (edge[0], edge[1] - search_radius + i) ) 
            except IndexError: 
                continue

        # Find upper bounds 
        for i in range( search_radius ):
            # Incase a boundary is hit. Only need to pass so it can just be ignored 
            try: 
                if (edge[1] + 1  + i) < width: 
                    upperBounds.append( (edge[0], edge[1] + 1 + i)) 
            except IndexError: 
                continue 

    # When edge direction is in y 
    elif edge_direction == "y" or edge_direction == "Y": 

        # set list of local points in correct direction(x or y) 
        upperBounds = [] 
        lowerBounds = [] 

        # Find lower bounds 
        for i in range( search_radius ): 
            # Incase a boundary is hit. Only need to pass so it can just be ignored. 
            try: 
                if (edge[0] - search_radius + i) >= 0 : # -1 
                    lowerBounds.append( (edge[0] - search_radius + i, edge[1]) ) 
            except IndexError: 
                continue

        # Find upper bounds 
        for i in range( search_radius ):
            # Incase a boundary is hit. Only need to pass so it can just be ignored 
            try: 
                if (edge[0] + 1 + i) < height: 
                    upperBounds.append( (edge[0] + 1 + i, edge[1]) ) 
            except IndexError: 
                continue 

    return upperBounds, lowerBounds  


def findEdgeLengths(src, edges, edge_direction, search_radius, avg=True): 
    '''

    This function is written to be used in getPerceptualBlur() above. The idea to calculate the edge lengths 
    as a parameter to an image blur metric is proposed in the paper: "Perceptual blur and ringing metrics: 
    application to JPEG2000" by Pina Marziliano, Frederic Dufaux, Stefan Winkler, Touradj Ebrahimi. 

    Args: 

        (param1): 
            (np.ndarray): Input image to look for edge lengths. 
        (param2): 
            (list): List of edges to calculate length of in format (y,x).
        (param3): 
            (string): What type of edge to filter edges in to calculate lengths. 
                      Takes in either strings 'y' or 'x' to specify edge gradients. 
        (param4): 
            (int): Radius around each edge point used to search for local extrema. 
        (param5): 
            (bool): Whether or not to average the list of calculated lengths or not. 
    
    Returns:

        (return1, avg == True):
            (float): Average length of all edges in the source src image. 
        (return1, avg == False): 
            (list): List of all edges and their calculated lengths in format (y, x, length). 


    '''
    # Initialize src characteristics to check bounds 
    height, width = src.shape 
    # Initialize variable storing all edge lengths here 
    edge_lengths = [] 

    # For each edge, want to save range of closest pixels along appropriate axis 
    # to find the local extrema.

    # When edge_direction is in x  
    if edge_direction == "x" or edge_direction == "X": 
    
        # Iterating through all edges 
        for edge in edges: 

            # set list of local points in correct direction(x or y) 
            currPointList = [] 

            # Find lower bounds 
            for i in range( search_radius ): 
                # Incase a boundary is hit. Only need to pass so it can just be ignored. 
                try: 
                    if  (edge[1] - search_radius + i) >= 0:
                        currPointList.append( (edge[0], edge[1] - search_radius + i) ) 
                except IndexError: 
                    continue

            # Find upper bounds 
            for i in range( search_radius ):
                # Incase a boundary is hit. Only need to pass so it can just be ignored 
                try: 
                    if (edge[1] + 1  + i) < width: 
                        currPointList.append( (edge[0], edge[1] + 1 + i)) 
                except IndexError: 
                    continue 

            # Find local maxes 
            # max_index = signal.argrelmax(currPointList)
            currMax = 0 
            currMaxIndex = 0 
            currMin = 255 
            currMinIndex = 0 

            # Check points surrounding the edge for max and min 
            for point in currPointList: 

                # If another edge point is in the search radius 
                # if point in edges: 
                #     currMaxIndex = 0 
                #     currMinINdex = 0 
                #     break 

                if src[point[0], point[1]] > currMax: 
                    currMax = src[point[0], point[1]] 
                    currMaxIndex = point 
                if src[point[0], point[1]] < currMin: 
                    currMin = src[point[0], point[1]] 
                    currMinIndex = point 
            
            if currMaxIndex == 0 or currMinIndex == 0: 
                currMaxIndex = [0, 0]
                currMinIndex = [0, 0] 

            edge_length = np.abs( currMaxIndex[1] - currMinIndex[1] ) 

            edge_lengths.append(edge_length) 
        
         


    # When edge direction is in y 
    elif edge_direction == "y" or edge_direction == "Y": 

        # Iterating through all edges 
        for edge in edges: 

            # set list of local points in correct direction(x or y) 
            currPointList = [] 

            # Find lower bounds 
            for i in range( search_radius ): 
                # Incase a boundary is hit. Only need to pass so it can just be ignored. 
                try: 
                    if (edge[0] - search_radius + i) >= 0 : # -1 
                        currPointList.append( (edge[0] - search_radius + i, edge[1]) ) 
                except IndexError: 
                    continue

            # Find upper bounds 
            for i in range( search_radius ):
                # Incase a boundary is hit. Only need to pass so it can just be ignored 
                try: 
                    if (edge[0] + 1 + i) < height: 
                        currPointList.append( (edge[0] + 1 + i, edge[1]) ) 
                except IndexError: 
                    continue 
            
            # Find local maxes 
            # max_index = signal.argrelmax(currPointList)
            currMax = 0 
            currMaxIndex = 0 
            currMin = 255 
            currMinIndex = 0 

            # Check points surrounding the edge for max and min 
            for point in currPointList: 

                # When an edge point is in the search radius 
                # if point in edges: 
                #     currMaxIndex = 0 
                #     currMinIndex = 0 
                #     break

                if src[point[0], point[1]] > currMax: 
                    currMax = src[point[0], point[1]] 
                    currMaxIndex = point 
                if src[point[0], point[1]] < currMin: 
                    currMin = src[point[0], point[1]] 
                    currMinIndex = point 
            
            if currMinIndex == 0 or currMaxIndex == 0: 
                currMinIndex = [0,0] 
                currMaxIndex = [0,0] 

            edge_length = np.abs( currMaxIndex[0] - currMinIndex[0] ) 

            edge_lengths.append(edge_length) 

    else: 
        print("Please choose either 'x' or 'y' for edge_direction. This function will return None") 

    
    if avg == False: 
        
        return edge_lengths 

    else: 

        try: 
            avg_edge_lengths = sum(edge_lengths) / len(edge_lengths)
        except ZeroDivisionError: 
            return 0 

        return avg_edge_lengths 

def graph1DSignal(src, points, mark_ROI1=None, mark_ROI2=None, name_ROI1=None, name_ROI2=None):  
    fig, ax = plt.subplots(1, 1)
    

def getBlurExtent(src, levels, threshold, saveDir, figSaveName, displayBool=False, normalizeReturnValues=False): 
    '''
    This function follows the algorithms derived from 'Blur Detection for Digital Images Using Wavelet Transform'
    which can be found at this link: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.5508&rep=rep1&type=pdf. 
    This function takes in the base image, number of levels to be computed (recommended is 3), a save directory, and a 
    name to save the files as. The blurness value described in the article will be returned by the function, as well as 
    having diagnostic data sent and saved to the user specified save directory 'saveDir'.

    Important to note this function currently only accepts levels as 3.  
    '''
    # Block to make sure levels is 3 or less 
    try: 
        assert levels < 4 
    except AssertionError: 
        print("This function currently only accepts a max level of 3. This will now return None.") 
        return None 


    # Calculate haar wavelet transforms at specified levels 
    # print("Calculating haar wavelet transforms at {} levels...".format(levels)) 

    '''
    Method using pywt.wavdec2 --> Automatically downsamples first level to half 
    original. This causes sharp edges to be blurred even more 
    coeffs2 = pywt.wavedec2(src, 'haar', level=levels) 
    # Extract image data from haar transforms 
    [cAn, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = coeffs2 
    '''

    # Calculate haar wavelet transform at each level. 
    # Level 1, i = 1 
    coeffs1 = pywt.dwt2(src, 'haar') 
    cAn, (cH1, cV1, cD1) = coeffs1 

    # Downsample to half...
    src = resizeImage(src, 0.5) 
    coeffs2 = pywt.dwt2(src, 'haar') 
    cAn, (cH2, cV2, cD2) = coeffs2 

    # Downsample to half again... 
    src = resizeImage(src, 0.5) 
    coeffs3 = pywt.dwt2(src, 'haar') 
    cAn, (cH3, cV3, cD3) = coeffs3 

    # Compute edge mapping at first level, i = 1 
    edge_map_1 = getEdgeMapPartitioned(cH1, cV1, cD1, 8) 
    edges_1 = thresholdEdgeMap(edge_map_1, threshold) 

    # Compute edge mapping at second level, i = 2 
    edge_map_2 = getEdgeMapPartitioned(cH2, cV2, cD2, 4) 
    edges_2 = thresholdEdgeMap(edge_map_2, threshold) 

    # Compute edge mapping at third level, i = 3  
    edge_map_3 = getEdgeMapPartitioned(cH3, cV3, cD3, 2) 
    edges_3 = thresholdEdgeMap(edge_map_3, threshold) 

    # Debug line to test final partition size: 
    # print("Wavelet transforms calculated and partitioned to size: {}".format(edge_map_1.shape)) 






    # # TESTING BLOCK ------------------------------------------- #
    # edgeMap1 = resizeImage(normalizeImage(edge_map_1), 30) 
    # edgeMap2 = resizeImage(normalizeImage(edge_map_2), 30) 
    # edgeMap3 = resizeImage(normalizeImage(edge_map_3), 30) 

    # printImage('map1', edgeMap1) 
    # printImage('map2', edgeMap2) 
    # printImage('map3', edgeMap3) 
    # # ---------------------------------------------------------- #

    '''
    Rule 1: Finding edge points.  
    ''' 
    # all edges_1/2/3 should have the same shape...
    # assert edge_map_3.shape == edge_map_1.shape and edge_map_3.shape == edge_map_2.shape
    # Initialize prerequisites
    height, width = edge_map_1.shape
    # Edge coordinates appended as format (y,x) 
    edge_points = [] 

    for i in range(height):
        for j in range(width): 
            if ( (edges_1[i,j,0] == 1) or (edges_2[i,j,0] == 1) or (edges_3[i,j,0] == 1) ): 
                edge_points.append( (i,j) )
    
    # # ------- display block for debugging ---------# 
    # mask = np.zeros_like(edge_map_1, dtype=np.uint8) 
    # for i in range(len(edge_points)): 
    #     y, x = edge_points[i] 
    #     mask[y,x] = 255 
    # mask = resizeImage(mask, 10)
    # printImage('all edges found', mask) 
    # # ------- display block for debugging ---------# 
    
    '''
    Rule 2: Find Dirac-Structures or Astep-Structures. 
    Rule 3: Find Roof-Structures or Gstep-Structures. 
    '''
    edge_dirac_or_astep = [] 
    edge_roof_or_gstep = [] 
    for i in range(len(edge_points)): 
        y, x = edge_points[i] 

        if (edge_map_1[y,x] > edge_map_2[y,x]) and (edge_map_2[y,x] > edge_map_3[y,x]): 
            edge_dirac_or_astep.append( (y,x) ) 
        elif (edge_map_1[y,x] < edge_map_2[y,x]) and (edge_map_2[y,x] < edge_map_3[y,x]): 
            edge_roof_or_gstep.append( (y,x) ) 
        
    '''
    Rule 4: Finding Roof structures and appending to edge_roof_or_gstep 
    '''
    edge_roof = [] 
    for i in range(len(edge_roof_or_gstep)): 
        y, x = edge_roof_or_gstep[i] 

        if edge_map_2[y,x] > edge_map_1[y,x] and edge_map_2[y,x] > edge_map_3[y,x]: 
            edge_roof.append( (y,x) ) 
    '''
    Rule 5: Finding blurred edge pixels. 
    '''
    blurred = [] 
    for i in range(len(edge_roof_or_gstep)): 
        y, x = edge_roof_or_gstep[i] 
        if edge_map_1[y,x] < threshold: 
            blurred.append( (y,x) ) 
    
    '''
    2.3 Proposed blur detection scheme 
    '''
    num_edge = len(edge_points) 
    num_da = len(edge_dirac_or_astep) 
    num_rg = len(edge_roof_or_gstep) + len(edge_roof)
    num_brg = len(blurred) 

    # Calculate ratio of Dirac-Structure and Astep-Structure to all edges: 
    per = num_da/num_edge 
    # Calclate how many Roof-Structure and Gstep-Structure edges are blurred: 
    try: 
        blur_extent = num_brg/num_rg 
    except ZeroDivisionError: 
        print("No Roof-Gstep Structures found.")
        return 1 
    # blur_extent is the blur confident coefficient for an image. 
    print("The ratio of blurred Roof-Structure and Gstep-Structure edges is: {}/{}".format(num_brg, num_rg)) 
    return blur_extent, (num_brg, num_rg)  

def calibrateBilateralFilter(d, sigma_start, sigma_increment, iterations, saveDir): 
    folderDir = loadFolderDir() 
    folder_directories = listFolders(folderDir) 

    bestImageIndex = [] 

    for f,folder in enumerate(folder_directories): 
        print("\n\nOpening Image Stack: {}/{}\n\n".format(f+1, len(folder_directories) ) )
        image_stack = orderFrames(folder, '00', imgType='.bmp', startNum=0)

        # Get AFAS data
        curr_folder = folder.split('\\') 
        curr_save_directory = createFolder(saveDir, curr_folder[-1]) 

        # Initialize Average Blur_extent histogram parameters 
        allBlurExtents = [] 

        # For each iteration through a stack 
        for i in range(iterations):
            # Initialize current blur_extents 
            blur_extents = []  

            # For each image in the stack   
            for j in range(len(image_stack)): 
                src = cv2.imread(image_stack[j], 0) 
                # src = cv2.GaussianBlur(src, (3 + int(2*i),3 + int(2*i)), 0)  
                # src = cv2.medianBlur(src, 3 + int(2*i))
                src = cv2.bilateralFilter(src, d, ( sigma_start+int(sigma_increment*i) ), ( sigma_start+int(sigma_increment*i) ) )
                print("Data: Image {}".format(j+1)) 
                blur_extents.append( getBlurExtent(src, 3, 10, None, None)[0] ) 
                print("The Blur-Extent-Coefficient is: {}\n".format(blur_extents[j]))
                
            curr_min = 1
            curr_index = 0 
            for k, blur_coefficient in enumerate(blur_extents): 
                if blur_coefficient < curr_min: 
                    curr_min = blur_coefficient
                    curr_index = k+1
            

            allBlurExtents.append(blur_extents) 
            bestImageIndex.append(curr_index) 
            print("\n\nProcessed Iteration {}/{} of Folder {}/{}\n\n".format(i+1, iterations, f+1, len(folder_directories) )) 
                

        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(30,8) 
        # x = np.linspace(sigma_start, int(sigma_start+sigma_increment*iterations), iterations, dtype=np.uint32)
        # x = np.arange(sigma_start, int(sigma_start+sigma_increment*iterations), step=sigma_increment)
        # y = bestImageIndex 
        x = np.arange( sigma_start, int(sigma_start + sigma_increment*iterations), step=sigma_increment ) 
        y_ticks = np.arange(1, len(blur_extents)+1, step=1) 
        x_ticks = np.arange( sigma_start, int(sigma_start + sigma_increment*iterations), step=20)
        ax[0].plot(x, bestImageIndex) 
        ax[0].set_yticks(y_ticks)   
        ax[0].set_xticks(x_ticks) 
        ax[0].set_title('Index of Best Image VS. Bilateral Blur Factor (d={})'.format(d)) 
        ax[0].set_xlabel('Sigma Blur Factor') 
        ax[0].set_ylabel('Image Index') 

        x = np.arange(1, len(blur_extents)+2, step=1, dtype=np.uint32)

        ax[1].hist(bestImageIndex, align='left', bins=x, rwidth=0.25)  
        ax[1].set_title('Index of Best Images Histogram') 
        ax[1].set_xlabel('Image Index') 
        ax[1].set_ylabel('Num') 
        ax[1].set_xticks(x) 
        ax[1].set_xticklabels(x[:-1]) 

        # Calculate average blur extent for each folder opened 
        averageBlurExtents = [] 
        curr_total = 0 
        for imageNum in range(len(blur_extents)): 
            for i, iteration in enumerate(allBlurExtents): 
                curr_total += iteration[imageNum] 
            averageBlurExtents.append( curr_total/iterations )
            curr_total = 0  


        
        x = np.arange(1, len(blur_extents)+1, step=1, dtype=np.uint32) 
        y_ticks = np.arange(0, 1, step=0.2, dtype=np.float32) 
        x_ticks = x  
        ax[2].bar(x, averageBlurExtents) 
        ax[2].set_xticks(x_ticks)  
        ax[2].set_yticks(y_ticks) 
        ax[2].set_title("Average Blur Extent Coefficient VS. Image Index Num") 
        ax[2].set_xlabel('Image Index') 
        ax[2].set_ylabel('Num') 


        # ax[0].plot(x,y) 
        # ax[0].set_title('Index of Best Image VS. Bilateral Blur Factor (d={})'.format(d)) 
        # ax[0].set_xlabel('Sigma Blur Factor') 
        # ax[0].set_ylabel('Image Index (starting at 1)') 

        # x = np.linspace(1, len(blur_extents), len(blur_extents), dtype=np.uint32) 
        # y = blur_extents
        # ax[1].plot(x,y) 
        # ax[1].set_title('Blur Coefficient VS. Image Index')
        # ax[1].set_xlabel('Image Index') 
        # ax[1].set_ylabel('Blur Coefficient')

        fig.savefig("{}/d={}_sigStart={}_sigIncrement={}.png".format(curr_save_directory, d, sigma_start, sigma_increment))
        # plt.show() 
        
        # Reset bestImageIndex 
        bestImageIndex = [] 
        averageBlurExtent = [] 


    print("WRRRYYYYYYYYYYYYYYYYYYY JOJO " ) 
    print("Reiner Braun: \nI've been here too long for my own good. Three years of this madness, \nsurrounded by idiots. We were kids! What did we know about anything? \nWhy did there have to be people like this? Why? Why did I let myself devolve into such a half-assed piece of shit?! \nIt's too late now... (begins removing Historia's sling from his injured arm) Damned if I know what's right anymore. \nWho cares, it is what it is. But the only choice left for me now. (holds up now steaming arm) \nAs a Warrior... no road left but the one that leads to the end! (it heals fully in a crackle of embers) \nBertholdt Hoover: \nReiner! Right now, here?! We're doing this?! \nReiner: \nYes. Right here, right now, we settle this once and for all!")



def findBestImage(folderDir, saveDir, method=SHARPNESS_EDGE_CHARACTERIZATION, frameName='00', startNum=0, imgType='.bmp', bilateral_filter=None, gaussian_filter=None, SVM_model='brisque_svm.txt'): 
    '''
    This method takes in one image dataset enclosed in a folder and processes it with a specified sharpness metric to 
    generate and save a figure displaying the image metrics through the dataset. 

    Args: 

        (param1): 
            (str): Dataset location. 
        (param2): 
            (str): Save location. 
        (param3): 
            (int): Sharpness metric to be used. Precursor is "SHARPNESS_".
        (param4): 
            (str): Beginning of dataset name.
        (param5): 
            (int): Dataset index system start number. 
        (param6): 
            (str): Image type as string. Examples: ".tif", ".bmp" etc... 
        (param7): 
            (int): Optional parameter, defaulted to None. If included, a bilateral filter 
                   will be applied to each image before wavelet transformation with the 
                   specified sigma color and sigma space parameters filled by this param. 
        (param8): 
            (int): Optional parameter, defaulted to None. If included, a gaussian blur 
                   will be applied to each image before wavelet transformation with the 
                   specified gaussian blur kernel size parameters filled by this param. 
    
    Returns: 

        (return1) 
            (int): 0 start index of best image. NOT TO BE CONFUSED WITH BEST IMAGE NUMBER (which starts on 1.) 
    '''
    
    # Assign reference to imagestack in order 
    imageStack = orderFrames(folderDir, frameName=frameName, startNum=startNum, imgType=imgType)

    if method == SHARPNESS_EDGE_CHARACTERIZATION: 
        blur_extents = [] 
        blur_ratios = [] 

        for i, image_dir in enumerate(imageStack): 
            img = cv2.imread(image_dir, 0) 
            img = cv2.bilateralFilter(img, 9, 120, 120) 
            curr_blur_params = getBlurExtent(img, 3, 10, None, None) 
            blur_extents.append( round(curr_blur_params[0], 4) )
            blur_ratios.append( curr_blur_params[1]) 

        curr_best = [0, 1] 

        for i in range(len(blur_extents)): 
            if blur_extents[i] < curr_best[1]: 
                curr_best[1] = blur_extents[i] 
                curr_best[0] = i + 1 

        # Display all images in a plot 
        img_num = len(imageStack) 

        subplot_width = 3 
        if img_num % 3 == 0: 
            subplot_height = img_num//3 
        else: 
            subplot_height = img_num//3 + 1

        fig, ax = plt.subplots(subplot_height,subplot_width) 
        fig.set_size_inches(int(10*subplot_width), int(10*subplot_height)) 
        counter = 1 
        row_num = 0 
        coloumn_num = 0 
        
        for i in range(img_num): 
            if counter == 4: 
                row_num += 1 
                counter = 1 
                coloumn_num = 0

            curr_image = cv2.imread(imageStack[i], 0) 
            ax[row_num,coloumn_num].imshow(curr_image, interpolation='nearest', cmap=plt.cm.gray) 

            if i+1 == curr_best[0]: 
                ax[row_num, coloumn_num].set_title("Image {}: Blur Extent = {}, BEST IMAGE".format(i+1, blur_extents[i]), fontsize=20 )
                ax[row_num, coloumn_num].set_ylabel("BEST IMAGE", fontsize='xx-large', color='red') 
                ax[row_num, coloumn_num].set_xlabel("Ratio of Blurred Structures: {}/{}".format(blur_ratios[i][0], blur_ratios[i][1]), fontsize='xx-large', color='blue') 
            else: 
                ax[row_num, coloumn_num].set_title("Image {}: Blur Extent = {}".format(i+1, blur_extents[i]), fontsize=20)
                ax[row_num, coloumn_num].set_xlabel("Ratio of Blurred Structures: {}/{}".format(blur_ratios[i][0], blur_ratios[i][1]), fontsize='xx-large', color='blue') 

            coloumn_num += 1 
            counter += 1 
    
    # Determining best image based on edge gradient sharpness metric 
    elif method == SHARPNESS_EDGE_GRADIENT: 
        
        gradient_coefficients = [] 
        curr_best_gradient = 0
        curr_best_index = 0 

        # Read all images in the image stack, calculate and store sharpness coefficient 
        for i, image_dir in enumerate(imageStack): 
            
            img = cv2.imread(image_dir, 0) 

            # If bilateral filter is to be applied... 
            if bilateral_filter != None: 
                img = cv2.bilateralFilter(img, 9, bilateral_filter, bilateral_filter) 

            # If Gaussian Blur is to be applied... 
            if gaussian_filter != None: 
                img = cv2.GaussianBlur(img, (gaussian_filter, gaussian_filter) , 0) 

            curr_gradient = getSharpnessMetricHaarGradient(img)
            gradient_coefficients.append( curr_gradient ) 

            # Check if best so far..
            if curr_gradient > curr_best_gradient: 
                curr_best_gradient = curr_gradient 
                curr_best_index = i 


        # Prepare subplot figure dimensions based on number of images 
        subplot_width = 3 
        if len(imageStack) % 3 == 0: 
            subplot_height = len(imageStack)//3 
        else: 
            subplot_height = len(imageStack)//3 + 1
        
        # Initialize subplot and relative counter params 
        fig, ax = plt.subplots(subplot_height, subplot_width) 
        fig.set_size_inches( int(10*subplot_width), int(10*subplot_height)) 
        counter = 1 
        row_num = 0 
        coloumn_num = 0 

        for i in range(len(imageStack)): 

            if counter == 4: 
                row_num += 1 
                counter = 1 
                coloumn_num = 0 
            
            curr_image = cv2.imread(imageStack[i], 0) 
            ax[row_num, coloumn_num].imshow(curr_image, interpolation='nearest', cmap=plt.cm.gray) 

            # Check if current image is best image w/highest local gradient coefficient 
            if i == curr_best_index: 
                ax[row_num, coloumn_num].set_title("Image {}: Sharpness Metric = {}, BEST IMAGE".format(i+1, round( gradient_coefficients[i], 3 )) , fontsize=20 )
                ax[row_num, coloumn_num].set_ylabel("BEST IMAGE", fontsize='xx-large', color='red') 

            else: 
                ax[row_num, coloumn_num].set_title("Image {}: Sharpness Metric = {}".format(i+1, round( gradient_coefficients[i], 3 )), fontsize=20)

            coloumn_num += 1 
            counter += 1 

    elif method == SHARPNESS_BRISQUE: 

        quality_coefficients = [] 
        best_image_quality = 100 
        curr_best_index = 0 

        # Save all calculated scores to get an average of the dataset 
        total_score = 0

        # Read all images in the image stack, calculate and store quality metric 
        for i, image_dir in enumerate(imageStack): 
            
            img = cv2.imread(image_dir, 0) 

            # Apply any specified filters 
            if gaussian_filter != None: 
                img = cv2.GaussianBlur(img, (gaussian_filter, gaussian_filter), 0) 

            if bilateral_filter != None: 
                img = cv2.bilateralFilter(img, 9, bilateral_filter, bilateral_filter) 

            curr_quality = getImageQualityBRISQUE(img, SVM_model=SVM_model) 

            quality_coefficients.append(curr_quality) 
            total_score += curr_quality

            # Check if best so far... 
            if curr_quality < best_image_quality: 
                best_image_quality = curr_quality
                curr_best_index = i 
        

        # Calculate average score of the dataset: 
        avg_score = total_score/(len(quality_coefficients)) 


        # Prepare subplot figure dimensions based on number of images 
        subplot_width = 3 
        if len(imageStack) % 3 == 0: 
            subplot_height = len(imageStack)//3 
        else: 
            subplot_height = len(imageStack)//3 + 1 

        # Initialize subplot and relative counter params 
        fig, ax = plt.subplots(subplot_height, subplot_width) 
        fig.set_size_inches( int(10*subplot_width), int(10*subplot_height)) 
        counter = 1 
        row_num = 0 
        coloumn_num = 0 

        for i in range(len(imageStack)): 

            if counter == 4: 
                row_num += 1 
                counter = 1 
                coloumn_num = 0 
            
            curr_image = cv2.imread(imageStack[i], 0) 
            ax[row_num, coloumn_num].imshow(curr_image, interpolation='nearest', cmap=plt.cm.gray) 

            # Check if current image is best image w/highest local gradient coefficient 
            if i == curr_best_index: 
                ax[row_num, coloumn_num].set_title("Image {}: Sharpness Metric = {}, BEST IMAGE".format(i+1, round( quality_coefficients[i], 3 )) , fontsize=20 )
                ax[row_num, coloumn_num].set_ylabel("BEST IMAGE", fontsize='xx-large', color='red') 

            else: 
                ax[row_num, coloumn_num].set_title("Image {}: Sharpness Metric = {}".format(i+1, round( quality_coefficients[i], 3 )), fontsize=20)

            coloumn_num += 1 
            counter += 1 





    saveName = folderDir.split("\\") 
    saveName = saveName[-1] 
    fig.savefig("{}/{}.png".format(saveDir, saveName)) 

    # Need to close fig or else will get memory error 
    plt.close(fig) 

    return curr_best_index, avg_score 

def findBestImageFFT(folderDir, c, threshold, saveDir, frameName='00', startNum=0, imgType='.bmp'): 
    saveName = folderDir.split("\\") 
    saveName = saveName[-1] 

    # Assign reference to imagestack in order 
    imageStack = orderFrames(folderDir, frameName=frameName, startNum=startNum, imgType=imgType)

    edges_FFT = [] 

    # Set subplot parameters to display image and corresponding fourier transform 
    # Only display want to display best image, and 2 before and 2 after in image stack sequence 
    # Total 10 images 
    fig, ax = plt.subplots(2, 5) 
    fig.set_size_inches(50, 20)  
    fig.suptitle("saveName, c = {}, threshold = {}".format(c, threshold), fontsize=16) 

    for i in range(len(imageStack)): 
        curr_img = cv2.imread(imageStack[i], 0) 
        power_spectrum, curr_points = DFTEstimate(curr_img, c, threshold) 
        edges_FFT.append( len(curr_points) ) 

    '''
    Ignore this block: 
        if i < 5: 
            ax[0, i].imshow(curr_img, interpolation='nearest', cmap=plt.cm.gray) 
            ax[1,i].imshow(power_spectrum, interpolation='nearest', cmap=plt.cm.gray)
            ax[0, i].set_title("Image {}: Edge Count: {}".format(i+1, edges_FFT[i])) 
        else: 
            ax[0, i-5].imshow(curr_img, interpolation='nearest', cmap=plt.cm.gray) 
            ax[1, i-5].imshow(power_spectrum, interpolation='nearest', cmap=plt.cm.gray) 
            ax[0, i-5].set_title("Image {}: Edge Count: {}".format(i+1, edges_FFT[i])) 
    ''' 

    # Set current_highest with format [image number (starting at 1), number of edges] 
    curr_highest = [0, 0] 
    for i in range(len(edges_FFT)): 
        if edges_FFT[i] > curr_highest[1]: 
            curr_highest[0] = i+1
            curr_highest[1] = edges_FFT[i] 

    # Label best image on figure 
    best_img = cv2.imread(imageStack[curr_highest[0]-1], 0)
    best_img_power, some_points = DFTEstimate(best_img, c, threshold)  #some_points not used 
    ax[0, 2].imshow(best_img, cmap=plt.cm.gray) 
    ax[1, 2].imshow(best_img_power, cmap=plt.cm.gray) 
    ax[0, 2].set_title("Image {}: Edge Count = {}".format(curr_highest[0], curr_highest[1]))


    try: 
        displayImages = [cv2.imread(imageStack[curr_highest[0]-2], 0), cv2.imread(imageStack[curr_highest[0]-3], 0), \
                         cv2.imread(imageStack[curr_highest[0]], 0), cv2.imread(imageStack[curr_highest[0]+1], 0)]
        st = False 
    except IndexError: 
        displayImages = [cv2.imread(imageStack[curr_highest[0]-2], 0), cv2.imread(imageStack[curr_highest[0]-3], 0), \
                         cv2.imread(imageStack[curr_highest[0]-4], 0), cv2.imread(imageStack[curr_highest[0]-5], 0)]
        # Error status 
        st = True 
        

    displayPower = [DFTEstimate(displayImages[0], c, threshold), DFTEstimate(displayImages[1], c, threshold), \
                    DFTEstimate(displayImages[2], c, threshold), DFTEstimate(displayImages[3], c, threshold) ]

    displayPower = [displayPower[0][0], displayPower[1][0], displayPower[2][0], displayPower[3][0] ] 
    ax[0,0].imshow(displayImages[0], cmap=plt.cm.gray) 
    ax[0,0].set_title("Image {}: Edge Count = {}".format(curr_highest[0]-2, edges_FFT[curr_highest[0]-3]))

    ax[0,1].imshow(displayImages[1], cmap=plt.cm.gray) 
    ax[0,1].set_title("Image {}: Edge Count = {}".format(curr_highest[0]-1, edges_FFT[curr_highest[0]-2]))

    ax[0,3].imshow(displayImages[2], cmap=plt.cm.gray) 
    if st == True: 
        ax[0,3].set_title("Image {}: Edge Count = {}".format(curr_highest[0]-3, edges_FFT[curr_highest[0]-4]))
    else: 
        ax[0,3].set_title("Image {}: Edge Count = {}".format(curr_highest[0]+1, edges_FFT[curr_highest[0]]))


    ax[0,4].imshow(displayImages[3], cmap=plt.cm.gray) 
    if st == True: 
        ax[0,4].set_title("Image {}: Edge Count = {}".format(curr_highest[0]-4, edges_FFT[curr_highest[0]-5]))
    else: 
        ax[0,4].set_title("Image {}: Edge Count = {}".format(curr_highest[0]+2, edges_FFT[curr_highest[0]+1]))

    ax[1,0].imshow(displayPower[0], cmap=plt.cm.gray) 
    ax[1,1].imshow(displayPower[1], cmap=plt.cm.gray)
    ax[1,3].imshow(displayPower[2], cmap=plt.cm.gray)
    ax[1,4].imshow(displayPower[3], cmap=plt.cm.gray)

    saveName = folderDir.split("\\") 
    saveName = saveName[-1] 
    fig.savefig("{}/{}.png".format(saveDir, saveName)) 


# ------------------------------------- Auto Stigmation Tools ------------------------ # 
def getStigData(src_under, src_over, FFT_c, FFT_threshold, resize=None, drawMask=False): 
    '''
    Implementation of article: K.H. Ong, J.C.H. Phang, J.T.L. Thong, “A Robust Focusing and Astigmatism Correction 
    Method for the Scanning Electron Microscope”, Centre for Integrated Circuit Failure Analysis asnd Reliability (CICFAR). 

    Args:

        (param1): 
            (np.ndarray): Under focused image. 
        (param2): 
            (np.ndarray): Over focused image. 
        (param3): 
            (int): Fast Fourier Transform scaling factor. 
        (param3): 
            (int): Fast Fourier Transform threshold value. 
        (param4): 
            (float): If not none, resize to input factor. 
        (param5): 
            (bool): Whether or not to draw a mask. 

    '''

    if resize != None: 
        src_under = resizeImage(src_under, resize) 
        src_over = resizeImage(src_over, resize) 

    # Take FFT of under focused and over focused images 
    under_FFT, under_points = DFTEstimate(src_under, FFT_c, FFT_threshold) 
    over_FFT, over_points = DFTEstimate(src_over, FFT_c, FFT_threshold) 

    # Get r1 r2 r3 r4 s1 s2 s3 s4 data 
    under_data = partitionImage8(under_FFT, under_points) 
    over_data = partitionImage8(over_FFT, over_points) 

    # Store stig params 
    under_r1 = under_data[0] 
    under_r2 = under_data[1] 
    under_r3 = under_data[2] 
    under_r4 = under_data[3] 

    under_s1 = under_data[4] 
    under_s2 = under_data[5] 
    under_s3 = under_data[6] 
    under_s4 = under_data[7] 

    over_r1 = over_data[0] 
    over_r2 = over_data[1] 
    over_r3 = over_data[2] 
    over_r4 = over_data[3] 

    over_s1 = over_data[4] 
    over_s2 = over_data[5] 
    over_s3 = over_data[6] 
    over_s4 = over_data[7] 

    # Initialize algorithm parameters: 
    P_of = len(over_points) 
    P_uf = len(under_points) 

    P_r12_of = ( ( len(over_r1) + len(over_r2) )/P_of )*100 
    P_r12_uf = ( ( len(under_r1) + len(under_r2) )/P_uf )*100

    P_r34_of = ( ( len(over_r3) + len(over_r4)/P_of )/P_of )*100 
    P_r34_uf = ( ( len(under_r3) + len(under_r4) )/P_uf )*100

    P_s12_of = ( ( len(over_s1) + len(over_s2) )/P_of )*100 
    P_s12_uf = ( ( len(under_s1) + len(under_s2) )/P_uf )*100

    P_s34_of = ( ( len(over_s3) + len(over_s4)/P_of )/P_of )*100 
    P_s34_uf = ( ( len(under_s3) + len(under_s4) )/P_uf )*100

    '''
    Calculate "R" = Image Sharpness Metric: 

    1. When R > 0, overfocused image is sharper than underfocused 
    image, which implies that the focal length should be decreased. 

    2. When R < 0, underfocused image is sharper than the overfocused 
    image, which implies that the focal length should be increased. 

    '''

    R = (P_of - P_uf)/(P_of + P_uf) 
    R_abs = np.abs(R) 


    '''
    Calculate delta_r's and s's = Astigmatism metrics 

    1. When delta_r12 > 0, delta_r34 < 0, 
    there is astigmatism along A-B axis, stigma x should be decreased. 

    2. When delta_r12 < 0 and delta_r34 > 0, 
    there is astigmatism along A-B axis, stigma x should be increased. 

    3. When delta_s12 > 0 and delta_s34 < 0, 
    there is astigmatism along C-D axes and stigma y should be increased. 

    4. When delta_s12 < 0 and delta_s34 > 0, 
    there is astigmatism along C-D axes, and stigma y should be decreased. 

    '''

    delta_r12 = P_r12_of - P_r12_uf 
    delta_r34 = P_r34_of - P_r34_uf 
    delta_s12 = P_s12_of - P_s12_uf 
    delta_s34 = P_s34_of - P_s34_uf 

    ''''
    Since there is 90 degree rotation stretch as image goes under to over focussed, 
    corresponding FFTs also experience 90 degree rotation. This means: 

    1. Regions R1 R2 of overfocused have similar characteristics to R3 R4 of underfocused. 

    2. Regions R3 R4 of overfocused have similar characteristics to R1 R2 of underfocused. 

    3. Regions S1 S2 of overfocused have similar characteristics to S3 S4 of underfocused. 

    4. Regions S3 S4 of overfocused have similar characteristics to S1 S2 of underfocused.  

    Astigmatism along A-B axes and C-D axes quantified by variable A_x and A_y where: 
    '''

    # 0.5 factor for normalization so that values between 0 and 100%. 
    A_x = np.abs( (delta_r12 - delta_r34)/2 )
    A_y = np.abs( (delta_s12 - delta_s34)/2 )

    # A_x and A_y only indicate the more sever astigmatism to correct 

    return [A_x, A_y], [delta_r12, delta_r34, delta_s12, delta_s34]

def thresholdFFT(src_FFT, threshold): 
    '''
    Non-Static method takes in a FFT normalized power spectrum image 
    and thresholds using a given threshold. Used specifically in 
    adaptiveFFTThreshold(self).  
    '''

    mask = np.zeros_like(src_FFT, dtype=np.uint8) 
    points = [] 

    height, width = src_FFT.shape 

    for i in range(height): 
        for j in range(width): 
            if src_FFT[i,j] >= threshold: 
                mask[i,j] = src_FFT[i,j] 
                points.append( [i,j])
    
    return mask, points

class Autostig: 
    '''
    Class implements auto focus and auto stig algorithm described in 
    A Robust Focusing and Astigmatism Correction Method for the Scanning
    Electron Microscope—Part III: An Improved Technique
    K.H. ONG, J.C.H. PHANG, J.T.L. THONG
    '''

    def __init__(self, under_focused, over_focused): 
        '''
        Default constructor of class Autostig. Only need to initialize 
        2 images as static variables as the rest of the autostig 
        parameters will be calculated and assigned as static variables
        in the accompanying static methods. 
        '''
        # Initialize TEMPORARY electron microscope variables --------------------- #
        self.focus = 1000 
        self.stig_x = 1000 
        self.stig_y = 1000 
        self.mag = 3000 

        # Assert we are only dealing with grayscale images. 
        assert len(under_focused.shape) == 2 
        assert len(over_focused.shape) == 2 

        self.under = under_focused 
        self.over = over_focused 
        self.optimal_T = 90 
        self.current_T = 90 
        self.max_delta = 0 
        self.current_delta = 0 
        self.lower_delta_count = 0 
        self.optimal_T_found = False 

        # Initialize AFAS checkStig variables: 
        self.all_R = []
        self.all_A_x = [] 
        self.all_A_y = [] 
        self.is_focused_is_stigmated = False 
        self.stigCorrectionRuns = 0 

        # Initialize focus/stig offset and step parameters 
        self.focus_offset = (10000/self.mag)*50 
        self.min_focus_res = (10000/self.mag)*5 
        self.min_stigma_res = self.min_focus_res

        self.optimal_focus = 0
        self.optimal_stig_x = 0
        self.optimal_stig_y = 0 
        # ------------------------------------------------------------------------ # 
    
    # def __init__(self, under_focused, over_focused, focus, magnification): 
    #     '''
    #     Constructur if magnification given from SEM. 
    #     '''
        
    #     # Assert we are only dealing with grayscale images. 
    #     assert len(under_focused.shape) == 2 
    #     assert len(over_focused.shape) == 2 

    #     self.under = under_focused 
    #     self.over = over_focused 
    #     self.mag = magnification 
    #     self.focus = focus 

    def calcFFT(self, FFT_c, FFT_threshold): 
        '''
        Static function to be used in by Autostig object. Takes underfocused 
        and overfocused image to calculate estimated discrete fourier transform 
        using the fast fourier transform method for power spectrum analysis. 

        Args: 
            (param1): 
                (self): Self parameter to get class variables. 
            (param2): 
                (int/float): Fourier estimate scaling factor from formula...
                             new_I = c*log(1 + FFT_I) 
            (param3): 
                (int): Threshold value to filter estimated discrete fourier 
                       power spectrum. 
        Returns: 
            (return1): 
                (np.ndarray): Fourier transform of under-focused image. 
            (return2): 
                (np.ndarray): Thresholded points of under-focused power spectrum. 
            (return3): 
                (np.ndarray): Fourier transform of over-focused image. 
            (return4): 
                (np.ndarray): Thresholded points of over-focused power spectrum. 
        '''
        under_FFT, under_points = DFTEstimate(self.under, FFT_c, FFT_threshold) 
        over_FFT, over_points = DFTEstimate(self.over, FFT_c, FFT_threshold) 

        # Assign calculated Autostig params as object static variables: 
        self.under_FFT = under_FFT
        self.over_FFT = over_FFT 
        self.under_points = under_points 
        self.over_points = over_points 

        return under_FFT, under_points, over_FFT, over_points 


    def setStigParams(self, FFT_c, FFT_threshold, under_FFT=None, under_points=None, over_FFT=None, over_points=None): 
        '''
        General purpose function used to determine stig params for FFT frequency
        spectrum analysis. Can calculate FFT 
        '''

        # First take fourier transform of under and over focused images. 
        if type(under_FFT) == type(self.under): 

            under_data = partitionImage8(under_FFT, under_points) 
            over_data = partitionImage8(over_FFT, over_points) 

        else:

            self.calcFFT(FFT_c, FFT_threshold) 
            under_data = partitionImage8(self.under_FFT, self.under_points) 
            over_data = partitionImage8(self.over_FFT, self.over_points) 

        # Store data in easy to remember names 
        self.under_r1 = under_data[0] 
        self.under_r2 = under_data[1] 
        self.under_r3 = under_data[2] 
        self.under_r4 = under_data[3] 

        self.under_s1 = under_data[4] 
        self.under_s2 = under_data[5] 
        self.under_s3 = under_data[6] 
        self.under_s4 = under_data[7] 

        self.over_r1 = over_data[0] 
        self.over_r2 = over_data[1] 
        self.over_r3 = over_data[2] 
        self.over_r4 = over_data[3] 

        self.over_s1 = over_data[4] 
        self.over_s2 = over_data[5] 
        self.over_s3 = over_data[6] 
        self.over_s4 = over_data[7]

        if type(under_FFT) == type(self.under): 
            self.P_of = len(over_points) 
            self.P_uf = len(under_points) 
        else: 
            
            # Initialize and set algorithm parameters to static class variables. 
            self.P_of = len(self.over_points) 
            self.P_uf = len(self.under_points) 
 


        self.P_r12_of = ( ( len(self.over_r1) + len(self.over_r2) )/self.P_of )*100 
        self.P_r12_uf = ( ( len(self.under_r1) + len(self.under_r2) )/self.P_uf )*100

        self.P_r34_of = ( ( len(self.over_r3) + len(self.over_r4)/self.P_of )/self.P_of )*100 
        self.P_r34_uf = ( ( len(self.under_r3) + len(self.under_r4) )/self.P_uf )*100

        self.P_s12_of = ( ( len(self.over_s1) + len(self.over_s2) )/self.P_of )*100 
        self.P_s12_uf = ( ( len(self.under_s1) + len(self.under_s2) )/self.P_uf )*100

        self.P_s34_of = ( ( len(self.over_s3) + len(self.over_s4)/self.P_of )/self.P_of )*100 
        self.P_s34_uf = ( ( len(self.under_s3) + len(self.under_s4) )/self.P_uf )*100

        '''
        Calculate "R" = Image Sharpness Metric: 

        1. When R > 0, overfocused image is sharper than underfocused 
        image, which implies that the focal length should be decreased. 

        2. When R < 0, underfocused image is sharper than the overfocused 
        image, which implies that the focal length should be increased. 

        '''

        self.R = (self.P_of - self.P_uf)/(self.P_of + self.P_uf) 
        R_abs = np.abs(self.R) 

        '''
        Calculate delta_r's and s's = Astigmatism metrics 

        1. When delta_r12 > 0, delta_r34 < 0, 
        there is astigmatism along A-B axis, stigma x should be decreased. 

        2. When delta_r12 < 0 and delta_r34 > 0, 
        there is astigmatism along A-B axis, stigma x should be increased. 

        3. When delta_s12 > 0 and delta_s34 < 0, 
        there is astigmatism along C-D axes and stigma y should be increased. 

        4. When delta_s12 < 0 and delta_s34 > 0, 
        there is astigmatism along C-D axes, and stigma y should be decreased. 

        '''

        self.delta_r12 = self.P_r12_of - self.P_r12_uf 
        self.delta_r34 = self.P_r34_of - self.P_r34_uf 
        self.delta_s12 = self.P_s12_of - self.P_s12_uf 
        self.delta_s34 = self.P_s34_of - self.P_s34_uf 

        ''''
        Since there is 90 degree rotation stretch as image goes under to over focussed, 
        corresponding FFTs also experience 90 degree rotation. This means: 

        1. Regions R1 R2 of overfocused have similar characteristics to R3 R4 of underfocused. 

        2. Regions R3 R4 of overfocused have similar characteristics to R1 R2 of underfocused. 

        3. Regions S1 S2 of overfocused have similar characteristics to S3 S4 of underfocused. 

        4. Regions S3 S4 of overfocused have similar characteristics to S1 S2 of underfocused.  

        Astigmatism along A-B axes and C-D axes quantified by variable A_x and A_y where: 
        '''

        # 0.5 factor for normalization so that values between 0 and 100%. 
        # self.A_x = np.abs( (self.delta_r12 - self.delta_r34)/2 )
        # self.A_y = np.abs( (self.delta_s12 - self.delta_s34)/2 )

        if (self.delta_r12*self.delta_r34) < 0: 
            self.A_x = (self.delta_r12 - self.delta_r34)/2 
        else: 
            self.A_x = 0 
        if (self.delta_s12*self.delta_s34) < 0: 
            self.A_y = (self.delta_s12 - self.delta_s34)/2 
        else: 
            self.A_y = 0 


        # A_x and A_y only indicate the more sever astigmatism to correct
    
    def printComparison(self, saveDir=None, saveName='Default', show=True): 
        '''
        This function is only to be used AFTER self.setStigParams() has been used. 
        Otherwise, errors will be thrown that static variables being referenced to 
        D.N.E. 
        Args: 
            (param1): 
                (str): *Optional, save directory to save final comparison figure. 
        '''
        height, width = self.under.shape 
        mask_under = np.zeros( (height, width, 3), dtype=np.uint8 ) 
        mask_over = np.zeros( (height, width, 3), dtype=np.uint8 ) 

        # Draw under focused mask 
        for point in self.under_r1: 
            mask_under[point[0], point[1]] = (0, 0, 255) 
        for point in self.under_r2: 
            mask_under[point[0], point[1]] = (0, 255, 0) 
        for point in self.under_r3: 
            mask_under[point[0], point[1]] = (255, 0, 0) 
        for point in self.under_r4: 
            mask_under[point[0], point[1]] = (0, 0, 0) 

        for point in self.under_s1: 
            mask_under[point[0], point[1]] = (0, 255, 255) 
        for point in self.under_s2: 
            mask_under[point[0], point[1]] = (255, 0, 255) 
        for point in self.under_s3: 
            mask_under[point[0], point[1]] = (255, 255, 0) 
        for point in self.under_s4: 
            mask_under[point[0], point[1]] = (214, 24, 173) 

        # Draw over focused mask
        for point in self.over_r1: 
            mask_over[point[0], point[1]] = (0, 0, 255) 
        for point in self.over_r2: 
            mask_over[point[0], point[1]] = (0, 255, 0) 
        for point in self.over_r3: 
            mask_over[point[0], point[1]] = (255, 0, 0) 
        for point in self.over_r4: 
            mask_over[point[0], point[1]] = (0, 0, 0) 

        for point in self.over_s1: 
            mask_over[point[0], point[1]] = (0, 255, 255)
        for point in self.over_s2: 
            mask_over[point[0], point[1]] = (255, 0, 255) 
        for point in self.over_s3: 
            mask_over[point[0], point[1]] = (255, 255, 0)
        for point in self.over_s4: 
            mask_over[point[0], point[1]] = (214, 24, 173) 

        # convert masks from BGR to RGB 
        mask_under = cv2.cvtColor(mask_under, cv2.COLOR_BGR2RGB) 
        mask_over = cv2.cvtColor(mask_over, cv2.COLOR_BGR2RGB) 

        # Draw subplots for imaging 
        fig, ax = plt.subplots(2, 3) 
        fig.set_size_inches(30, 20) 
        fig.suptitle("Under-Focused and Over-Focused FFT 8 Piece Segmentation", fontsize=40)

        ax[0, 0].imshow(self.under, cmap=plt.cm.gray) 
        ax[0, 0].set_title("Under Focused", fontsize=20) 
        ax[0, 1].imshow(self.under_FFT, cmap=plt.cm.gray) 
        ax[0, 1].set_title("Under Focused FFT", fontsize=20) 
        ax[0, 2].imshow(mask_under, interpolation='nearest') 
        ax[0, 2].set_title("Under Focused FFT Segmented", fontsize=20)

        ax[1, 0].imshow(self.over, cmap=plt.cm.gray) 
        ax[1, 0].set_title("Over Focused", fontsize=20)
        ax[1, 1].imshow(self.over_FFT, cmap=plt.cm.gray) 
        ax[1, 1].set_title("Over Focused FFT", fontsize=20) 
        ax[1, 2].imshow(mask_over, interpolation='nearest') 
        ax[1, 2].set_title("Over Focused FFT Segmented", fontsize=20)

        if show == True:  
            plt.show() 
        
        if saveDir != None: 
            # create new folder to save diagnostic data 
            saveFolder = createFolder(saveDir, saveName) 
            cv2.imwrite("{}\\{}.bmp".format(saveFolder, "under focused"), self.under) 
            cv2.imwrite("{}\\{}.bmp".format(saveFolder, "under focused FFT"), self.under_FFT) 

            cv2.imwrite("{}\\{}.bmp".format(saveFolder, "over focused"), self.over) 
            cv2.imwrite("{}\\{}.bmp".format(saveFolder, "over focused FFT"), self.over_FFT) 

            cv2.imwrite("{}\\{}.png".format(saveFolder, "under focused segmented"), mask_under) 
            cv2.imwrite("{}\\{}.png".format(saveFolder, "over focused segmented"), mask_over) 
            
            fig.savefig("{}\\{}.png".format(saveFolder, 'comparison'))

    def adaptiveFFTThreshold(self):
        '''
        Function is to be used after default constructor giving underfocused and 
        overfocused image. These two images will be used to compute the optimal 
        estimated fast fourier transform threshold value to be used in focus and 
        stigmatism correction. 

        Args: 

            (param1): 
                (self): No 'real' parameters. Static method of class Autostig. 
        
        Returns: 

            (return1): 
                (bool): Returns boolean of whether the best T has been found so far. 
                        Should always return True, otherwise something is wrong... 
        ''' 

        # Stop function if optimal T is somehow past 170 
        if self.current_T > 169:
            print("\nWarning, self.optimal_T is very high...\n") 
            



        '''
        Only uncomment when implemented in live system. 
        '''

        # focus_offset = (10000/self.mag)*50 

        # focus_under = focus - focus_offset 
        # focus_over = focus + focus_offset 

        '''
        End focus_offset adjustment 
        '''

        # Calculate FFT's of over and under images, store as local method variables 
        # NOTE: NEED TO SET THRESHOLD AT 0 HERE. WE ALL INFORMATION FOR NOW 
        under_FFT = DFTEstimate(self.under, OPTIMAL_C)
        over_FFT = DFTEstimate(self.over, OPTIMAL_C) 
        

        # Set prerequisites 
        self.current_T = 90 
        self.optimal_T = 90 
        self.max_delta = 0 

        # Begin filtering loop 
        while(self.current_T < 255):
            
            # Apply current thresholds to the original FFT images saved as non-static variables 
            under_FFT_thresholded, under_points = thresholdFFT(under_FFT, self.current_T) 
            over_FFT_thresholded, over_points = thresholdFFT(over_FFT,  self.current_T) 
 
            print("Current T = {}, Current Delta = {}, Avg. Point Count = {}.".format(self.current_T, self.current_delta, (len(under_points)+len(over_points)/2) ))

            # printImage('', under_FFT_thresholded) 
            # printImage('', over_FFT_thresholded)

            # Apply stig parameters for current threshold value 
            self.setStigParams(OPTIMAL_C, self.current_T, under_FFT_thresholded,  under_points, over_FFT_thresholded, over_points) 

            # Check if too few points 
            if len(under_points) < 400 or len(over_points) < 400: 
                self.current_T = 256 
                # Set current T to break the while loop at the end of this iteration 

                # Print diagnostic message
                print("OPTIMAL DELTA FOUND...\n\nOptimal FFT Threshold = {}, Best Delta = {}".format(self.optimal_T, self.max_delta)) 

            # If there are still more than 400 points in the thresholded FFT 
            else: 
                
                self.current_delta = (  np.abs(self.A_x) + np.abs(self.A_y) + 10*np.abs(self.R)  ) / 3 

                if self.current_delta > self.max_delta: 

                    self.max_delta = self.current_delta
                    self.optimal_T = self.current_T  
                
                self.current_T += 2 







        # -------------------------------- Implementation Changed July 19th 2019 ------------------------- # 
        # self.setStigParams(OPTIMAL_C, self.current_T) 

        # # Check if there are too few points for computation. 
        # # If there are less than 100, the optimal T has been found. 
        # if ( (self.P_of + self.P_uf)/2 ) < 400 or self.current_T > 120:  
        #     self.optimal_T_found = True 
        #     return self.optimal_T_found  
        
        # elif self.optimal_T_found == False: 
        
        #     # Compute parameter DELTA here: 
        #     '''
        #     R is multiplied by 10 to normalize R to same range of A_x, and A_y. 
        #     Typically with no astigmatism, A_x and A_y should be less than 1, and 
        #     R should be less than 0.1. Parameter DELTA serves as measure of defocus
        #     and astigmatism information that can be extracted from the thresholded
        #     FFT's. Therefore, algorithm iterates and keeps track of maximum DELTA 
        #     that has been found. However, when DELTA goes too high so that there 
        #     are too few points in the FFT's to grab valuable data, the algorithm 
        #     is stopped, and the most recent optimal_threshold with corresponding 
        #     largest DELTA will be chosen as the optimal threshold for FFT's. 
        #     '''
        #     self.current_delta = ( np.abs(self.A_x) + np.abs(self.A_y) + 10*np.abs(self.R) )/3 

        #     if self.current_delta > self.max_delta: 
        #         self.max_delta = self.current_delta  
        #         self.optimal_T = self.current_T 
        #         self.current_T += 2 

        #     else: 
        #         self.current_T += 2
        #         self.lower_delta_count += 1  

        #     return self.optimal_T_found 

        # ---------------------------------------------------------------------------------------------- # 


    def adjustFocus(self):
        '''
        This function is to be used in self.checkStig(). It adjusts
        the current focus value found in self.focus using the SEM
        magnification from self.mag to alculate the new focus value. 
        '''
        self.min_focus_res = (10000/self.mag)*5
        self.min_focus_res = (10000/self.mag)*5 
        self.focus = self.focus + (10*self.R*self.min_focus_res) 
        return None 

    def adjustStigmaX(self): 
        '''
        This function is to be used in self.checkStig(). It adjusts
        the current stig_x value found in self.stig_x using the SEM
        magnification from self.mag to alculate the new focus value. 
        '''
        self.min_stigma_res = (10000/self.mag)*5 
        self.stig_x = self.stig_x + (self.A_x*self.min_stigma_res) 
        return None 
    
    def adjustStigmaY(self): 
        '''
        This function is to be used in self.checkStig(). It adjusts
        the current stig_y value found in self.stig_y using the SEM
        magnification from self.mag to alculate the new focus value. 
        '''
        self.min_stigma_res = (10000/self.mag)*5 
        self.stig_y = self.stig_y + (self.A_y*self.min_stigma_res) 
        return None 


    def checkStig(self): 
        '''
        This function takes in an object of class Autostig.py with only underfocused and 
        overfocused images assigned as static variables so far. It will compute whether 
        or not the image is stigmated, and will return instruction for the electron 
        microscope to increase/decrease stig x and stig y. 

        This method assumes that static method adaptiveFFTThreshold has been run prior
        to running this method to assess the stigmation. 

        It also works closely in tandem with static methods adjustStigmaX() and adjustStigmaY(). 

        Args:

            (param1): 
                (self): (self): No 'real' parameters. Static method of class Autostig. 
        
        Returns: 

            (return1): 
                (bool): Conditional representing if the current sample is correctly 
                        focused and stigmated. If True, the sample needs no more 
                        correction. If false, self.checkStig() needs to be run 
                        for further iterations. 
        '''

        # Set stig parameters with optimal threshold value for the dataset. 
        self.setStigParams(OPTIMAL_C, self.optimal_T) 

        # Only append R, A_x and A_y values when run from this method. 
        self.all_R.append(self.R) 
        self.all_A_x.append(self.A_x) 
        self.all_A_y.append(self.A_y) 
        
        # Run this indented block when image focus is not good enough (not enough data). 
        if np.abs(self.R) < 0.08: 
            corrected_A_x = np.abs(self.A_x) < 1 
            corrected_A_y = np.abs(self.A_y) < 1 
        
            if corrected_A_x == False:
                adjustStigmaX() 

            elif corrected_A_y == False: 
                adjustStigmaY() 

        else: 
            adjustFocus() 
        
        if len(self.avg_R >= 3): 
            avg_R = ( self.all_R[-1] + self.all_R[-2] + self.all_R[-3] ) / 3 
            avg_A_x = ( self.all_A_x[-1] + self.all_A_x[-2] + self.all_A_x[-3] ) / 3 
            avg_A_y = ( self.all_A_y[-1] + self.all_A_y[-2] + self.all_A_y[-3] ) / 3 
        
        elif len(self.avg_R == 2): 
            avg_R = ( self.all_R[-1] + self.all_R[-2] ) / 2 
            avg_A_x = ( self.all_A_x[-1] + self.all_A_x[-2] ) / 2
            avg_A_y = ( self.all_A_y[-1] + self.all_A_y[-2] ) / 2 
        
        elif len(self.avg_R == 1): 
            avg_R = self.R 
            avg_A_x = self.A_x
            avg_A_y = self.A_y

        # NOW check conditionals to see if image is correctly focussed and stigmated 
        if (np.abs(avg_R) < 0.08) and (np.abs(avg_A_x) < 1) and (np.abs(avg_A_y) < 1): 

            self.optimal_focus = self.focus 
            self.optimal_stig_x = self.stig_x
            self.optimal_stig_y = self.stig_y 

            self.is_focused_is_stigmated = True 


        # This else block may cause problems later...but is for peace of mind for now 
        else: 
            self.is_focused_is_stigmated = False 

        self.stigCorrectionRuns += 1 
        # This return statment runs no matter what conditionals are satisfied or not. 
        return self.is_focused_is_stigmated


# ------------------------------ Spatial Spectrum Image Quality References -------------------- # 
def getGlobalVariance(src): 
    m = cv2.meanStdDev(src) 

    print(m) 

    return m 














# ------------------------------------------------------------------------------------ # 


# ----------------------------------- ELLIPSE FITTING -------------------------------- # 
def findEllipses(src): 

    contours, hierarchy = cv2.findContours(src.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    ellipseMask = np.zeros_like(src, dtype=np.uint8) 
    contourMask = np.zeros_like(src, dtype=np.uint8) 

    pi_4 = np.pi*4 

    largest_contour_count = 0
    largest_contour_index = 0  

    for i, contour in enumerate(contours): 

        if len(contour) > largest_contour_count: 
            largest_contour_count = len(contour) 
            largest_contour_index = i
        
        # If contour has less than 5 points, ignore 
        if len(contour) < 5: 
            continue 

        area = cv2.contourArea(contour) 
        # Skip ellipses smaller than 10x10 
        if area <= 100 or area >= 1500: 
            continue 

        arclen = cv2.arcLength(contour, True) 
        circularity = (pi_4*area) / (arclen * arclen) 
        ellipse = cv2.fitEllipse(contour) 
        poly = cv2.ellipse2Poly( (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), 
                                int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        
        # If contour is circular enough...
        if circularity > 0.01: 
            cv2.fillPoly(ellipseMask, [poly], 255) 
            continue 
        # else: 
        #     cv2.fillPoly(ellipseMask, [poly], 255) 
        
        # If contour has enough similarity to an ellipse 
        similarity = cv2.matchShapes(poly.reshape(( poly.shape[0], 1, poly.shape[1])), contour, 
                                    1, 0) 
        if similarity <= 0.2: 
            cv2.fillPoly(contourMask, [poly], 255) 
        # else: 
        #     cv2.fillPoly(contourMask, [poly], 255) 

    # print("Largest Contour Index: {}, with Contour Count of: {}".format(largest_contour_index, largest_contour_count)) 

    return ellipseMask, contourMask 

def getNoiseMetricFFT(src, ellipseMask):
    assert len(ellipseMask.shape) == 2
    assert len(src.shape) == 2 

    mask = src.copy() 

    height, width = ellipseMask.shape 

    for i in range(height): 
        for j in range(width): 
            if ellipseMask[i,j] == 255: 
                mask[i,j] = 0 
    
    noise_count = 0 
    for i in range(height): 
        for j in range(width): 
            if mask[i,j] == 255: 
                noise_count += 1 
    
    return noise_count, mask    

def getNoiseMetricFFTFolder(folderDir=None, saveDir=None, FFT_threshold=80): 

    images = [] 
    images_FFT = [] 
    images_ellipse = [] 
    images_mask = [] 
    images_noisecount = [] 

    if folderDir == None: 
        folderDir = loadFolderDir("Location of Folder to Load Datasets") 
    
    if saveDir == None: 
        saveDir = loadFolderDir("Location of save directory")  

    folders = listFolders(folderDir) 

    for folder in folders: 

        # Within each nested folder of images.... 
        images = [] 
        images_FFT = [] 
        images_ellipse = [] 
        images_mask = [] 
        images_noisecount = [] 

        imagesDir = orderFrames(folder, '00', '.bmp', 0) 

        for imageDir in imagesDir: 
            img = cv2.imread(imageDir, 0) 
            img_FFT, points = DFTEstimate(img, 10, FFT_threshold)  

            ellipseMask, contourMask = findEllipses(img_FFT) 
            noise_count, mask = getNoiseMetricFFT(img_FFT, ellipseMask) 

            images.append(img) 
            images_FFT.append(img_FFT) 
            images_ellipse.append(ellipseMask) 
            images_mask.append(mask) 
            images_noisecount.append(noise_count) 
        
        # Find noisiest image: 
        curr_worst_index = 0 
        curr_best_index = 0 
        curr_highest_noise = 0
        curr_lowest_noise = 1000000  
        total_counts = 0 
        for i in range(len(imagesDir)): 
            if curr_highest_noise < images_noisecount[i]: 
                curr_worst_index = i 
                curr_highest_noise = images_noisecount[i] 

            if curr_lowest_noise > images_noisecount[i]: 
                curr_best_index = i 
                curr_lowest_noise = images_noisecount[i] 
            
            total_counts += images_noisecount[i] 
        avg_noise = round(total_counts/len(images_noisecount)) 


        # Computations done, create folder and save data plots..
        saveName = folder.split('\\') 
        saveName = saveName[-1] 

        curr_folder = createFolder(saveDir, saveName) 

        fig, ax = plt.subplots(4, len(imagesDir)) 
        fig.suptitle("Ellipse Fitting FFT Thresholding for Noise Metric", fontsize=25)   
        fig.set_size_inches(int(2*len(imagesDir)), 8) 

        # Set image labels
        ax[0, 0].set_title("Initial Images") 
        ax[1, 0].set_title("Fast Fourier Transformed") 
        ax[2, 0].set_title("Ellipse Drawn") 
        ax[3, 0].set_title("Masked with Ellipse")  
        for i in range(len(imagesDir)): 
            ax[0, i].imshow(images[i], cmap=plt.cm.gray)  
            ax[1, i].imshow(images_FFT[i], cmap=plt.cm.gray) 
            ax[2, i].imshow(images_ellipse[i], cmap=plt.cm.gray) 
            ax[3, i].imshow(images_mask[i], cmap=plt.cm.gray) 
            ax[3, i].text(0.95, 0.01, "Number of Points: {}".format(images_noisecount[i]), verticalalignment='bottom', 
                          horizontalalignment='right', transform=ax[3, i].transAxes, color='green', fontsize=5) 
            # if i == curr_worst_index: 
            #     ax[3, i].set_xlabel("WORST IMAGE", color='red') 
            # if i == curr_best_index: 
            #     ax[3, i].set_xlabel("BEST IMAGE", color='green') 
            ax[3, 0].set_xlabel("Average noise count: {}".format(avg_noise), fontsize=10, color='red') 
        
        
        fig.savefig("{}/{}.png".format(curr_folder, saveName)) 


# In-situ tensile test methods:


# ------------------------------------------------------------------------------------ #
def printEndMessage(): 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⠶⣿⣭⡧⡤⣤⣻⣛⣹⣿⣿⣿⣶⣄') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⢀⣼⣊⣤⣶⣷⣶⣧⣤⣽⣿⣿⣿⣿⣿⣿⣷') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⢀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇') 
    print('⢀⢀⢀⢀⢀⢀⢀⣠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧ ')
    print('⢀⢀⢀⢀⢀⢀⠸⠿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣻⣿⣿⣿⣿⣿⡆ ')
    print('⢀⢀⢀⢀⢀⢀⢀⢸⣿⣿⡀⠘⣿⡿⢿⣿⣿⡟⣾⣿⣯⣽⣼⣿⣿⣿⣿⡀  ')
    print('⢀⢀⢀⢀⢀⢀⡠⠚⢛⣛⣃⢄⡁⢀⢀⢀⠈⠁⠛⠛⠛⠛⠚⠻⣿⣿⣿⣷ ')
    print('⢀⢀⣴⣶⣶⣶⣷⡄⠊⠉⢻⣟⠃⢀⢀⢀⢀⡠⠔⠒⢀⢀⢀⢀⢹⣿⣿⣿⣄⣀⣀⣀⣀⣀⣀ ') 
    print('⢠⣾⣿⣿⣿⣿⣿⣿⣿⣶⣄⣙⠻⠿⠶⠒⠁⢀⢀⣀⣤⣰⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄  ')
    print('⢿⠟⠛⠋⣿⣿⣿⣿⣿⣿⣿⣟⡿⠷⣶⣶⣶⢶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄  ') 
    print('⢀⢀⢀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠉⠙⠻⠿⣿⣿⡿  ') 
    print('⢀⢀⢀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⢀⢀⢀⠈⠁  ') 
    print('⢀⢀⢀⢀⢸⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿  ') 
    print('⢀⢀⢀⢀⢸⣿⣿⣿⣿⣄⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣹⣿⣿⣿⣿⣿⣿⣿⣿⠇  ') 
    print('⢀⢀⢀⢀⢀⢻⣿⣿⣿⣿⣧⣀⢀⢀⠉⠛⠛⠋⠉⢀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⠏  ') 
    print('⢀⢀⢀⢀⢀⢀⢻⣿⣿⣿⣿⣿⣷⣤⣄⣀⣀⣤⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋  ') 
    print('⢀⢀⢀⢀⢀⢀⢀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛  ') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⢀⢹⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁  ') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⢀⢸⣿⡇⢀⠈⠙⠛⠛⠛⠛⠛⠛⠻⣿⣿⣿⠇  ') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⢀⣸⣿⡇⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢨⣿⣿ ') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⣾⣿⡿⠃⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢸⣿⡏ ') 
    print('⢀⢀⢀⢀⢀⢀⢀⢀⠻⠿⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢠⣿⣿⡇ ')

# def main(): 

    # #CALIBRATING NONLOCALMEANS FILTER TEST 
    # #This one tests getting the mean intensity of a black 1000x1000 image with one white circle of radius 50 pixels 
    # img = cv2.imread(r"C:\Users\Irenaeus Wong\Desktop\dirtyImage.tif" , 0)

    # #Print original Metrics: 
    # print("\nMean Intensity: " + str( round(getMeanIntensity(img),2) ) )
    # print("Sharpness originally is: " + str( round(getSharpness(img),2) ) )
    # print("Gaussian Noise originally is: " + str( round(estimateGaussNoise(img),2)), end="\n\n")

    # cv2.imshow("Non Filtered", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    # hValues = calibrateNLMFilter(img, (10, 25))

    # print("\nThe best h value for sharpness is h = " + str(hValues[0]) ) 
    # print("The best h value for gaussian noise reduction is h = " + str(hValues[1]) ) 

    # #FOCUSSING EIFEL TOWER TEST
    # # img1 = cv2.imread(r"C:\Users\Irenaeus Wong\Desktop\focusData\eifel NF.tif", 0)
    # # img2 = cv2.imread(r"C:\Users\Irenaeus Wong\Desktop\focusData\eifel F.tif", 0)
    
    # # nfocussed = getSharpness(img1)
    # # focussed = getSharpness(img2) 

    # # print("Sharpness of not focussed image is: " + str(round(nfocussed,2)))
    # # print("Sharpness of focussed image is: " + str(round(focussed,2)))

# main() 