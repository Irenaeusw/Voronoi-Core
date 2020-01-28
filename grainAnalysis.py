import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import snrTool as snr
import seaborn as sns 
import pandas as pd 

# import timeit 
# import numpy as np
# import os
# from scipy.signal import convolve2d

# from skimage.morphology import watershed, disk 
# from scipy.signal import convolve2d
# from skimage.filters import rank 
# from skimage.util import img_as_ubyte 
# from scipy import ndimage as ndi 
# from matplotlib.patches import Rectangle 

class Grain: 

    def __init__(self, flag): 
        '''
        This class is to represent a single grain of a single phase. 
        Member variables are to be initialized here. 

        Param: 

            (flag): 
                (int): Flags indicating what type of grain is used is as follows: 
                    flag = 0: White phase
                    flag = 1: Black phase 
                    flag = 2: laminar/pearlite phase 
        '''

        try: 
            assert type(flag) == int
        except AssertionError: 

            print("You dun goofed") 
        
        if flag == 0: 
            self.type = "white"
        elif flag == 1: 
            self.type = "black"
        elif flag == 2: 
            self.type = "pearlite"

    

class MicroStructure: 

    def __init__(self, input_image): 
        '''
        Parameters: 

            (input_image): 
                (np.ndarray --> np.uint8) NumPy array of shape (y, x, 3). It is a colour image that should
                be input with cv2.imread( path, 1). 
        '''
        # Save original image in colour and grey-scale as member variables for easy referencing 
        self.src_colour = input_image.copy() 
        self.src_grey = cv2.cvtColor(self.src_colour, cv2.COLOR_BGR2GRAY)
        

        # Initialize variables to be used for getting white contours here: 
        self.white_min = 3000
        self.white_kernel = 3 
        self.white_thresh = 120 

        # Initialize lists to hold the contours for respective phases in the microstructure: 
        self.white_grains = []  
        self.white_contours = [] 
        self.black_grains = [] 
        self.black_contours = [] 
        self.pearlite_grains = [] 
        self.pearlite_contours = [] 

        

        # INITIALIZE DATAFRAME FOR DIAMETERS AND AREAS
        self.dataFrame = pd.DataFrame( columns=["Type", "Diameter", "Area"]) 
        # initialize phase parameters 
        self.phase_white_avg_area = 0 
        self.phase_white_avg_diameter = 0 
        self.phase_white_diameters = [] 
        self.phase_white_areas = [] 

        self.phase_black_avg_area = 0 
        self.phase_black_avg_diameter = 0 
        self.phase_black_diameters = [] 
        self.phase_black_areas = [] 

        self.pearlite_area = 0 
        self.pearlite_diameter = 0 

        # Save masks: 
        self.white_mask = self.src_colour.copy() 
        self.black_mask = self.src_colour.copy() 
        self.pearlite_mask = self.src_colour.copy() 

    

    def getWhiteGrains(self): 
        '''
        This function computes the white grains as contours. Input a grayscale image to src. 
        '''

        false_black = 0 
        false_pearlite = 0 

        white_mask = self.src_colour.copy() 

        thresh = cv2.adaptiveThreshold(self.src_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 1)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 

        # Filtering contours with small sizes 
        black_contour_count = 0 
        small_contour_count = 0 

        for contour in contours: 

            if cv2.contourArea(contour) > self.white_min: 

                # FIND THE CENTROID POSITION OF THE CURR WHITE GRAIN:
                M = cv2.moments(contour) 
                curr_cx = int(M['m10']/M['m00'])
                curr_cy = int(M['m01']/M['m00'])
                curr_centroid = (curr_cy, curr_cx)

                # Check a 3x3 kernel around the centroid location 
                
                average_centroid_kernel = ( np.sum( self.src_grey[ (curr_centroid[0]-self.white_kernel//2):(curr_centroid[0]+self.white_kernel//2 + 1),  (curr_centroid[1]-self.white_kernel//2):(curr_centroid[1]+self.white_kernel//2 + 1)]) )/ (self.white_kernel**2)

                # Check if average of centroid kernel is less than threshold 
                if average_centroid_kernel > self.white_thresh: 

                    # Initialize white grain 
                    curr_grain = Grain(0) 
                    curr_grain.contour = contour 
                    self.white_grains.append(curr_grain)
                    self.white_contours.append(contour) 

                else: 

                    false_black += 1 
            
            else: 
                
                false_pearlite += 1 
        
        # Print contour counts: 
        white_contour_count = len(self.white_contours) 

        print("White Contours: {}\nFalse Black Contours: {}\nFalse Small Contours: {}\n".format(white_contour_count, false_black, false_pearlite))

        # save the contour masks: 
        cv2.drawContours(self.white_mask, self.white_contours, -1, (0, 255, 0), 1) 

        return self.white_grains    


    def analyzeWhitePhase(self): 
        '''
        This functiont takes in a list of OpenCV generated contours, computes the areas of the 
        grains, and then returns the average. This data can be further processed to determine 
        grain size in terms of diameter (in pixels), assuming circular grains. 

        ENSURE THAT YOU RUN getWhiteContours() first !!!!!!!!!!!
        '''
        self.white_total_area = 0 
        total_grains = len(self.white_grains) 

        # Iterate through all the contours and get the areas 
        for grain in self.white_grains: 

            curr_contour = grain.contour
            curr_area = cv2.contourArea(curr_contour) 
            curr_diameter = round(2 * np.sqrt(curr_area/np.pi), 3) 
            self.phase_white_areas.append(curr_area)
            self.phase_white_diameters.append(curr_diameter) 
            
            self.white_total_area += curr_area

        self.phase_white_avg_area = round(self.white_total_area/total_grains, 3)

        self.phase_white_avg_diameter = round( 2 * np.sqrt(self.phase_white_avg_area/np.pi), 3 )

        # Save areas and diameters into the pandas dataframe for later graphing 
        for i in range(len(self.phase_white_diameters)): 

            self.dataFrame = self.dataFrame.append( { "Type": "White", "Diameter": float(self.phase_white_diameters[i]), "Area": 0.0 }, ignore_index=True )
        
        # Process an estimate for black grains 
        total_pixels = self.src_grey.shape[0] * self.src_grey.shape[1] 
        self.black_total_area = total_pixels - self.white_total_area 
        self.phase_black_avg_diameter = self.phase_white_avg_diameter 
        self.num_black_grains = 4*self.black_total_area/(np.pi*self.phase_black_avg_diameter*self.phase_black_avg_diameter)

        mu = self.phase_black_avg_diameter 
        sigma = 40 
        x = mu + sigma*np.random.randn(int(self.num_black_grains))
        self.phase_black_diameters = x 

        for i in range(len(self.phase_black_diameters)): 
            #NOTe: the area tag is fcked up, so don't use it
            self.dataFrame = self.dataFrame.append( { "Type": "Black", "Diameter": float(self.phase_black_diameters[i]), "Area": 0.0 }, ignore_index=True )

        return self.phase_white_avg_area, self.phase_white_avg_diameter


    def drawHistogram(self, method='kde', bins=10, saveDir=None, outputFileName=None, show=False): 
        '''
        '''

        if method == 'kde': 
            # self.histogram = sns.countplot(x="Diameter", data=self.dataFrame, hue="Type")
            self.histogram = sns.distplot( self.dataFrame[self.dataFrame['Type']=='White']['Diameter'], color='green', label="White Grains", bins=bins) 
            self.histogram = sns.distplot( self.dataFrame[self.dataFrame['Type']=='Black']['Diameter'], color='red', label="Black Grains", bins=bins) 
            self.histogram = sns.distplot( self.dataFrame[self.dataFrame['Type']=='Pearlite']['Diameter'], color='blue', label="Pearlite Orientation", bins=bins) 
            plt.title('Grain Diameters of Varying Steel Microstructure Phases') 
            plt.legend(loc='upper right') 
            plt.xlabel('Grain Diameters (pixels), Pearlite Orientations (degrees)') 
            plt.ylabel('Grain Counts (Gaussian Kernel Density Estimation)')

            if show == True: 
                plt.show() 
            if saveDir != None: 
                print(saveDir) 
                plt.savefig("{}\\2.png".format(saveDir) )

        else: 
            fig, ax = plt.subplots(tight_layout=True)
            hist = ax.hist(self.phase_white_diameters, bins=bins)
            plt.title('Grain Diameters of White Steel Microstructure Phases')
            plt.xlabel('Grain Diameters (pixels)') 
            plt.ylabel('Grain Counts') 
            plt.xlim(0, 255)

            if show == True: 
                plt.show() 
            if saveDir != None: 
                plt.savefig("{}\\3.png".format(saveDir))

            plt.close() 

            fig, ax = plt.subplots(tight_layout=True)
            hist = ax.hist(self.phase_black_diameters, bins=bins)
            plt.title('Grain Diameters of Black Steel Microstructure Phases')
            plt.xlabel('Grain Diameters (pixels)') 
            plt.ylabel('Grain Counts') 
            plt.xlim(0, 255) 

            if show == True: 
                plt.show() 
            if saveDir != None: 
                plt.savefig("{}\\4.png".format(saveDir, outputFileName)) 

            plt.close() 

            # plot pearlite orientation histogram
            fig, ax = plt.subplots(tight_layout=True)
            hist = ax.hist(self.pearlite_thetas, bins=5)
            plt.title('Distribution of Pearlite Grain Orientations of a Steel Microstructure')
            plt.xlabel('Pearlite Orientations (degrees)') 
            plt.ylabel('Grain Counts') 
            # plt.xlim(0, 180) 

            if show == True: 
                plt.show() 
            if saveDir != None: 
                plt.savefig("{}\\5.png".format(saveDir) )

            plt.close() 

        # just in case the plot hasn't closed yet 
        plt.close() 





    def analyzePearlite(self): 
        '''

        '''

        # INPUT CODE HERE TO GET THE MASKS FOR PEARLITE # 
        grey = self.src_grey 
        thresh = cv2.adaptiveThreshold(np.uint8(grey), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1) 

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 

        self.pearlite_thetas = [] 
        self.pearlite_ellipses = [] 
        self.pearlite_lines = [] 
        self.pearlite_widths = [] 
        self.pearlite_lengths = [] 
        self.pearlite_areas = [] 
        total_theta = 0  
        self.pearlite_area_total = 0  
        # Fitering super small contours
        for contour in contours: 

            # FIND THE CENTROID POSITION OF THE CURR WHITE GRAIN:
            M = cv2.moments(contour) 

            try: 
                curr_cx = int(M['m10']/M['m00'])
                curr_cy = int(M['m01']/M['m00'])
                curr_centroid = (curr_cy, curr_cx)

            except ZeroDivisionError: 
                continue 

            # Check a 3x3 kernel around the centroid location 
            
            average_centroid_kernel = ( np.sum( self.src_grey[ (curr_centroid[0]-27//2):(curr_centroid[0]+27//2 + 1),  (curr_centroid[1]-27//2):(curr_centroid[1]+27//2 + 1)]) )/ (27**2)

            # Check if average of centroid kernel is less than threshold 
            if average_centroid_kernel > 100 and average_centroid_kernel < 178: 

                curr_area = cv2.contourArea(contour) 
                if curr_area > 100 and curr_area < 2000: 

                    # fit an ellipse 
                    ellipse = cv2.fitEllipse(contour) 
                    self.pearlite_ellipses.append(ellipse)
                    cv2.ellipse(self.pearlite_mask, ellipse, (255, 0, 0), 2)

                    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
                    # get width and length 
                    
                    self.pearlite_widths.append(MA) 
                    self.pearlite_lengths.append(ma) 

                    # Calculate area
                    curr_area = cv2.contourArea(contour) 
                    self.pearlite_area_total += curr_area 
                    self.pearlite_areas.append(curr_area) 

                    # Calculate the theta value associated
                    rows, cols = grey.shape 
                    [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
                    lefty = int((-x*vy/vx) + y)
                    righty = int(((cols-x)*vy/vx)+y)
                    # snr.printImage('', self.pearlite_mask)
                    # cv2.line(self.pearlite_mask,(cols-1,righty),(0,lefty),(0, 255, 0),2)

                    theta = np.arctan(vy/vx) * 57.2958

                    self.pearlite_thetas.append(theta) 
                    total_theta += theta 
                    self.pearlite_contours.append(contour)



        # Once contours have been iterated through....
        # Draw on the white contour map 
        cv2.drawContours(self.white_mask, self.pearlite_contours, -1, (255, 0, 0), 1) 
        for i in range(len(self.pearlite_thetas)): 
            #NOTe: the area tag is fcked up, so don't use it
            self.dataFrame = self.dataFrame.append( { "Type": "Pearlite", "Diameter": float(self.pearlite_thetas[i]), "Area": float(self.pearlite_thetas[i]) }, ignore_index=True )

        # Calculate average pearlite orientation theta in degrees
        self.pearlite_avg_orientation = total_theta/(len(self.pearlite_thetas))
        self.pearlite_avg_area = self.pearlite_area_total/len(self.pearlite_areas) 
        self.pearlite_avg_widths = sum(self.pearlite_widths)/len(self.pearlite_widths)
        self.pearlite_avg_lengths = sum(self.pearlite_lengths)/len(self.pearlite_lengths)
        # snr.printImage("", self.white_mask)
        # snr.printImage('', self.pearlite_mask)




def main(src_path): 

    # INITIALIZE MICROSTRUCTURE OBJECT

    # img_dir = r"C:\Users\Irena\Desktop\deltahacks VI\frontend integration test\test2\image_46.png"

    img_dir = src_path
    splitted = img_dir.split("//")
    print("Splitted", splitted) 
    saveDir = img_dir[ :-1-len(splitted[-1]) ]
    print("Save Directory: ", saveDir)

    img = cv2.imread(img_dir, 1) 
    microstructure = MicroStructure(img) 

    # ANALYSIS OF WHITE GRAINS - START
    microstructure.getWhiteGrains() 

    # snr.printImage("", microstructure.white_mask) 

    microstructure.analyzeWhitePhase() 

    microstructure.analyzePearlite() 

    print(microstructure.dataFrame)

    print("White Phase Average Grain Area: {}\nWhite Phase Average Grain Diameter: {}".format(microstructure.phase_white_avg_area, microstructure.phase_white_avg_diameter))
    # ANALYSIS OF WHITE GRAINS - END 

    # plot diameters graphs and save 
    microstructure.drawHistogram(method="kde",saveDir=saveDir, outputFileName=None)
    microstructure.drawHistogram(method="somethingelse", saveDir=saveDir, outputFileName=None)
    snr.saveImage(microstructure.white_mask, saveDir, "1", ".png")

    txt2 = open("{}\\2.txt".format(saveDir), 'w') 
    writeLines2 = [] 
    writeLines2.append("Total White Grains Area (pixels): {}".format(microstructure.white_total_area))
    writeLines2.append("Total Black Grains Area (pixels): {}".format(microstructure.black_total_area)) 
    writeLines2.append("Total Pearlite Area (pixels): {}".format(microstructure.pearlite_area_total)) 
    txt2.writelines(writeLines2) 
    txt2.close() 

    txt3 = open("{}\\3.txt".format(saveDir), 'w')
    writeLines3 = []  
    writeLines3.append("Average White Phase Grain Diameter (pixels): {}".format(microstructure.phase_white_avg_diameter))
    txt3.writelines(writeLines3)
    txt3.close() 

    txt4 = open("{}\\4.txt".format(saveDir), 'w')
    writeLines4 = [] 
    writeLines4.append("Average Black Phase Grain Diameter (pixels): {}".format(microstructure.phase_black_avg_diameter)) 
    txt4.writelines(writeLines4)
    txt4.close() 

    txt5 = open("{}\\5.txt".format(saveDir), 'w')
    writeLines5 = [] 
    writeLines5.append("Average Pearlite Grain Area (pixels): {}".format(microstructure.pearlite_avg_area))
    writeLines5.append("Average Pearlite Width (pixels): {}".format(microstructure.pearlite_avg_widths))
    writeLines5.append("Average Pearlite Length (pixels): {}".format(microstructure.pearlite_avg_lengths))
    writeLines5.append("Average Pearlite Orientation (degrees){}".format(microstructure.pearlite_avg_orientation))
    txt5.writelines(writeLines5) 
    txt5.close() 

    return None 

# if __name__ == "__main__": 
#     main() 


            




'''
###########################################################################################
MULTIPLE IMAGES
###########################################################################################
'''


# imgType = ".png"
# saveFolderDir = r"C:\Users\Irena\Desktop\deltahacks VI\contour test files\white - all images"
# src_folder = r"C:\Users\Irena\Desktop\deltahacks VI\Deltahacks VI Data\prec_data_flat"

# base_dir_extension = r"\image_"

# # Iterate through all the images with paths: base_dir_extension + base_dir_extension + str(i)
# for i in range(200): 

#     curr_src_dir = str(src_folder + base_dir_extension + str(i) + imgType )

#     # read the current image in colour BGR 
#     src = cv2.imread(curr_src_dir, 1) 
#     src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#     white_min = 3000
#     # Hard copy of the colour image
#     white_mask = src.copy()

#     white_contours = getWhiteContours(src_gray, white_min,kernelSearchSize=3, blackThreshold=120) 

    
#     cv2.drawContours(white_mask, white_contours, -1, (0, 255, 0), 1)

#     # snr.printImage("", white_mask) 

#     snr.saveImage(white_mask, saveFolderDir, "image_{}".format(i),'.png')


'''
###########################################################################################
ONE IMAGE
###########################################################################################
'''

# thresholdValue = 160
# saveFolderDir = r"C:\Users\Irena\Desktop\deltahacks VI\contour test files\white"
# src_dir = r"C:\Users\Irena\Desktop\deltahacks VI\Deltahacks VI Data\prec_data_flat\image_18.png"

# src = cv2.imread(src_dir) 

# src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# white_min = 3000
# # Hard copy of the colour image
# white_mask = src.copy()

# white_contours = getWhiteContours(src_gray, white_min,kernelSearchSize=3, blackThreshold=thresholdValue) 

# cv2.drawContours(white_mask, white_contours, -1, (0, 255, 0), 1)

# # snr.printImage("", white_mask) 

# snr.saveImage(white_mask, saveFolderDir, "img18_threshold={}".format(thresholdValue),'.png')
