import numpy as np
import matplotlib.pyplot as plt
from os import environ
import os
import random
import sys

from sympy import true
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    nb = sys.argv = sys.argv[1]
    print(nb)
    suppress_qt_warnings()

    TOTAL_LENGTH = 19 # cm 
    TOTAL_HEIGHT_ =  -7# cm 
    TOTAL_HEIGHT =  5 # cm 
    HORIZONTAL_CENTER = 10 # cm 

    base = np.arange(-19, 19, 0.01)
    height = np.arange(-7 , 5, 0.01)
    SMALL_LG = np.arange(-19, -11.9, 0.01)
    SMALL_LD = np.arange(12, 19, 0.01)
    SMALL_MG = np.arange(-12, -4.4, 0.01)
    SMALL_MD = np.arange(4.5, 12.1, 0.01)
    MIDDLE = np.arange(-4.5, 4.6, 0.01)
    
    # add values 
    # choose coordinates randomly 
    # number of points 
    N_POINST = 3
    # generate images 
    N_IMAGES = int(nb)
    for index in range(N_IMAGES):
        # fill_between(x, y1,y2)
        plt.fill_between(SMALL_LG, -7, 5, facecolor='black', interpolate=True)
        plt.fill_between(SMALL_LD, -7, 5, facecolor='black', interpolate=True)
        plt.fill_between(SMALL_MG, 0, 5, facecolor='black', interpolate=True)
        plt.fill_between(SMALL_MD, 0, 5, facecolor='black', interpolate=True)
        plt.fill_between(MIDDLE, 2, 5, facecolor='black', interpolate=True)
        plt.xticks(list(range(-19, 19)))
        plt.yticks(list(range(-7, 5)))
        plt.grid(color='black', linestyle='-', linewidth=1, zorder = 0)
        x_coordinates = []
        y_coordinates = []
        for i in range(N_POINST):
            x = random.sample(range(-19, 19),1)
            y = random.sample(range(-7, 5),1)
            x_coordinates.append(x[0])
            y_coordinates.append(y[0])
        
        # PMI coordinates 
        x_pmi = sum(x_coordinates) / len(x_coordinates)
        y_pmi = sum(y_coordinates) / len(y_coordinates)
        # compute the correction to do 
        x_pmi_int = round(x_pmi)
        y_pmi_int = round(y_pmi)
        if x_pmi_int < 0:
            corrx = "V-" + str(abs(x_pmi_int))
            #print(corrx)
        else:
            corrx = "D-" + str(abs(x_pmi_int))
            #print(corrx)
        if y_pmi_int < 0:
            corry = "V-" + str(abs(y_pmi_int))
            #print(corrx)
        else:
            corry = "D-" + str(abs(y_pmi_int))
            #print(corrx)
        print(corrx + corry)
        label = corrx + corry
        # draw the points 
        plt.scatter(x_coordinates, y_coordinates, s = 500, c = 'gray', zorder = 3)
        plt.scatter(x_pmi, y_pmi, s = 500, c = 'red', zorder = 3)

        if os.path.isdir(label):
            plt.savefig(label + "/"  +str(index) + ".png")
        else:
            os.mkdir(label)
            plt.savefig( label + "/" +str(index) + ".png")

        plt.close()
 