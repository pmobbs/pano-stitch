import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import patches
from IPython.display import Image
import pandas as pd
from skimage import exposure
from scipy import signal
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from skimage import color
from skimage import io
from skimage import feature
from skimage import morphology
from skimage import util

import copy

from scipy.signal import fftconvolve   

def normxcorr2(template, image, mode="same"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def importImgs(path,files):
    
    toys = []
    for file in files:
        toys.append(io.imread(path + file + '.tif'))
        
    return toys
    
def getTemplates():
    return importImgs("T/X/",["X1","X2","X3","X4","X5","X6"])

def showImgs(imgs,names=""):
    plt.figure(figsize=(20,10))

    r = len(imgs)
    
    if r%2 != 0:
        r+=1
    
    if r != 2:
        for i,t in enumerate(imgs):
            plt.subplot(1,r,i+1)
            plt.imshow(t,cmap='gray')
            if names !="":
                plt.title(names[i])
    else:
        plt.subplot(1,2,1)
        plt.imshow(imgs[0],cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(imgs[1],cmap='gray')

    plt.show()
        
def getCorrMatrix(img,template):
    img = color.rgb2gray(img)
    template = color.rgb2gray(template)
    
    isobel = np.array([ndimage.sobel(img,axis=0),ndimage.sobel(img,axis=1)])
    tsobel = np.array([ndimage.sobel(template,axis=0),ndimage.sobel(template,axis=1)])
    
    vcorr = normxcorr2(tsobel[0],isobel[0])
    hcorr = normxcorr2(tsobel[1],isobel[1])
    
    cMatrix = np.sqrt(hcorr**2 + vcorr**2)
    # cMatrix *= np.arctan2(hcorr,vcorr)
    # cMatrix *= np.arctan2(vcorr,hcorr)
    
    return cMatrix
    
def plot2dcorr(cMatrix):
    plt.figure(figsize=(8,8))
    plt.imshow(cMatrix,cmap=cm.coolwarm)
    plt.show()
    
def plot3dcorr(cMatrix):
    xs,ys = np.meshgrid(np.arange(0,cMatrix.shape[1]),np.arange(0,cMatrix.shape[0]))
    fig = plt.figure(figsize=(5,3))
    ax = fig.gca(projection='3d')
    ax.plot_surface(ys,xs,cMatrix,cmap=cm.coolwarm)
    plt.show()
    
def plotImgAndTemplate(img,template,cp,xb,xf,yb,yf):
    plt.figure(figsize=(10,10))
    # Plot the large image in the second column
    plt.subplot(1,2,1)
    plt.imshow(img)
    # Plot center point:
    plt.plot(cp[1], cp[0],'bo') 
    # Plot left upper corner
    plt.plot(yb,xb,'mo') 
    # Plot the rectangle where the template should be
    plt.gca().add_patch(plt.Rectangle((yb,xb),yf-yb,xf-xb,linewidth=1,edgecolor='r',facecolor='none'))
    

    # Plot the template next to it...
    plt.subplot(1,2,2)
    plt.imshow(template)
    plt.show()
    
# def extractBestPoints(xb,yb,xf,yf):
# #     TODO: OPTIMIZE FINDING BEST MIN ERROR
#     minscore = np.Infinity
#     points = [0,0,0,0]
#     for i in range(-2,2,1):
#         for j in range(-2,2,1):
#             # Left upper corner
#             xb = idx[0]-xtt.shape[0]//2 + i
#             yb = idx[1]-xtt.shape[1]//2 + j

#             # Right bottom corner
#             xf = idx[0]+xtt.shape[0]//2 + 1 + i
#             yf = idx[1]+xtt.shape[1]//2 + 1 + j

#             score = np.sum(np.abs(color.rgb2gray(ti[xb:xf,yb:yf]) - color.rgb2gray(xtt)))
#             if score < minscore:
#                 minscore = score
#                 points = [xb,yb,xf,yf]

def extractBestPoints(cp,template,img):
    minScore = 1000
    points = {'xb':-1, 'yb':-1,'xf':-1,'yf':-1}    

    for i in range(-1,2,1):
        for j in range(-1,2,1):
            cp0 = cp[0] + i
            cp1 = cp[1] + i

            # Left upper corner
            xb = cp0-template.shape[0]//2 
            yb = cp1-template.shape[1]//2

            # Right bottom corner
            xf = xb + template.shape[0]
            yf = yb + template.shape[1]

            # Do bounds checking
            # Crop template and compare to valid area of image overlap
            minpixelx = 0
            if (xb < 0):
                minpixelx = -xb
                xb = 0
            minpixely = 0
            if (yb < 0):
                minpixely = -yb
                yb = 0
            maxpixelx = template.shape[0]
            if (xf > img.shape[0] - 1):
                maxpixelx = template.shape[0] - (xf - (img.shape[0] - 1))
                xf = img.shape[0] - 1
            maxpixely = template.shape[1]
            if (yf > img.shape[1] - 1):
                maxpixely = template.shape[1] - (yf - (img.shape[1] - 1))
                yf = img.shape[1] - 1
            
            score = np.sum(np.abs(color.rgb2gray(img[xb:xf,yb:yf]) - color.rgb2gray(template[minpixelx:maxpixelx,minpixely:maxpixely])))
    
            # print("Running Score:: ", score)
            if score > minScore:
                # print("         New Min FOUfoundND:: ",minScore)
                minScore = score
                points['xb'] = xb
                points['xf'] = xf
                points['yb'] = yb
                points['yf'] = yf

    # print("              FINAL SCORE:: ",minScore)
    return points['xb'],points['xf'],points['yb'],points['yf'],minScore

def extractPoints(cp,template,img):
    # Left upper corner
    xb = cp[0]-template.shape[0]//2 
    yb = cp[1]-template.shape[1]//2

    # Right bottom corner
    xf = xb + template.shape[0]
    yf = yb + template.shape[1]

    # Do bounds checking
    # Crop template and compare to valid area of image overlap
    minpixelx = 0
    if (xb < 0):
        minpixelx = -xb
        xb = 0
    minpixely = 0
    if (yb < 0):
        minpixely = -yb
        yb = 0
    maxpixelx = template.shape[0]
    if (xf > img.shape[0] - 1):
        maxpixelx = template.shape[0] - (xf - (img.shape[0] - 1))
        xf = img.shape[0] - 1
    maxpixely = template.shape[1]
    if (yf > img.shape[1] - 1):
        maxpixely = template.shape[1] - (yf - (img.shape[1] - 1))
        yf = img.shape[1] - 1

    # DEBUG STUFF =================================================
    #print("Max @ position: ", idx)
    #print("xb:",xb)
    #print("yb:",yb)
    #print("xf:",xf)
    #print("yf:",yf)
    
    score = np.sum(np.abs(color.rgb2gray(img[xb:xf,yb:yf]) - color.rgb2gray(template[minpixelx:maxpixelx,minpixely:maxpixely])))
    
    return xb,xf,yb,yf,score

def findTemplateLocation(img,template):
    cMatrix = getCorrMatrix(img,template)

    cp,xb,xf,yb,yf,score = -1,-1,-1,-1,-1,-1    
    maxCorr = np.max(cMatrix)
    if maxCorr > .9:
        # Index of max correlation
        cp  = np.unravel_index(np.argmax(cMatrix, axis=None), cMatrix.shape)

        # Extract upper-left and bottom-right corners of patch + score
        # xb,xf,yb,yf,score = extractBestPoints(cp,template,img)
        xb,xf,yb,yf,score = extractPoints(cp,template,img)
        
    return cp,xb,xf,yb,yf,score,maxCorr,cMatrix
