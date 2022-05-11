import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation
from numba import jit, prange, cuda
import time
import math
import sys
import numba as nb

def load_model(Image):
    ''' 
    Load YOLOv3 model and detect objects 
    return:
    outs: list of detected objects
    '''
    try:
        configuration = "yolov3.cfg"
        weights = "yolov3.weights"
        classesFile = "coco.names"
        classes = None
    
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        net = cv2.dnn.readNetFromDarknet(configuration, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except:
        print("Error: Cannot load model")
        sys.exit()
    try:
        inputWidth,inputHeight=608,608
        blob = cv2.dnn.blobFromImage(Image, 1 / 255, (inputWidth, inputHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outs = net.forward([layersNames[index[0] - 1] for index in net.getUnconnectedOutLayers()])
    except:
        print("Error: Cannot detect objects")
        sys.exit()
    return outs
    
def postprocess(frameHeight, frameWidth, outs,confThreshold=0.0):
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
    return boxes
#convert to CIE-LAB

def BRG2CIELAB(inPixels):
    '''
    Convert BRG to CIELAB

    praram:
    ----
    inPixels: numpy array of shape (B,G,R)

    output:
    ----
    outPixels: numpy array of shape (L,a,b)
    '''

    #convert BGR to XYZ
    #https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
    #BRG -> CIE XYZ.Rec 709 with D65 white point
    index = 0
    for value in inPixels:
        if value/255 > 0.04045:
            inPixels[index] = (((value/255 + 0.055) / 1.055) ** 2.4)*100
        else:
            inPixels[index] = value/255 * 100/12.92
        index += 1

    XYZ_colvolution = np.matrix([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]], dtype=np.float32)
    BGR = np.matrix([inPixels[2], inPixels[1], inPixels[0]]).T #.T is transpose to 3x1 matrix
    XYZ = np.dot(XYZ_colvolution, BGR)
    # Observer= 2°, Illuminant= D65
    XYZ[0]=XYZ[0]/95.047
    XYZ[1]=XYZ[1]/100.000
    XYZ[2]=XYZ[2]/108.883

    #convert XYZ to CIE-LAB
    #https://en.wikipedia.org/wiki/Lab_color_space

    index=0
    for value in XYZ:
        if value>0.008856:
            XYZ[index]=np.power(value,1/3)
        else:
            XYZ[index]=(7.787*value)+(16/116)
        index+=1

    L=float(116*XYZ[1]-16)
    a=float(500*(XYZ[0]-XYZ[1]))
    b=float(200*(XYZ[1]-XYZ[2]))
    return [round(L,4),round(a,4),round(b,4)]

def convert2CIELAB(inImg):
    '''
    Convert RGB image to CIELAB
    params:
    inImg - image input
    return:
    outImg - image output
    '''    
    #initialize the output image
    outImg=np.zeros(inImg.shape,dtype=np.float32)

    #loop over the image, and convert the RGB values to CIELAB each pixel
    for h in range(inImg.shape[0]):
        for w in range(inImg.shape[1]):
            outImg[h,w,:]=BRG2CIELAB(inImg[h,w,:])
            
    return outImg

def find_local_minimum(inImg,center):
    minGradient = 1
    localMinium = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            cluster1 = inImg[j+1, i]
            cluster2 = inImg[j, i+1]
            cluster3 = inImg[j, i]
            C=np.sqrt(pow(float(cluster1[0] - cluster3[0]),2)) +  np.sqrt(pow(float(cluster2[0] - cluster3[0]),2))
            if C < minGradient:
                minGradient = abs(cluster1[0] - cluster3[0]) + abs(cluster2[0] - cluster3[0])
                localMinium = [i, j]
    return localMinium

def calculate_centers(inImg,S):
    '''
    Calculate the centers of the segments
    params:
    inImg - image input
    S - number of segments
    return:
    centers - list of centers
    '''
    centers = []
    for w in range(S, inImg.shape[1] - int(S/2), S):
        for h in range(S, inImg.shape[0] - int(S/2), S):
            nc = find_local_minimum(inImg,center=(w, h))
            color = inImg[nc[1], nc[0]] #height x width
            center = [color[0], color[1], color[2], nc[0], nc[1]] # l, a, b, height, width
            centers.append(center)
    return centers

def generate_pixels(inImg,interations,distances,centers,clusters,S,m):
    '''
    Generate the segments
    params:
    --------
    inImg - image input
    interations - number of loops
    distances - list of distances
    centers - list of centers
    clusters - list of clusters
    S - number of segments
    m - parameter for SLIC segmentation

    return:
    --------
    centers - list of centers
    distances - list of distances
    clusters - list of clusters
    '''

    #init grid of pixels
    indnp = np.mgrid[0:inImg.shape[0],0:inImg.shape[1]].swapaxes(0,2).swapaxes(0,1)
    Img=inImg.copy()

    #loop over the iterations
    for i in range(interations):
        distances = 1 * np.ones(Img.shape[:2])
        for index in range(centers.shape[0]):
            
            #limit of the grid
            x_low = int(max(centers[index][3]-S,0))
            x_high = int(min(centers[index][3]+S,Img.shape[1]))
            y_low = int(max(centers[index][4]-S,0))
            y_high = int(min(centers[index][4]+S,Img.shape[0]))

            #crop
            cropimg = Img[y_low : y_high , x_low : x_high]
            
            # color difference
            color_diff = cropimg - Img[int(centers[index][4]), int(centers[index][3])]

            # color distance
            color_distance = np.sqrt(np.sum(np.square(color_diff), axis=2))
            ny, nx = np.ogrid[y_low : y_high, x_low : x_high]
            pixdist = ((ny-centers[index][4])**2 + (nx-centers[index][3])**2)**0.5
            dist = ((color_distance/m)**2 + (pixdist/S)**2)**0.5
            distance_crop = distances[y_low : y_high, x_low : x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            distances[y_low : y_high, x_low : x_high] = distance_crop
            clusters[y_low : y_high, x_low : x_high][idx] = index

        for k in range(len(centers)):
            idx = (clusters == k)
            colornp = inImg[idx]
            distnp = indnp[idx]
            centers[k][0:3] = np.sum(colornp, axis=0)
            sumy, sumx = np.sum(distnp, axis=0)
            centers[k][3:] = sumx, sumy
            centers[k]= centers[k]/np.sum(idx)
    return centers,distances,clusters

def create_connectivity(inImg,centers,clusters):
    '''
    Create the connectivity matrix
    params:
    inImg - image input
    centers - list of centers
    clusters - list of clusters
    return:
    SLIC_new_clusters - list of clusters
    '''
    label = 0
    adj_label = 0
    lims=int(inImg.shape[0]*inImg.shape[1]/centers.shape[0])
    
    new_clusters = -1 * np.ones(inImg.shape[:2]).astype(np.int64)
    elements = []
    for i in range(inImg.shape[1]):
        for j in range(inImg.shape[0]):
            if new_clusters[j, i] == -1:
                elements = []
                elements.append((j, i))
                for dx, dy in [(-1,0), (0,-1), (1,0), (0,1)]:
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if (x>=0 and x < inImg.shape[1] and 
                        y>=0 and y < inImg.shape[0] and 
                        new_clusters[y, x] >=0):
                        adj_label = new_clusters[y, x]
            count = 1
            counter = 0
            while counter < count:
                for dx, dy in [(-1,0), (0,-1), (1,0), (0,1)]:
                    x = elements[counter][1] + dx
                    y = elements[counter][0] + dy

                    if (x>=0 and x<inImg.shape[1] and y>=0 and y<inImg.shape[0]):
                        if new_clusters[y, x] == -1 and clusters[j, i] == clusters[y, x]:
                            elements.append((y, x))
                            new_clusters[y, x] = label
                            count+=1
                counter+=1
            if (count <= lims >> 2):
                for counter in range(count):
                    new_clusters[elements[counter]] = adj_label
                label-=1
            label+=1
    SLIC_new_clusters = new_clusters
    return SLIC_new_clusters
