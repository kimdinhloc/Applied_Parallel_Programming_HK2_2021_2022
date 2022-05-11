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

