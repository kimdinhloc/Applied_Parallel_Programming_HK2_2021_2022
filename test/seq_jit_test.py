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

@jit(nopython=True, cache=True)
def BGR2Lab(in_pixels, out_pixels):
    '''
    convert color image to Lab
    #https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_lab
    prams:
        in_pixels: input color image
        out_pixels: output cielab image
    '''
    #XYZ_colvolution = np.array([[0.412453, 0.357580, 0.180423],
    #                            [0.212671, 0.715160, 0.072169],
    #                            [0.019334, 0.119193, 0.950227]])
    for row in range(len(in_pixels)):
        for col in range(len(in_pixels[0])):
            RGB = [in_pixels[row][col][2], in_pixels[row][col][1], in_pixels[row][col][0]]

            X = (RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805)/95.047
            Y = (RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722)/100.000
            Z = (RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505)/108.883

            if Y>0.008856:
                Y=Y**(1/3)
            else:
                Y=(7.787*Y)+16/116
            if X>0.008856:
                X=X**(1/3)
            else:
                X=(7.787*X)+16/116
            if Z>0.008856:
                Z=Z**(1/3)
            else:
                Z=(7.787*Z)+16/116
            L=116*Y-16
            a=500*(X-Y)+128
            b=200*(Y-Z)+128
            out_pixels[row][col][0]=round(L,0)
            out_pixels[row][col][1]=round(a,0)
            out_pixels[row][col][2]=round(b,0)
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

def SLIC(srcImage,segmentSize,m=20,enforce_connectivity=False):
    '''
    SIMPLE LINEAR IMAGE SEGMENTATION

    params:
    ----------
    srcImage - image to be segmented
    segmentSize - size of segments
    m = parameter for SLIC
    enforce_connectivity - if True, enforce connectivity

    return:
    ----------
    labels - segmented image
    '''
    # initialize SLIC
    Image = srcImage.copy()
    S=segmentSize
    width=Image.shape[1]
    height=Image.shape[0]
    interations=10

    # convert image to CieLAB
    labImage=np.zeros(shape=Image.shape, dtype=np.float64)

    BGR2Lab(Image,labImage)
    
    # initialize distance
    distances = 1 * np.ones(labImage.shape[:2])

    # initialize cluster
    clusters = -1 * distances

    # initialize cluster centers counter
    center_counts = np.zeros(len(calculate_centers(labImage,S)))

    # initialize cluster centers
    centers = np.array(calculate_centers(labImage,S))

    # generate superpixels
    centers,distances,clusters=generate_pixels(labImage,interations,distances,centers,clusters,S,m)

    if enforce_connectivity:
        new_cluster=create_connectivity(labImage,centers,clusters)
        centers=calculate_centers(labImage,S)
    else:
        new_cluster=clusters
    return new_cluster

@jit(nopython=True, cache=True)
def create_segmentation_mask(label,segmented,seg_mask,outline):
    '''
    Create segmentation mask
    Parameters:
    ----
    label: label of segment
    segmented: segmented image
    seg_mask: segmentation mask
    outline: outline of segment
    seg_mask: segmentation mask
    '''
    for i in range(outline.shape[0]):
        for j in range(outline.shape[1]):
            if label[outline[i,j]]:
                seg_mask[i,j]=1
            else:
                seg_mask[i,j]=0
@jit(nopython=True, cache=True)
def Graussian_Blur_kernel_2D(size,kernel):
    '''
    Create 2D Gaussian kernel
    Parameters:
    ----
    size: size of kernel
    '''
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - size / 2) ** 2 + (j - size / 2) ** 2) / (2 * (size / 2) ** 2))
    kernel = kernel / np.sum(kernel)
@jit(nopython=True, cache=True)
def meger_mask(srcImage,seg_mask,outImage):
    '''
    Merge mask with original image
    Parameters:
    ----
    srcImage: Original image
    seg_mask: Segmentation mask
    '''
    height, width = srcImage.shape[:2]
    for x in range(0, width):
        for y in range(0, height):
            if seg_mask[y, x] ==0:
                outImage[y, x] = [0, 0, 0]
            else:
                outImage[y, x] = srcImage[y, x]

def Graussian_Blur(kernel,srcImage,dstImage):
    '''
    Gaussian blur
    Parameters:
    ----
    kernel: kernel matrix
    srcImage: source image
    dstImage: destination image
    '''
    shape_of_kernel = kernel.shape[0]
    #Create new image with same size as srcImage + kernel size
    new_image = np.zeros(shape=(srcImage.shape[0] + shape_of_kernel-1, srcImage.shape[1] + shape_of_kernel-1), dtype=np.uint8)
    #Copy srcImage to new_image
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            if i <shape_of_kernel//2 or i>=new_image.shape[0]-shape_of_kernel//2 or j<shape_of_kernel//2 or j>=new_image.shape[1]-shape_of_kernel//2:
                new_image[i,j]=0
            else:
                new_image[i,j]=srcImage[i-shape_of_kernel//2,j-shape_of_kernel//2]

    #convolution
    for width in range(0, srcImage.shape[1]):
        for height in range(0, srcImage.shape[0]):

            new_image[height, width] = np.ceil(np.sum(new_image[height:height+shape_of_kernel, width:width+shape_of_kernel]*kernel , axis=(0,1)))
    
    dstImage[:,:] = new_image[shape_of_kernel//2-1:-shape_of_kernel//2, shape_of_kernel//2-1:-shape_of_kernel//2]
@jit(nopython=True, cache=True)
def count_pixels_segment(outline,regions,pxseg):
    '''
    Count pixels in each segment
    Parameters:
    ----
    outline: outline of segment
    regions: segmented image
    '''
    for h in range(outline.shape[0]):
        for w in range(outline.shape[1]):
            if regions[h,w]!=0:
                pxseg[outline[h,w]]+=1
@jit(nopython=True, cache=True)
def threshold_each_label(pxseg,pxtotal,threshold,label):
    '''
    Threshold each segment
    Parameters:
    ----
    pxseg: pixels in each segment
    pxtotal: total pixels
    threshold: threshold
    '''
    for i in range(pxseg.shape[0]):
        if pxseg[i]/pxtotal[i]>=threshold:
            label[i]=True
        else:
            label[i]=False
def remove_background(srcImage,segmentSize,m=20,enforce_connectivity=False,threshold=0.25,graussianKernelSize=5):
    '''
    Removing background - Xoá background
    Parameters:
    ----
    filename: Đường dẫn chứa hình ảnh đầu vào
    segmentSize: Kích thước của mỗi segment
    m: Parameter for SLIC
    enforce_connectivity: If True, enforce connectivity
    threshold: Threshold for removing background
  
    '''
    Image=np.zeros(srcImage.shape)
    #Load image
    Image=srcImage.copy()
    Width, Height = Image.shape[1], Image.shape[0]

    #load model
    start = time.time()
    outs= load_model(Image)

    #Box of Object
    boxes = postprocess(Height, Width, outs)
    if len(boxes):
      [left, top, width, height] = boxes[0]
    else:
      [left, top, width, height] = [0, Height, Width, Height]
    print("Load model: ", time.time() - start)
    #Superpixels segmentation using SLIC
    start = time.time()
    outline=SLIC(Image, segmentSize,m=20,enforce_connectivity=True)
    print("SLIC: ", time.time() - start)
    #Grab Cut
    start = time.time()
    mask = np.zeros(Image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = [left, top, left + width, top + height]
    cv2.grabCut(Image, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(Image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    print("Grab Cut: ", time.time() - start)
    #grab mask
    start = time.time()
    grabMask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    regions=grabMask * outline

    segmented = np.unique(regions)
    segmented = segmented[1 : len(segmented)]

    pxtotal = np.bincount(outline.flatten())

    pxseg=np.zeros(np.unique(outline).shape[0],dtype=np.int64)
    count_pixels_segment(outline,regions,pxseg)

    label = np.zeros(pxseg.shape[0], dtype=np.bool_)
    threshold_each_label(pxseg,pxtotal,threshold,label)
    seg_mask = np.zeros(shape=Image.shape[:2], dtype=np.uint8)
    create_segmentation_mask(label,segmented,seg_mask,outline)
    print("Segmentation mask: ", time.time() - start)
    
    #Graussian Blur
    start = time.time()
    blurImg=np.zeros(shape=Image.shape[:2], dtype=np.float64)
    #blurImg = cv2.GaussianBlur(seg_mask, (graussianKernelSize, graussianKernelSize), 0)

    if graussianKernelSize % 2 == 0:
        graussianKernelSize=graussianKernelSize+1

    kernel = np.zeros((graussianKernelSize, graussianKernelSize))
    Graussian_Blur_kernel_2D(graussianKernelSize,kernel)
    
    blurImg=np.zeros(shape=Image.shape[:2], dtype=np.float64)
    Graussian_Blur(kernel,seg_mask,blurImg)
    print("Graussian Blur: ", time.time() - start)
    # Meger segmentation mask with original image
    start = time.time()
    outImg = np.empty(shape=Image.shape, dtype=np.uint8)
    meger_mask(Image,blurImg,outImg)
    print("Meger mask: ", time.time() - start)
    return outImg

def main():
    '''
    args:
    ----
    filename: file name of image
    segmentSize: size of segment
    m: Parameter for SLIC
    enforce_connectivity: If True, enforce connectivity
    threshold: Threshold for removing background

    example:
    ----
    python remove_background.py filename segmentSize m enforce_connectivity threshold
    > python remove_background.py test.jpg 200 20 True 0.25

    OR USING RECOMMENDED PARAMETERS
    python remove_background.py filename
    > python remove_background.py test.jpg
    '''
    parser = sys.argv
    if len(parser) != 7 and len(parser) != 2:
        print("Please input image path")
        return
    elif len(parser) == 7:
        filename = parser[1]
        segmentSize = int(parser[2])
        m = int(parser[3])
        enforce_connectivity = bool(parser[4])
        threshold = float(parser[5])
        graussianKernelSize = int(parser[6])
        srcImage = cv2.imread(filename)
        outImg = remove_background(srcImage,segmentSize,m,enforce_connectivity,threshold,graussianKernelSize)
        cv2.imwrite(str(filename)+"_output.jpg", outImg)
    elif len(parser) == 2:
        filename = parser[1]
        srcImage = cv2.imread(filename)
        segmentSize=5
        # if square image >10000000, segmentSize increase 2.5 each 1 milion more than 1 million
        if srcImage.shape[0]*srcImage.shape[1]>10000000:
            segmentSize=int(segmentSize+2.5*(srcImage.shape[0]*srcImage.shape[1]-1000000)//1000000)
        m=20
        enforce_connectivity=False
        threshold=0.25
        graussianKernelSize=segmentSize
        outImg = remove_background(srcImage,segmentSize,m,enforce_connectivity,threshold,graussianKernelSize)
        cv2.imwrite(str(filename.split(".")[0])+"_output.jpg", outImg)

if __name__ == '__main__':
    main()