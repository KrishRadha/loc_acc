import numpy as np
import cv2,os
from matplotlib import pyplot as plt
from osgeo import gdal,ogr
from math import ceil,floor
def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
def getMasSlaCoords(kp1,kp2,matches,val):
    points=[]
    vals=[]
    i=0
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        points.append([(x1,y1,x2,y2),val[i]])
        i+=1
    return points
def converto8bit(img1,img2):
    stretch=(2**16)-1             #"""-------------------->>>>>>must change with input data type, as of now 16 bit"""
    max1=np.max(img1)             #"""-------------------->>>>>>Not considering no data or assuming no data value is > 0"""
    min1=np.min(img1)
    img1=(img1-min1)*(stretch/(max1-min1))
    img1=img1.astype('float')
    img1 = (img1*255/float(np.max(img1))).astype('uint8')
    max2=np.max(img2)            #"""-------------------->>>>>>Not considering no data or assuming no data value is > 0"""
    min2=np.min(img2)
    img2=(img2-min2)*(stretch/(max2-min2))
    img2=img2.astype('float')
    img2 = (img2*255/float(np.max(img2))).astype('uint8')
    return img1,img2
def imageMatch(img1,img2):
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]
    val=[]
    good=[]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
            val.append(m.distance/n.distance)
    points=getMasSlaCoords(kp1,kp2,good,val)
    drawMatches(img1,kp1,img2,kp2,good)                 #-------------------------------enable to show images
    return points
def chunks(raster1,raster2,bands):
    columns=raster2.RasterXSize
    rows=raster2.RasterYSize
    for b in bands:
        band1=raster1.GetRasterBand(b)
        band2=raster2.GetRasterBand(b)
        approxPixels=1000     #------------------------------------approximate block size/ no of pixels in each block
        parts=max(rows,columns)/approxPixels
        for i in range(0,parts):
            if i<parts-1:
                p=int(floor(float(columns)/parts))
            else:
                p=columns-i*int(floor(float(columns)/parts))
            for j in range(0,parts):
                if j<parts-1:
                    q=int(floor(float(rows)/parts))
                else:
                    q=rows-j*int(floor(float(rows)/parts))
                raster1Gt=raster1.GetGeoTransform()
                raster2Gt=raster2.GetGeoTransform()
                print raster1Gt,raster2Gt
                scaleDiff=[raster2Gt[1]/raster1Gt[1],raster2Gt[5]/raster1Gt[5]]
                img2=band2.ReadAsArray(i*int(floor(float(columns)/parts)),j*int(floor(float(rows)/parts)),p,q)

                midColLon=raster2.GetGeoTransform()[0]+((i*int(floor(float(columns)/parts)))+int(float(img2.shape[1])/2))*raster2.GetGeoTransform()[1]

                midRowLat=raster2.GetGeoTransform()[3]+((j*int(floor(float(rows)/parts)))+int(float(img2.shape[0])/2))*raster2.GetGeoTransform()[5]
                raster1Cols=int(img2.shape[1]*scaleDiff[0])-10          ##########################-10 for safety
                raster1Rows=int(img2.shape[0]*scaleDiff[1])-10          ##########################-10 for safety
                img1col,img1row= (int((midColLon-raster1Gt[0])/raster1Gt[1]),int((midRowLat-raster1Gt[3])/raster1Gt[5]))

                img1=band1.ReadAsArray(img1col-int(raster1Cols/2.0),img1row-int(raster1Rows/2.0),raster1Cols,raster1Rows)
                img1,img2=converto8bit(img1,img2)
                if i==j:                                #--------------------------------for testing else remove if clause and run in main thread
                    matchPoints=imageMatch(img1,img2)
                    print matchPoints
def main():

    raster1 = gdal.Open('C:\\Users\\Bharath\\Documents\\TDP\\Images\\sentinel2set1.tif')
    raster2 = gdal.Open('C:\\Users\\Bharath\\Documents\\TDP\\Images\\l4set1.img')
    bands=[2]
    chunks(raster1,raster2,bands)
    os.sys.exit()
    stretch=(2**16)-1             #"""-------------------->>>>>>must change with input data type, as of now 16 bit"""
    img1 = raster1.GetRasterBand(2).ReadAsArray()
    # cv2.imshow('Matched Features', img1)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    max1=np.max(img1)             #"""-------------------->>>>>>Not considering no data or assuming no data value is > 0"""
    min1=np.min(img1)
    img1=(img1-min1)*(stretch/(max1-min1))
    img1=img1.astype('float')
    img1 = (img1*255/float(np.max(img1))).astype('uint8')
    img2 = raster2.GetRasterBand(2).ReadAsArray()
    max2=np.max(img2)            #"""-------------------->>>>>>Not considering no data or assuming no data value is > 0"""
    min2=np.min(img2)
    img2=(img2-min2)*(stretch/(max2-min2))
    img2=img2.astype('float')
    img2 = (img2*255/float(np.max(img2))).astype('uint8')
    img1=img1[:1000,:1000]
    img2=img2[:1000,:1000]
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    good=[]
    val=[]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance and n.distance!=0:
            matchesMask[i]=[1,0]
            good.append(m)
    getMasSlaCoords(kp1,kp2,good,val)
    drawMatches(img1,kp1,img2,kp2,good)
if __name__=="__main__":
    main()
