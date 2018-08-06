import numpy as np
import cv2,os
from matplotlib import pyplot as plt
from osgeo import gdal,ogr
from math import ceil,floor
import getReferenceFiles
import json


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
    factor1 = img1.nbytes/(img1.shape[0]*img1.shape[1])
    factor2 = img2.nbytes/(img2.shape[0]*img2.shape[1])
    stretch1=(2**(8*factor1))-1
    stretch2=(2**(8*factor2))-1
    max1=np.max(img1)             #"""-------------------->>>>>>Not considering no data or assuming no data value is > 0"""
    min1=np.min(img1)
    if (max1-min1)!=0:
        img1=(img1-min1)*(stretch1/(max1-min1))
    img1=img1.astype('float')
    if np.max(img1)!=0:
        img1 = ((img1*254/float(np.max(img1)))+1).astype('uint8')
    else:
        return 0
    max2=np.max(img2)            #"""-------------------->>>>>>Not considering no data or assuming no data value is > 0"""
    min2=np.min(img2)
    if (max2-min2)!=0:
        img2=(img2-min2)*(stretch2/(max2-min2))
    img2=img2.astype('float')
    if np.max(img2)!=0:
        img2 = ((img2*254/float(np.max(img2)))+1).astype('uint8')
    else:
        return 0
    return img1,img2
def imageMatch(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    if (des1 is not None and  des2 is not None):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(0,len(matches))]
        val=[]
        good=[]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.65*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
                val.append(m.distance/n.distance)
        points=getMasSlaCoords(kp1,kp2,good,val)
        # drawMatches(img1,kp1,img2,kp2,good)                 #-------------------------------enable to show images
        return points,1
    else:
        return [],0
def imageMatchORB(img1,img2):
    orb = cv2.ORB()
    kp1 = orb.detect(img1,None)
    kp2 = orb.detect(img2,None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    if (des1 is not None and  des2 is not None):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1,des2)
        # Need to draw only good matches, so create a mask
        # matchesMask = [[0,0] for i in xrange(len(matches))]
        good=[]
        val=[]
        # good = sorted(matches, key = lambda x:x.distance)
        # good=good[:int(len(good)/2)]
        # val=[0*x for x in range(0,len(good))]
        # ratio test as per Lowe's paper
        for eachMatch in matches:
            print (eachMatch.distance)
            if eachMatch.distance < 0.99999:
                good.append(eachMatch)
                val.append(eachMatch.distance)
        points=getMasSlaCoords(kp1,kp2,good,val)
        # drawMatches(img1,kp1,img2,kp2,good)                 #-------------------------------enable to show images
        return points,1
    else:
        return [],0
def processPair(raster1,raster2,bands):
    raster1Gt,raster2Gt=(raster1.GetGeoTransform(),raster2.GetGeoTransform())
    raster1Bounds=(raster1Gt[0],raster1Gt[3],raster1Gt[0]+raster1.RasterXSize*raster1Gt[1],raster1Gt[3]+raster1.RasterYSize*raster1Gt[5])
    raster2Bounds=(raster2Gt[0],raster2Gt[3],raster2Gt[0]+raster2.RasterXSize*raster2Gt[1],raster2Gt[3]+raster2.RasterYSize*raster2Gt[5])
    intersectionBounds=(max(raster1Bounds[0],raster2Bounds[0]),min(raster1Bounds[1],raster2Bounds[1]),min(raster1Bounds[2],raster2Bounds[2]),max(raster1Bounds[3],raster2Bounds[3])) #---------------------------Works only with GCS/WGS84
    array1Bounds=(int(ceil((intersectionBounds[0]-raster1Bounds[0])/raster1Gt[1])),int(ceil((intersectionBounds[1]-raster1Bounds[1])/raster1Gt[5])),int(floor((intersectionBounds[2]-raster1Bounds[0])/raster1Gt[1])),int(floor((intersectionBounds[3]-raster1Bounds[1])/raster1Gt[5])))
    array2Bounds=(int(ceil((intersectionBounds[0]-raster2Bounds[0])/raster2Gt[1])),int(ceil((intersectionBounds[1]-raster2Bounds[1])/raster2Gt[5])),int(floor((intersectionBounds[2]-raster2Bounds[0])/raster2Gt[1])),int(floor((intersectionBounds[3]-raster2Bounds[1])/raster2Gt[5])))
    array1Gt=(raster1Gt[0]+raster1Gt[1]*array1Bounds[0],raster1Gt[1],raster1Gt[2],raster1Gt[3]+raster1Gt[5]*array1Bounds[1],raster1Gt[4],raster1Gt[5])
    array2Gt=(raster2Gt[0]+raster2Gt[1]*array2Bounds[0],raster2Gt[1],raster2Gt[2],raster2Gt[3]+raster2Gt[5]*array2Bounds[1],raster2Gt[4],raster2Gt[5])
    print (array1Gt,array2Gt)
    points=chunks(raster1,raster2,array1Gt,array2Gt,array1Bounds,array2Bounds,bands)
    return points
def chunks(raster1,raster2,array1Gt,array2Gt,array1Bounds,array2Bounds,bands):
    columns=array2Bounds[2]-array2Bounds[0]
    rows=array2Bounds[3]-array2Bounds[1]
    refcolumns=array1Bounds[2]-array1Bounds[0]
    refrows=array1Bounds[3]-array1Bounds[1]
    returnPoints=[]
    for b in bands:
        band1=raster1.GetRasterBand(b)
        band2=raster2.GetRasterBand(b)
        approxPixels=512     #------------------------------------approximate block size/ no of pixels in each block
        parts=max(rows,columns)/approxPixels
        for i in range(0,int(parts)):
            if i<parts-1:
                p=int(floor(float(columns)/parts))
                p2=int(floor(float(columns)/parts))
                p1=int(floor(float(refcolumns)/parts))
            else:
                p=columns-i*int(floor(float(columns)/parts))
                p2=columns-i*int(floor(float(columns)/parts))
                p1=refcolumns-i*int(floor(float(refcolumns)/parts))
            for j in range(0,int(parts)):
                if j<parts-1:
                    q=int(floor(float(rows)/parts))
                    q2=int(floor(float(rows)/parts))
                    q1=int(floor(float(refrows)/parts))
                else:
                    q=rows-j*int(floor(float(rows)/parts))
                    q2=rows-j*int(floor(float(rows)/parts))
                    q1=refrows-j*int(floor(float(refrows)/parts))
                if (i%2==0 and j%2==0):
                #if i==j:
                    print ('processing part',i,j,parts)
                    scaleDiff=[array2Gt[1]/array1Gt[1],array2Gt[5]/array1Gt[5]]
                    img2=band2.ReadAsArray(array2Bounds[0]+i*int(floor(float(columns)/parts)),array2Bounds[1]+j*int(floor(float(rows)/parts)),p2,q2)
                    img1=band1.ReadAsArray(array1Bounds[0]+i*int(floor(float(refcolumns)/parts)),array1Bounds[1]+j*int(floor(float(refrows)/parts)),p1,q1)
                    # ulColLon=array2Gt[0]+((i*int(floor(float(columns)/parts))))*array2Gt[1]
                    # ulRowLat=array2Gt[3]+((j*int(floor(float(rows)/parts))))*array2Gt[5]
                    # raster1Cols=int(img2.shape[1]*scaleDiff[0])-10          ##########################-10 for safety
                    # raster1Rows=int(img2.shape[0]*scaleDiff[1])-10          ##########################-10 for safety
                    # img1col,img1row= (int((ulColLon-array1Gt[0])/array1Gt[1]),int((ulRowLat-array1Gt[3])/array1Gt[5]))
                    #
                    # img1=band1.ReadAsArray(array1Bounds[0]+img1col+5,array1Bounds[1]+img1row+5,raster1Cols,raster1Rows)
                    conv=converto8bit(img1,img2)
                    if conv==0:
                        continue
                    else:
                        img1,img2=conv                          #--------------------------------for testing else remove if clause and run in main thread
                        # print img2
                        # print np.max(img2),np.min(img2)
                        # cv2.imshow('Matched Features',img1)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow('Matched Features')
                        matchPoints,logPoints=imageMatch(img1,img2)###########################################change this function to use orb or sift
                        if logPoints==0:
                            print ('Error in descriptor calculation in',i,j,'part')
                        if len(matchPoints)>0:
                            for eachMatchPoint in matchPoints:
                                lon2=array2Gt[0]+(float(i*int(floor(float(columns)/parts)))+eachMatchPoint[0][2])*array2Gt[1]
                                lat2=array2Gt[3]+(float(j*int(floor(float(rows)/parts)))+eachMatchPoint[0][3])*array2Gt[5]
                                lon1=array1Gt[0]+(float(i*int(floor(float(refcolumns)/parts)))+eachMatchPoint[0][0])*array1Gt[1]
                                lat1=array1Gt[3]+(float(j*int(floor(float(refrows)/parts)))+eachMatchPoint[0][1])*array1Gt[5]
                                returnPoints.append([lon1,lat1,lon2,lat2])
                            # lon2=array2Gt[0]+(float(i*int(floor(float(columns)/parts)))+matchPoints[0][0][2])*array2Gt[1]
                            # lat2=array2Gt[3]+(float(j*int(floor(float(rows)/parts)))+matchPoints[0][0][3])*array2Gt[5]
                            # lon1=array1Gt[0]+(float(i*int(floor(float(refcolumns)/parts)))+matchPoints[0][0][0])*array1Gt[1]
                            # lat1=array1Gt[3]+(float(j*int(floor(float(refrows)/parts)))+matchPoints[0][0][1])*array1Gt[5]
                            # lon2=array1Gt[0]+(float(img1col+5)+matchPoints[0][0][3])*array1Gt[1]
                            # lat2=array1Gt[3]+(float(img1row+5)+matchPoints[0][0][2])*array1Gt[5]
                            # print lon1,lat1,lon2,lat2
                            # print "mystery"
                # print img1,img2
    return returnPoints



def main():
    dir = os.sys.argv[1]
    bands = [1]
    paramFiles=getReferenceFiles.filesinsidefolder(dir,['.json'])
    points=[]
    for eachFile in paramFiles:
        with open(eachFile) as paramFile:
            files=json.load(paramFile)
        for i in range(0,len(files.keys())):
            if os.path.exists(list(files.keys())[i][:-4]+'.csv'):###############################remove orb if sift is used and change in line no 225 too
                os.remove(list(files.keys())[i][:-4]+'.csv')###############################remove orb if sift is used and change in line no 225 too
            with open(list(files.keys())[i][:-4]+'.csv','w+') as pointsFile:###############################remove orb if sift is used and change in line no 225 too
                for eachRefFile in files[list(files.keys())[i]]:
                    raster1=gdal.Open(eachRefFile)
                    raster2=gdal.Open(list(files.keys())[i])
                    points=processPair(raster1,raster2,bands)
                    print (list(files.keys())[i], 'is processed with', eachRefFile, 'with', len(points), 'points')
                    # print points[0]
                    if len(points)>0:
                        for eachPoint in points:
                            pointsFile.write("{},{},{},{}\n".format(eachPoint[0],eachPoint[1],eachPoint[2],eachPoint[3]))
                        # pointsFile.write(points)

if __name__=="__main__":
    main()
