import numpy as np
import cv2
from osgeo import ogr,gdal,osr
import math

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
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Source: https://gis.stackexchange.com/a/56589/15183
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km
def Line_Shp_File(points,CRS2,CRS1,line_shp_location,point_shp_location):
    #print(CRS2)
    #print(CRS2)

    outputEPSG = 4326
    gcs_srs = osr.SpatialReference()
    gcs_srs.ImportFromEPSG(4326)


    coordTransform2 = osr.CoordinateTransformation(CRS2, gcs_srs) #(2=>out)
    coordTransform1 = osr.CoordinateTransformation(CRS1, gcs_srs)
    #transform.TransformPoint(raster1Bounds[0],raster1Bounds[1])
    points_list=[]
    #for eachPoint in points:
    #   points_list.append([[coordTransform1.TransformPoint(eachPoint[0],eachPoint[1])],[coordTransform2.TransformPoint(eachPoint[2],eachPoint[3])]])
    driver=ogr.GetDriverByName("ESRI Shapefile")
    #driver1=ogr.GetDriverByName("ESRI Shapefile")
    ds=driver.CreateDataSource(line_shp_location)
    #dsp=driver1.CreateDataSource(point_shp_location)
    layer=ds.CreateLayer('MATCHING FEATURES', gcs_srs, ogr.wkbLineString)
    #layerp=dsp.CreateLayer('MATCH POINTS',CRS1,ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('length', ogr.OFTInteger))
    #layer.CreateField(ogr.FieldDefn('length',))
    #layerp.CreateField(ogr.FieldDefn('id', ogr.OFTString))
    featureDefn = layer.GetLayerDefn()

    count=1
    #----------------------- CREATING A LINE -----------
    for eachPoint in points:
        #points_list.append([[coordTransform1.TransformPoint(eachPoint[0],eachPoint[1])],[coordTransform2.TransformPoint(eachPoint[2],eachPoint[3])]])
        line=ogr.Geometry(ogr.wkbLineString)
        #point_1=ogr.Geometry(ogr.wkbPoint)
        #point_2=ogr.Geometry(ogr.wkbPoint)
        point1=coordTransform1.TransformPoint(eachPoint[0],eachPoint[1])

        #point_1.AddPoint(eachPoint[0],eachPoint[1])

        point2=coordTransform2.TransformPoint(eachPoint[2],eachPoint[3])

        #point_2.AddPoint(eachPoint[2],eachPoint[3])

        line.AddPoint(point1[0],point1[1])
        line.AddPoint(point2[0],point2[1])
        length = haversine(point1[0],point1[1],point2[0],point2[1])*1000
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetField("length",length)

        #outFeaturep = ogr.Feature(layerp.GetLayerDefn())

        outFeature.SetGeometry(line)
        #----------------------------------------POINTS-----------
        # outFeaturep.SetGeometry(point_1)
        # outFeaturep.SetField('id',count)
        # layerp.CreateFeature(outFeaturep)
        # #
        # outFeaturep.SetGeometry(point_2)
        # outFeaturep.SetField('id',str(count)+'M')
        # layerp.CreateFeature(outFeaturep)
        #

        #------------------------------------------------
        outFeature.SetField('id',count)
        layer.CreateFeature(outFeature)
        count=count+1
    print('CREATED SHAPE FILE')
