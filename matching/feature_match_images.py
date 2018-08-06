import numpy as np
import cv2,os
from matplotlib import pyplot as plt
from osgeo import gdal,ogr,osr
from math import ceil,floor
import json
from img_proc_funcs import converto8bit,imageMatch,Line_Shp_File


class FM_Image:

    methods=['SIFT']

    def __init__(self, dic_im1,dic_im2,FMmethod):

        self.im1_file = dic_im1["file"]
        self.im1_band = dic_im1["band"]
        self.im2_file = dic_im2["file"]
        self.im2_band = dic_im2["band"]
        self.method = FMmethod

        prelim_check=1
        if not os.access(self.im1_file, os.R_OK):
            print('UNABLE TO ACCESS TEST IMAGE')
            prelim_check=0
        if not os.access(self.im2_file, os.R_OK):
            print('UNABLE TO ACCESS REFERENCE IMAGE')
            prelim_check=0
        if not (self.method in self.methods):
            print('FEATURE MATCH METHOD NOT CONFIGURED')
            prelim_check=0
        self.raster1=gdal.Open(self.im1_file)
        self.raster2=gdal.Open(self.im2_file)
        bands_1=self.raster1.RasterCount
        bands_2=self.raster2.RasterCount
        if not((self.im1_band<=bands_1) or (self.im2_band<=bands_2)):
            print('BANDS NOT EQUAL')
            prelim_check=0
        if(prelim_check==0):
            raise Exception('FMINIT_ERROR: Exception Caught while Accessing the images passed or the method of Feature Match')

    def MatchFeatures(self):

        #-------- GET THE GEO TRANSFORMS -----------
        raster1Gt=self.raster1.GetGeoTransform() # (UL_X, X_PIXEL_SIZE, X_SKEW, UL_Y, Y_SKEW, -Y_PIXEL_SIZE)
        raster2Gt=self.raster2.GetGeoTransform()

        #-------- CALCULATE BOUNDS --------------
        #ULX,ULY,BRX,BRY
        raster1Bounds=(raster1Gt[0],raster1Gt[3],raster1Gt[0]+self.raster1.RasterXSize*raster1Gt[1],raster1Gt[3]+self.raster1.RasterYSize*raster1Gt[5])

        raster2Bounds=(raster2Gt[0],raster2Gt[3],raster2Gt[0]+self.raster2.RasterXSize*raster2Gt[1],raster2Gt[3]+self.raster2.RasterYSize*raster2Gt[5])
        #----------------------------- REPROJECTING INTERSECTION STARTS ----------------

        #-- SET A REFERENCE TO CONVERT PROJECTION CO-ORDS for intersection
        raster1CRS=osr.SpatialReference()
        raster2CRS=osr.SpatialReference()
        raster1CRS.ImportFromWkt(self.raster1.GetProjectionRef())
        raster2CRS.ImportFromWkt(self.raster2.GetProjectionRef())
        #USING RASTER2 CRS AS Referecnce
        transform = osr.CoordinateTransformation(raster1CRS,raster2CRS) #transformation from 1 => 2
        rev_transform=osr.CoordinateTransformation(raster2CRS,raster1CRS)# reverse transformation from 2=>1

        raster1Bounds_reproj=(transform.TransformPoint(raster1Bounds[0],raster1Bounds[1]),transform.TransformPoint(raster1Bounds[2],raster1Bounds[3]))
        #FINDING INTERSECTION IN REFERENCE CRS
        intersectionBounds=(max(raster1Bounds_reproj[0][0],raster2Bounds[0]),min(raster1Bounds_reproj[0][1],raster2Bounds[1]),min(raster1Bounds_reproj[1][0],raster2Bounds[2]),max(raster1Bounds_reproj[1][1],raster2Bounds[3]))
        #--- CONVERTING intersectionBounds into 1st image CRS for bound calculation
        intersectionBounds_reverse_reproj=(transform.TransformPoint(intersectionBounds[0],intersectionBounds[1]),transform.TransformPoint(intersectionBounds[2],intersectionBounds[3]))

        #----------------------------- REPROJETING INTERSECTION ENDS ---------------------

        #---- FIND THE ARRAY BOUNDS ---------
        array1Bounds=(int(ceil((intersectionBounds_reverse_reproj[0][0]-raster1Bounds[0])/raster1Gt[1])),int(ceil((intersectionBounds_reverse_reproj[0][1]-raster1Bounds[1])/raster1Gt[5])),int(floor((intersectionBounds_reverse_reproj[1][0]-raster1Bounds[0])/raster1Gt[1])),int(floor((intersectionBounds_reverse_reproj[1][1]-raster1Bounds[1])/raster1Gt[5])))
        array2Bounds=(int(ceil((intersectionBounds[0]-raster2Bounds[0])/raster2Gt[1])),int(ceil((intersectionBounds[1]-raster2Bounds[1])/raster2Gt[5])),int(floor((intersectionBounds[2]-raster2Bounds[0])/raster2Gt[1])),int(floor((intersectionBounds[3]-raster2Bounds[1])/raster2Gt[5])))


        #----- intersection geotransforms
        array1Gt=(raster1Gt[0]+raster1Gt[1]*array1Bounds[0],raster1Gt[1],raster1Gt[2],raster1Gt[3]+raster1Gt[5]*array1Bounds[1],raster1Gt[4],raster1Gt[5])
        array2Gt=(raster2Gt[0]+raster2Gt[1]*array2Bounds[0],raster2Gt[1],raster2Gt[2],raster2Gt[3]+raster2Gt[5]*array2Bounds[1],raster2Gt[4],raster2Gt[5])


        #-------------- GRIDDING STARTS--------------
        refcolumns=array2Bounds[2]-array2Bounds[0]
        refrows=array2Bounds[3]-array2Bounds[1]
        columns=array1Bounds[2]-array1Bounds[0]
        rows=array1Bounds[3]-array1Bounds[1]
        returnPoints=[]
        band1=self.raster1.GetRasterBand(self.im1_band)
        band2=self.raster2.GetRasterBand(self.im2_band)

        Grid_Pixel_Parameter=512
        parts=max(rows,columns)/Grid_Pixel_Parameter
        for i in range(0,int(parts)):
            if i<parts-1:
                p=int(floor(float(columns)/parts))
                num_cols_1=int(floor(float(columns)/parts))
                num_cols_2=int(floor(float(refcolumns)/parts))
            else:
                p=columns-i*int(floor(float(columns)/parts))
                num_cols_1=columns-i*int(floor(float(columns)/parts))
                num_cols_2=refcolumns-i*int(floor(float(refcolumns)/parts))
            for j in range(0,int(parts)):
                if j<parts-1:
                    q=int(floor(float(rows)/parts))
                    num_rows_1=int(floor(float(rows)/parts))
                    num_rows_2=int(floor(float(refrows)/parts))
                else:
                    q=rows-j*int(floor(float(rows)/parts))
                    num_rows_1=rows-j*int(floor(float(rows)/parts))
                    num_rows_2=refrows-j*int(floor(float(refrows)/parts))
                if (i%2==0 and j%2==0):
                #if i==j:
                    print ('processing part',i,j,parts)

                    scaleDiff=[array2Gt[1]/array1Gt[1],array2Gt[5]/array1Gt[5]]
                    img2=band2.ReadAsArray(array2Bounds[0]+i*int(floor(float(refcolumns)/parts)),array2Bounds[1]+j*int(floor(float(refrows)/parts)),num_cols_2,num_rows_2)
                    img1=band1.ReadAsArray(array1Bounds[0]+i*int(floor(float(columns)/parts)),array1Bounds[1]+j*int(floor(float(rows)/parts)),num_cols_1,num_rows_1)

                    conv=converto8bit(img1,img2)
                    #conv=(img1,img2)
                    if conv==0:
                        continue
                    else:
                        img1,img2=conv                          #--------------------------------for testing else remove if clause and run in main thread
                        matchPoints,logPoints=imageMatch(img1,img2)###########################################change this function to use orb or sift
                        if logPoints==0:
                            print ('Error in descriptor calculation in',i,j,'part')
                        if len(matchPoints)>0:
                            for eachMatchPoint in matchPoints:
                                lon2=array2Gt[0]+(float(i*int(floor(float(refcolumns)/parts)))+eachMatchPoint[0][2])*array2Gt[1]
                                lat2=array2Gt[3]+(float(j*int(floor(float(refrows)/parts)))+eachMatchPoint[0][3])*array2Gt[5]
                                lon1=array1Gt[0]+(float(i*int(floor(float(columns)/parts)))+eachMatchPoint[0][0])*array1Gt[1]
                                lat1=array1Gt[3]+(float(j*int(floor(float(rows)/parts)))+eachMatchPoint[0][1])*array1Gt[5]
                                returnPoints.append([lon1,lat1,lon2,lat2])

        return returnPoints
def main():


    file_1="/home/champrakri/183776011/BAND1.tif"
    file_2="/home/champrakri/183776011/BAND2.tif"
    output_file="/home/champrakri/183776011/band1-2.csv"
    line_shp_location="/home/champrakri/183776011/band1-2_line.shp"
    point_shp_location="/home/champrakri/183776011/band1-2_point.shp"

    dic_im1={"file":file_1,"band":1}
    dic_im2={"file":file_2,"band":1}
    fm1=FM_Image(dic_im1,dic_im2,'SIFT')
    points=fm1.MatchFeatures()

    raster1CRS=osr.SpatialReference()
    raster1CRS.ImportFromWkt(fm1.raster1.GetProjectionRef())
    raster2CRS=osr.SpatialReference()
    raster2CRS.ImportFromWkt(fm1.raster2.GetProjectionRef())
    Line_Shp_File(points,raster1CRS,raster2CRS,line_shp_location,point_shp_location)



    print (file_1, 'is processed with', file_2, 'with', len(points), 'points')
    with open(output_file,'w+') as pointsFile:
        if len(points)>0:
            for eachPoint in points:
                pointsFile.write("{},{},{},{}\n".format(eachPoint[0],eachPoint[1],eachPoint[2],eachPoint[3]))
if __name__=="__main__":
    main()
