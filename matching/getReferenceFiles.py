from osgeo import gdal,ogr
from os import sys,path,walk
import numpy
def filesinsidefolder(myPath,form):
    fileNames=[]
    for eachForm in form:
        for dirpath, dirnames, filenames in walk(myPath):
            for filename in [f for f in filenames if eachForm in f]:
                if '.aux' in filename:
                    pass
                else:
                    fileNames.append(path.join(dirpath, filename))
    return fileNames
def createPolygon(points):
    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(points[0][0],points[0][1])
    ring.AddPoint(points[1][0],points[0][1])
    ring.AddPoint(points[1][0],points[1][1])
    ring.AddPoint(points[0][0],points[1][1])
    ring.AddPoint(points[0][0],points[0][1])
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly
def main():
    inputDirOrFile=sys.argv[1]
    refDir=sys.argv[2]
    forms=['.tif','.img']
    for eachForm in forms:
        if eachForm not in inputDirOrFile:
            fNames=filesinsidefolder(inputDirOrFile,forms)
            geomList=[]
            for eachFile in fNames:
                raster=gdal.Open(eachFile)
                gt=raster.GetGeoTransform()
                columns=raster.RasterXSize
                rows=raster.RasterYSize
                upperLeftPoint=(gt[0],gt[3])
                bottomRightPoint=(gt[0]+columns*gt[1],gt[3]+rows*gt[5])
                geom=createPolygon((upperLeftPoint,bottomRightPoint))
                geomList.append([geom,eachFile])
            indexLayerSource = driver.Open(indexFile, 0) # 0 means read-only. 1 means writeable.
            indexLayer=indexLayerSource.GetLayer()
            for eachGeom in geomList:
                for eachFeature in indexLayer:
                    featureGeom=eachFeature.GetGeometryRef()
                    if featureGeom.Intersects(eachGeom[0]):
                        intersectionGeom=featureGeom.Instersection(eachgeom[0])
                        if intersectionGeom.GetArea()/
                        eachGeom[0]=eachGeom[0].Difference(intersectionGeom)

if __name__=='__main__':
    main()
