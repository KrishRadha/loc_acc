from osgeo import gdal,ogr
import os,numpy
from os import walk,path
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
def createShapeFile(dir):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dstDs = drv.CreateDataSource(dir)
    dstLayer = dstDs.CreateLayer(dir, srs = None ,geom_type=ogr.wkbPolygon )
    fieldName = ogr.FieldDefn("fileName", ogr.OFTString)
    fieldName.SetWidth(250)
    dstLayer.CreateField(fieldName)
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
def addFeature(fileName,geom,layer):
    featureDefn = layer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(geom)
    outFeature.SetField("fileName", fileName)
    layer.CreateFeature(outFeature)
    outFeature = None
def main(dir):
    fNames=filesinsidefolder(dir,['.img','.tif'])
    indexFile=os.path.join(dir,'index.shp')
    if not os.path.exists(indexFile):
        createShapeFile(indexFile)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    indexLayerSource = driver.Open(indexFile, 1) # 0 means read-only. 1 means writeable.
    indexLayer=indexLayerSource.GetLayer()
    for eachFile in fNames:
        raster=gdal.Open(eachFile)
        gt=raster.GetGeoTransform()
        columns=raster.RasterXSize
        rows=raster.RasterYSize
        upperLeftPoint=(gt[0],gt[3])
        bottomRightPoint=(gt[0]+columns*gt[1],gt[3]+rows*gt[5])
        geom=createPolygon((upperLeftPoint,bottomRightPoint))
        addFeature(eachFile,geom,indexLayer)
if __name__=='__main__':
    main(os.sys.argv[1]) ############Input directory
