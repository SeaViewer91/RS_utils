from PIL import Image
from PIL.ExifTags import TAGS

def get_GPSInfo(img):
    
    # Get EXIF Info
    info = img._getexif()
    
    # Decode EXIFInfo
    exif = {}
    for tag, value in info.items():
        decoding = TAGS.get(tag)
        exif[decoding] = value
    # Read GPSInfo
    GPSInfo = exif['GPSInfo']
    rawLat = GPSInfo[2]
    rawLon = GPSInfo[4]
    
    # calculate the lat / long
    latDeg = rawLat[0][0] / float(rawLat[0][1])
    latMin = rawLat[1][0] / float(rawLat[1][1])
    latSec = rawLat[2][0] / float(rawLat[2][1])
    
    lonDeg = rawLon[0][0] / float(rawLon[0][1])
    lonMin = rawLon[1][0] / float(rawLon[1][1])
    lonSec = rawLon[2][0] / float(rawLon[2][1])
    
    # Degree/Min/Sec to Degree
    Lat = latDeg + latMin/60 + latSec/3600
    Lon = lonDeg + lonMin/60 + lonSec/3600

    return(Lat, Lon)
