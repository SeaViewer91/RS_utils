import os
import shutil
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def center_crop(img, crop_size):
    h, w = img.shape[:2]
    start_y = max((h - crop_size) // 2, 0)
    start_x = max((w - crop_size) // 2, 0)
    return img[start_y:start_y + crop_size, start_x:start_x + crop_size]

def freq_ratio(gray, radius_ratio=0.1):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    crow, ccol = h // 2, w // 2
    radius = int(min(h, w) * radius_ratio)
    Y, X = np.ogrid[:h, :w]
    mask_low = (X - ccol)**2 + (Y - crow)**2 <= radius**2
    low_energy = mag[mask_low].sum()
    high_energy = mag[~mask_low].sum()
    return high_energy / (low_energy + 1e-6)

def get_exif_lat_lon(img_path):
    img = Image.open(img_path)
    exif = img._getexif()
    if not exif:
        return None, None
    gps_info = {}
    for tag, value in exif.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            for t, v in value.items():
                sub = GPSTAGS.get(t, t)
                gps_info[sub] = v
    def to_deg(val):
        # val can be list/tuple of IFDRational
        vals = [float(v) for v in val]
        d, m, s = vals
        return d + m / 60 + s / 3600
    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        lat = to_deg(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef', 'N') != 'N':
            lat = -lat
        lon = to_deg(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef', 'E') != 'E':
            lon = -lon
        return lat, lon
    return None, None

def process_directory(input_dir, crop_size=500, threshold=1.1):
    blur_dir = os.path.join(input_dir, "Blur")
    os.makedirs(blur_dir, exist_ok=True)

    records = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        crop = center_crop(img, crop_size)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = freq_ratio(gray)

        if score < threshold:
            dst_path = os.path.join(blur_dir, fname)
            shutil.move(src_path, dst_path)
            lat, lon = get_exif_lat_lon(dst_path)
            records.append({
                "filename": fname,
                "latitude": lat,
                "longitude": lon
            })

    df = pd.DataFrame(records, columns=["filename", "latitude", "longitude"])
    csv_path = os.path.join(input_dir, "blur_images.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    if not df.empty and df["latitude"].notnull().all() and df["longitude"].notnull().all():
        geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        shp_path = os.path.join(input_dir, "blur_images.shp")
        gdf.to_file(shp_path)
        print(f"SHP saved to: {shp_path}")
    else:
        print("No valid GPS EXIF data found; skipping SHP creation.")

    print(f"Processed {len(records)} blurred images.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect blur via freq. ratio and organize results.")
    parser.add_argument("input_dir", help="Path to image directory")
    parser.add_argument("--crop", type=int, default=500, help="Center crop size (px)")
    parser.add_argument("--thresh", type=float, default=0.5, help="Frequency ratio threshold")
    args, _ = parser.parse_known_args()
    process_directory(args.input_dir, crop_size=args.crop, threshold=args.thresh)
