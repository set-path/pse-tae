import os
import json
import rasterio
import numpy as np
import geopandas as gpd
import argparse

def sizes(path,no_data=0):
    tiffs = os.listdir(path)
    pixel = 999999
    for tiff in tiffs:
        if tiff.endswith('.tiff'):
            data = rasterio.open(os.path.join(path,tiff)).read()
            # indices = np.all(data != np.iinfo(np.int16).max, axis=0)
            indices = np.all(data != no_data, axis=0)
            cur = int(np.sum(indices))
            if cur < pixel:
                pixel = cur
    return pixel

def geometry(path,pixel,shp_path):
    geomfeat = {}
    tiffs = os.listdir(path)
    tiff = sorted(tiffs)[0]
    data = rasterio.open(os.path.join(path,tiff)).read()

    gdf = gpd.read_file(shp_path)
    # 单位转化
    gdf_utm = gdf.to_crs({'init': 'epsg:32618'})
    gdf_utm['perimeter'] = gdf_utm.geometry.length
    gdf_utm['area'] = gdf_utm.geometry.area
    ratio = gdf_utm['perimeter'] / gdf_utm['area']
    ratio_2 = pixel / (data.shape[1]*data.shape[2])

    # 4个特征
    geomfeat[0] = [float(gdf_utm['perimeter']), float(
        gdf_utm['area']), float(ratio), float(ratio_2)]
    json.dump(geomfeat, open(os.path.join(path,'META','geomfeat.json'), 'w'), indent=4)

def generate_data(path,size,no_data):
    tiffs = os.listdir(path)
    tiffs = sorted(tiffs)
    max_seq = 20
    time_pixels = []
    for tiff in tiffs:
        if tiff.endswith('.tiff'):
            data = rasterio.open(os.path.join(path,tiff)).read()
            # indices = np.all(data != np.iinfo(np.int16).max, axis=0)
            indices = np.all(data != no_data, axis=0)
            pixels = data[:,indices]
            pixels = pixels[:,:size]
            time_pixels.append(pixels)
    time_pixels = np.stack(time_pixels, axis=0)
    if time_pixels.shape[0] < max_seq:
        add_seq = max_seq - time_pixels.shape[0]
        add_zero = np.zeros((add_seq, 4, size))
        time_pixels = np.concatenate([time_pixels, add_zero], axis=0)
    np.save(os.path.join(path, 'DATA', '0.npy'), time_pixels)

def preprocessing(path,no_data,shp_path):
    size = sizes(path,no_data=no_data)
    generate_data(path,size,no_data=no_data)
    geometry(path,size,shp_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='', type=str)
    parser.add_argument('--noData', default=0, type=int)
    parser.add_argument('--shpPath', default='', type=str)

    config = parser.parse_args()
    config = vars(config)

    os.makedirs(os.path.join(config['path'],'META'),exist_ok=True)
    os.makedirs(os.path.join(config['path'],'DATA'),exist_ok=True)

    preprocessing(path=config['path'],no_data=config['noData'],shp_path=config['shpPath'])