# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from shapely.geometry import Point
import shapely.wkt as wkt
import geopandas as gpd
import pyproj

try:
    from ipyleaflet import *
except:
    print('ipyleaflet library not available. You will have to use static plotting.')


def read_data(input_file, nrows=None):
    """Function to read the Rome / taxi dataset.
    (https://crawdad.org/roma/taxi/20140717/)
    
    Args:
        nrows [int]: Number of rows that are read from the .txt file.
        
    Returns:
        numpy array with [id, taxi-id, date, lat, lon]:
        id [int]: Unique id of observation
        taxi-id [int]: Unique id of taxi
        date [str]: Timestamp in datetime format
        lat [float]: Coordinates in wgs84
        lon [float]: Coordinates in wgs84
    """
    data = pd.read_csv(input_file, nrows=nrows, sep=";", 
                       names=["id", "Taxi-id", "time", "geometry"])
    
    data["time"] = pd.to_datetime(data["time"])
    data["time"] = data["time"].dt.tz_convert(None)  # to utc, remove tz
    # data["time"] = data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    geometry = data["geometry"].apply(wkt.loads)
    data.drop(["geometry", "id", "Taxi-id"], axis=1, inplace=True)
    
    data["lat"] = [geom.x for geom in geometry]
    data["lon"] = [geom.y for geom in geometry]
    
    data = data[["lon", "lat", "time"]]
    return data.values.tolist()


def plot_nb_dists(X, nearest_neighbor, metric='euclidean', ylim=None):
    """ Plots distance sorted by `nearest_neighbor`th

    Args:
        X (list of lists): list with data tuples
        nearest_neighbor (int): nr of nearest neighbor to plot
        metric (string): name of scipy metric function to use
    """
    tree = KDTree(X, leaf_size=2) 

    if not isinstance(nearest_neighbor, list):
        nearest_neighbor = [nearest_neighbor]
    max_nn = max(nearest_neighbor)

    dist, _ = tree.query(X, k=max_nn + 1)

    plt.figure(figsize=(16, 8))
    for nnb in nearest_neighbor:
        col = dist[:, nnb]
        col.sort()
        plt.plot(col, label="{}th nearest neighbor".format(nnb))
    # plt.ylim(0, min(250, max(dist[:, max_nn])))
    plt.ylabel("Distance to k nearest neighbor")
    plt.xlabel("Points sorted according to distance of k nearest neighbor")
    plt.ylim(0, ylim)
    plt.grid()
    plt.legend()
    plt.show()


def plot_cluster(data, labels, core_samples_indices, proj_wgs84, proj_target, background=True, linestyle="solid"):
    """ Function derived from scipy dbscan example
    http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py.
    """
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_samples_indices] = True

    plt.figure(figsize=(16, 11.537))
    if background:
        xmin_wgs = 12.2080
        xmax_wgs = 12.6645
        ymin_wgs = 41.7345
        ymax_wgs = 41.9964
        transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_target, always_xy=True)
        
        xmin, ymin = transformer.transform(xmin_wgs, ymin_wgs)
        xmax, ymax = transformer.transform(xmax_wgs, ymax_wgs)
    
        img = mpimg.imread("background_rome.png")
        plt.imshow(img, extent=[xmin, xmax, ymin, ymax], zorder=0)
            
    X = np.array(data)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        col = tuple(col)
        if k == -1:
            # Gray used for noise.
            col = 'silver'
            markeredgecolor='k'
        else:
            markeredgecolor='k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot([d[0] for d in data], [d[1] for d in data], zorder=1, linestyle=linestyle)
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14, zorder=3, linestyle=linestyle)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor=markeredgecolor, markersize=6, zorder=2, linestyle=linestyle)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def plot_cluster_interactive(data, labels, core_samples_indices, proj_wgs84, proj_target):
    """Plots the output of the previously applied method onto an interactive map."""
    m = Map(center=(41.89, 12.46), zoom=11, basemap=basemaps.OpenStreetMap.Mapnik)    
    transformer = pyproj.Transformer.from_crs(proj_target, proj_wgs84)

    unique_labels = list(set(labels))
    colors = {ul: f"hsla({str(np.random.choice(range(360)))}, 100%, 50%, 1)" for ul in unique_labels}
    colors[-1] = 'rgba(60, 60, 60, 1)'
    for i in range(len(data)):
        pt = data[i]
        color = colors[labels[i]]
        mark = Circle(location=transformer.transform(*pt[0:2]), radius=100, fill_opacity=1.0, 
                    color=color, stroke=False)    
        m += mark
        
    for i in core_samples_indices:
        color = colors[labels[i]]
        mark = Circle(location=transformer.transform(*data[i, 0:2]), radius=500, fill_opacity=0.5, 
                      color=color, stroke=False)
        m += mark
    
    return m
    

def export_to_shp(data, labels, export_layer_name, crs):
    """Function that exports geodata as shapefile."""
    data = np.array(data)
    geometry = [Point(row[0],row[1]) for row in data]
    gdf = gpd.GeoDataFrame(data[:,0], columns=["t_rel"], crs=crs, geometry=geometry)
    gdf["label"] = labels
    gdf.to_file(driver='ESRI Shapefile', filename=export_layer_name)
