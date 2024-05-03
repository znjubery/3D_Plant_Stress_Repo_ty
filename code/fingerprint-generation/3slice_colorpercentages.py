# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:05:22 2023

@author: therin young, therin.young@gmail.com
"""

import open3d as o3d
import numpy as np
import os
from glob import *
import cv2
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process point cloud data and calculate color percentages")
    parser.add_argument("csv_file", help="Path to the CSV file containing point cloud data e.g. filenames, IDC scoring, etc")
    parser.add_argument("data_folder", help="Path to the data folder containing 3-Slice point cloud files .pcd")
    parser.add_argument("output_file", help="Output file name for results (e.g., results.csv)")
    args = parser.parse_args()

    # Read the CSV data file
    data = pd.read_csv(args.csv_file, index_col=False)

    #extract plot id from filename and assign to new column (plot id schema = month_day_plot_plotno, e.g. "aug_2_plot_65")
    plotno_lst = []
    for i in range(0,data.shape[0]):
        plotno_lst.append(data['filename'][i].split('.')[0])
        
    #assign to new column
    data['plotno'] = plotno_lst
    
    
    #create dictionary mapping plotno to idc_score
    dictionary = dict(zip(data['plotno'],data['idc_score']))

    # Change directory to the data folder
    os.chdir(args.data_folder)

    # Rest of your code here...
    #from matlab hsv investigating some sample images
    # hue value in 0-1 scale

    # brown 0.0 to 0.17 (0-31) # in the paranthesis conversion to 0-180 scale used in opencv
    # yellow 0.17 to 0.25 (31-45 )
    # green 0.25 to 0.35 (45- 63)

    def count_pixels_in_range(hsv_values, color_range):
        # Count the number of pixels in the HSV array that fall within the specified color range
        min_values = color_range[0]
        max_values = color_range[1]
        #print(f'color range {min_values},{max_values}')
        in_range = np.logical_and.reduce((hsv_values >= min_values, hsv_values <= max_values))
        #print(in_range)
        count = np.count_nonzero(in_range)
        #print(count)
        return count

    def calculate_color_percentages(rgb_points):
        # Convert RGB points to uint8 and reshape to 3D array
        

        rgb_values = rgb_points.astype(np.uint8).reshape((-1, 1, 3))
        #print(np.shape(rgb_points))
        #print(f'shape of rgb_values {np.shape(rgb_values)}')
        #input('press any key')
        
        non_black_pixels = (rgb_values  != [0, 0, 0]).all(axis=-1)
        total_pixels = np.count_nonzero(non_black_pixels)
            
        # Convert RGB values to HSV values
        hsv_values = cv2.cvtColor(rgb_values, cv2.COLOR_RGB2HSV)
        hsv_values_h = hsv_values[:,:,0]

        #print(f'maximum values of hsv {np.max(hsv_values[:,:,0])}')
        
        # Define color ranges for brown, yellow, and green
        brown_range = (0, 31)  # (0-43, 0-255, 0-95) in HSV
        yellow_range = (31, 45)  # (22-60, 60-255, 46-255) in HSV
        green_range = (45,64)  # (61-150, 60-255, 46-255) in HSV
        
        # Calculate the number of pixels that fall within each color range
        brown_pixels = count_pixels_in_range(hsv_values_h[non_black_pixels], brown_range)
        yellow_pixels = count_pixels_in_range(hsv_values_h[non_black_pixels], yellow_range)
        green_pixels = count_pixels_in_range(hsv_values_h[non_black_pixels], green_range)

        
        
        # Calculate the total number of pixels in the image
        total_pixels = hsv_values.shape[0] * hsv_values.shape[1]*3
        
        #print(brown_pixels, yellow_pixels, green_pixels, total_pixels)
        #print(np.shape(rgb_points))
        
        # Calculate the percentage of pixels that fall within each color range
        brown_percentage = (brown_pixels / total_pixels) * 100
        yellow_percentage = (yellow_pixels / total_pixels) * 100
        green_percentage = (green_pixels / total_pixels) * 100
        
        return brown_percentage, yellow_percentage, green_percentage
    
    percgreen = []
    percyellow = []
    percbrown = []
    idc_score =[]
    filename = []
    
    pcdfiles = glob('*.pcd')
    
    for pcd_name in pcdfiles:
        
        pcd=o3d.io.read_point_cloud(pcd_name)
        
        #get plot id dictionary key to map back to IDC score
        fullname = pcd_name.split('_')
        dictkey = fullname[0] + '_' + fullname[1] + '_' + fullname[2] + '_' + fullname[3]
        
        rgb_points = np.array(pcd.colors)
        #print(np.max(rgb_points))
        brown_percentage, yellow_percentage, green_percentage = calculate_color_percentages(rgb_points*255)
    
        #save corresponding points (xyz and rgb) that fall within hsv color range to a list
        filename.append(pcd_name)
        percgreen.append(green_percentage)
        percyellow.append(yellow_percentage)
        percbrown.append(brown_percentage)
        idc_score.append(dictionary[dictkey])
        
    
    results = pd.DataFrame()
    results['filename'] = filename
    results['percgreen'] = percgreen
    results['percyellow'] = percyellow
    results['percbrown'] = percbrown
    results['idc_score'] = idc_score
    
    results.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()

    

    
    
    
