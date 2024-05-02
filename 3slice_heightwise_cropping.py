# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:10:19 2023

@author: therin young, therin.young@gmail.com
"""


import open3d as o3d
import numpy as np
from glob import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
import scipy.io as sio



def main():
    parser = argparse.ArgumentParser(description="Perform height-wise slicing of point clouds")
    parser.add_argument("pcd_path", help="Folder path to pcd files")
    parser.add_argument("output_path", help="folder path to store results")


    args = parser.parse_args()


    def points2array(pc):#Convert xyz points into numpy array
        xyz_points = np.array(pc.points)
        return(xyz_points)
    
    def rgb2array(pc): #Convert rgb values into numpy array
        rgb_points = np.array(pc.colors) 
        return rgb_points
    
    def pc_crop(arg1,arg2,arg3,arg4):
        '''
        arg1:xyz coordinates
        arg2:rgb coordinates
        arg3:min bound xyz
        arg4:max bound xyz
        '''
        #cropPoints=arg1[(arg1[:,0] < arg4[0]) & (arg1[:,0] > arg3[0]) & (arg1[:,1] > arg3[1]) & (arg1[:,1] < arg4[1])]   
        #cropColors=arg2[(arg1[:,0] < arg4[0]) & (arg1[:,0] > arg3[0]) & (arg1[:,1] > arg3[1]) & (arg1[:,1] < arg4[1])]
        cropPoints=arg1[(arg1[:,2] < arg4[2]) & (arg1[:,2] > arg3[2])]   
        cropColors=arg2[(arg1[:,2] < arg4[2]) & (arg1[:,2] > arg3[2])]
    
        return(cropPoints,cropColors)
    
    os.chdir(args.pcd_path)
    
    output_dir = args.output_path
    
    missing_slice = [] 
    
    pc_files = glob('*.pcd')
  
    for file in pc_files:
        try:
            name,ext = file.split('.')
            pc = o3d.io.read_point_cloud(file)
  
            ############################################################################
            # Plant points processing
            ############################################################################  

            xyz = points2array(pc) 
            rgb = rgb2array(pc)

            #slice canopy into desired number of equal partitions
            no_slices = 3.0

            #calculate min and max bound of point cloud
            min_bound = pc.get_min_bound()
            max_bound = pc.get_max_bound()

            bound_diff = max_bound - min_bound
            slice_no = bound_diff[2]/no_slices

            count = 0
            count2 = no_slices - 1

   
            for i in range(0,int(no_slices)):

                #reset min_bound and max_bound for next slice
                min_bound = pc.get_min_bound()
                max_bound = pc.get_max_bound()

                #print(count)
                #print(count2)

                #create variables for the min bound of the slice and the max bound of the slice
                bound_min = min_bound
                bound_max = max_bound
                #print(bound_min,bound_max)

                #use the 'slice_no' varialbe to offset the min and max z coordinate for each slice
                bound_min[2] = bound_min[2] + (slice_no*float(count))
                bound_max[2] = bound_max[2] - (slice_no*float(count2))
                #print(bound_min,bound_max)



                #use min and max bound coordinates to crop the slice
                crop_xyz,crop_rgb = pc_crop(xyz,rgb,bound_min,bound_max)

                #assign cropped coordinates to new variables for upcoming tasks
                xyz_points = crop_xyz
                rgb_points = crop_rgb


                #save cropped points to pcd file
                crop_pc = o3d.geometry.PointCloud()
                crop_pc.points = o3d.utility.Vector3dVector(xyz_points)
                crop_pc.colors = o3d.utility.Vector3dVector(rgb_points)
                o3d.io.write_point_cloud('%s/%s_slice_%s.pcd' % (output_dir,name,count),crop_pc)
                print("sliced %s_%s" % (name,count))




                count = i + 1
                count2 = count2 - 1

                #print plots that were not completely sliced
                if crop_xyz.size == 0:
                    missing_slice.append(name)
                    print(name)
   
        except:
            continue

    
    
    
    

