# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 03:01:11 2020

@author: Khalil
"""

import cv2
import numpy as np
import os
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps,vtype):
    frame_array = []
    
    # count number of items inside directory
    DIR = './%s' %(pathIn)
    n_frames = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    
    im_names = []
    for n in range(1,n_frames - 1, 1):
        fp_in = "%i.png" %(n)
        im_names += [os.path.join(pathIn,fp_in)]
    
    for i in range(len(im_names)):
        filename=im_names[i]

        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    
    if vtype == 'avi':
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    elif vtype == 'mp4':
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'H264'), fps, size)
     
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
     
def main():
    vtype = 'mp4'
    indices = [0,7,8,9,10]

    for i in indices:
        pathIn= 'render_%i' %i
        pathOut = 'video_%i.%s' %(i,vtype)
        fps = 60.0
        convert_frames_to_video(pathIn, pathOut, fps, vtype)
 
if __name__=="__main__":
    main()