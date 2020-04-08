# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 03:01:11 2020

@author: Khalil
"""

# import cv2
# import os

# image_folder = 'render_1'
# video_name = 'video.avi'

# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()

import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
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
    
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
     
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
     
def main():
    pathIn= 'render_2'
    pathOut = 'video.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()