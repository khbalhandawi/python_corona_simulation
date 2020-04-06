# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 02:58:15 2017

@author: Khalil
"""

import os
from PIL import Image
import glob

def create_gif(filenames,d,GIF_folder):
    # filepaths
    
    current_path = os.getcwd() # Working directory of file
    
    # count number of items inside directory
    DIR = './%s' %(GIF_folder)
    n_frames = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    
    im_names = []
    for n in range(1,n_frames - 1, 1):
        fp_in = "%i.png" %(n)
        im_names += [os.path.join(GIF_folder,fp_in)]
        
    fp_out = "%s.gif" %(filenames)
    
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in im_names]
    
    # print(im_names)
    
    # # filepaths
    # fp_in = "%s/*.png" %(GIF_folder)
    # # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

    # print(sorted(glob.glob(fp_in)))
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=d, loop=0)
    

GIF_folder = 'render'
filename = "animation"
duration = 5
create_gif(filename,duration,GIF_folder)