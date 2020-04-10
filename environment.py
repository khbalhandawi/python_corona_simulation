'''
file that contains all functions to define destinations in the 
environment of the simulated world.
'''

import numpy as np
import matplotlib.patches as patches #for boundaries

def build_hospital(xmin, xmax, ymin, ymax, ax, bound_color, addcross=True):
    '''builds hospital
    
    Defines hospital and returns wall coordinates for 
    the hospital, as well as coordinates for a red cross
    above it
    
    Keyword arguments
    -----------------
    xmin : int or float
        lower boundary on the x axis
        
    xmax : int or float
        upper boundary on the x axis
        
    ymin : int or float
        lower boundary on the y axis
        
    ymax : int or float 
        upper boundary on the y axis
        
    plt : matplotlib.pyplot object
        the plot object to which to append the hospital drawing
        if None, coordinates are returned
        
    Returns
    -------
    None
    '''

    #plot walls
    lower_corner = (xmin,ymin)
    width = xmax - xmin
    height = ymax - ymin

    # Draw boundary of destination
    rect = patches.Rectangle(lower_corner, width, height, linewidth=1, edgecolor=bound_color, facecolor='none', fill='None', hatch=None)
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    #plot red cross
    if addcross:
        xmiddle = xmin + ((xmax - xmin) / 2)
        height = np.min([0.3, (ymax - ymin) / 5])
        ax.plot([xmiddle, xmiddle], [ymax, ymax + height], color='red',
                 linewidth = 3)
        ax.plot([xmiddle - (height / 2), xmiddle + (height / 2)],
                 [ymax + (height / 2), ymax + (height / 2)], color='red',
                 linewidth = 3)