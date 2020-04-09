# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:56:15 2020

@author: Khalil
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as cd


def force_calc(X, social_distance_factor = 1):

    # Potentially avoid neighbors
    i = 0
    
    force = np.zeros_like(X,dtype='float64')
    
    for center in X:
        
        repulsion_force = np.zeros(2)
        for point in X:
    
            to_point = point - center
            dist = np.linalg.norm(to_point)
            if dist > 0:
                repulsion_force -= social_distance_factor * to_point / (dist**3)
    
        # Update population forces
        force[i,:] += repulsion_force
        i += 1
        
    return force

def force_calc_fast(X, social_distance_factor = 1):
    
    force = np.zeros_like(X,dtype='float64')
    
    x = X[:,0]
    y = X[:,1]
    # x[:,0] = 0.0; y[:,1] = 0.0
    
    dist = cd.cdist(X,X)
    dist += np.eye(len(X[:,1]))

    # to_point_x = cd.cdist(x,x, lambda u, v: (u-v).sum())
    # to_point_y = cd.cdist(y,y, lambda u, v: (u-v).sum())

    to_point_x = pairwise_comp(x)
    to_point_y = pairwise_comp(y)

    min_dist = 0.6
    comp = dist <= min_dist
    comp = comp.astype(np.int)

    repulsion_force_x = - social_distance_factor * np.sum( (to_point_x * comp) / (dist**3) , axis = 1)
    repulsion_force_y = - social_distance_factor * np.sum( (to_point_y * comp) / (dist**3) , axis = 1)

    # repulsion_force_x = - social_distance_factor * np.sum( (to_point_x) / (dist**3) , axis = 1)
    # repulsion_force_y = - social_distance_factor * np.sum( (to_point_y) / (dist**3) , axis = 1)

    force[:,0] += repulsion_force_x
    force[:,1] += repulsion_force_y
        
    return force

def pairwise_comp(A): # Using NumPy broadcasting    
    a = np.asarray(A) # Convert to array if not already so
    mask = a - a[:,None]
    return mask

# x = np.arange(5)
# y = np.arange(10,15)
# X = np.hstack((x.reshape((5,1)),y.reshape((5,1))))

X = np.random.random((5,2))

X = np.random.uniform(low = -1, 
                      high = 1,
                      size = (5,2))

dist = np.linalg.norm(X,axis=1)[:,np.newaxis]

# plt.plot(X[:,0],X[:,1],'-xr')

d = pairwise_comp(X[:,0])
d_abs = cd.cdist(X,X)

F1 = force_calc(X)

F2 = force_calc_fast(X)  
         
print(F1 - F2)
# d = pairwise_comp(x[:,0])

# B = d < 0
# B = B.astype(np.int)

# input array
x = np.array([[ 1., 2., 3.], [ 4., 5., 6.], [ 7., 8., 9.]])

# random boolean mask for which values will be changed
mask = np.random.randint(0,2,size=x.shape).astype(np.bool)

# use your mask to replace values in your input array
x[mask] = 0