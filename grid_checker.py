# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:33:13 2020

@author: Khalil
"""

import numpy as np

n_gridpoints = 5

xbounds = [0,1]
ybounds = [0,1]

grid_points=(np.array(range(1,(n_gridpoints-1)**2 + 1) ))
# grid_points = np.reshape(grid_points,(n_gridpoints-1,n_gridpoints-1))

x = np.linspace(xbounds[0], xbounds[1], n_gridpoints)
y = np.linspace(ybounds[0], ybounds[1], n_gridpoints)

# short way
yy,xx=np.meshgrid(x,y, indexing = 'ij')

grid_coords_xlb = xx[:-1,:-1].reshape(((n_gridpoints-1)**2,1))
grid_coords_ylb = yy[:-1,:-1].reshape(((n_gridpoints-1)**2,1))
grid_coords_xub = xx[1:,1:].reshape(((n_gridpoints-1)**2,1))
grid_coords_yub = yy[1:,1:].reshape(((n_gridpoints-1)**2,1))

grid_coords = np.column_stack((grid_coords_xlb,grid_coords_ylb,grid_coords_xub,grid_coords_yub))

pop_size = 600

pos_vector = np.random.random((pop_size,2))
ground_covered = np.zeros((pop_size, (n_gridpoints-1)**2))

# inside_grid = (to_lower < wall_buffer) & (to_lower > -bounce_buffer)
#1D
pos_vector_x = pos_vector[:,0]
pos_vector_y = pos_vector[:,1]

g1 = np.tile(grid_coords[:,0],(pop_size,1)).T
g2 = np.tile(grid_coords[:,1],(pop_size,1)).T
g3 = np.tile(grid_coords[:,2],(pop_size,1)).T
g4 = np.tile(grid_coords[:,3],(pop_size,1)).T

gp = np.tile(grid_points,(pop_size,1))

l_x = (pos_vector_x - g1).T
l_y = (pos_vector_y - g2).T
u_x = (g3 - pos_vector_x).T
u_y = (g4 - pos_vector_y).T

cond = (l_x > 0) & (u_x > 0) & (l_y > 0) & (u_y > 0)
gp[cond]

ground_covered[cond] = 1
print(ground_covered)

#4D
# pos_vector_x = pos_vector[:,0]
# g1 = np.tile(grid_coords[:,0],(pop_size,1)).T

# l_x = (g1 - pos_vector_x).T


# grid_points[grid_coords]

# # long way
# idx = range(0,5)
# x_1 = np.insert(x, idx, 0.0, axis=0)
# idx = range(1,6)
# x_2 = np.insert(x, idx, 0.0, axis=0)

# x_t = (x_2 + x_1)[1:-1]
# x_t = x_t.reshape(-1,2)

# idy = range(0,5)
# y_1 = np.insert(y, idy, 0.0, axis=0)
# idy = range(1,6)
# y_2 = np.insert(y, idy, 0.0, axis=0)

# y_t = (y_2 + y_1)[1:-1]
# y_t = y_t.reshape(-1,2)

# yy,xx=np.meshgrid(x_t,y_t,indexing = 'ij')


# grid_coords_xlb = xx[::2,::2].reshape(((n_gridpoints-1)**2,1))
# grid_coords_ylb = yy[::2,::2].reshape(((n_gridpoints-1)**2,1))
# grid_coords_xub = xx[1::2,1::2].reshape(((n_gridpoints-1)**2,1))
# grid_coords_yub = yy[1::2,1::2].reshape(((n_gridpoints-1)**2,1))

# grid_coords = np.column_stack((grid_coords_xlb,grid_coords_ylb,grid_coords_xub,grid_coords_yub))

# # grid_coords=np.array((xx.ravel(), yy.ravel())).T
