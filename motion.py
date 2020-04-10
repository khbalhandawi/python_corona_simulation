'''
file that contains all function related to population mobility
and related computations
'''

import numpy as np
import scipy.spatial.distance as cd
from random import gauss

def update_positions(population,Config):
    '''update positions of all people

    Uses heading and speed to update all positions for
    the next time step

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information
    '''

    #update positions
    #x
    population[:,1] += population[:,3] * Config.dt
    #y
    population[:,2] += population [:,4] * Config.dt

    return population

def update_velocities(population,Config):
    '''update positions of all people

    Uses heading and speed to update all positions for
    the next time step

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information
    '''

    max_speed = Config.max_speed

    for i in range(2):
        # Apply force
        population[:,i + 3] += population[:,i + 15] * Config.dt

        # Limit speed
        speed = np.linalg.norm(population[:,3:5],axis = 1)
        population[speed > max_speed,i + 3] *= max_speed / speed[speed > max_speed]

        # Limit force
        population[:,i + 15] = 0.0

        # population[speed < min_speed,v_i[i]] *= min_speed / speed[speed < min_speed]

    return population

def update_wall_forces(population, xbounds, ybounds):

    '''checks which people are about to go out of bounds and corrects

    Function that calculates wall repulsion forces on individuals that are about to 
    go outside of the world boundaries.
    
    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    xbounds, ybounds : list or tuple
        contains the lower and upper bounds of the world [min, max]
    '''
    dl_bound = [ xbounds[:,0], ybounds[:,0] ]
    ur_bound = [ xbounds[:,1], ybounds[:,1] ]
    
    # Avoid walls
    wall_buffer = 1
    wall_buffer = 0.01
    wall_force = np.zeros_like(population[:,0:2])
    
    for i in range(2):
        to_lower = population[:,i + 1] - dl_bound[i]
        # to_lower[to_lower <= 0.0] = -wall_buffer

        to_upper = ur_bound[i] - population[:,i + 1]
        # to_upper[to_upper <= 0.0] = -wall_buffer

        # Bounce
        bounce_lo = (to_lower < 0.0) #& (population[:,i + 3] < 0)
        population[:,i + 3][bounce_lo] = abs(population[:,i + 3][bounce_lo])
        population[:,i + 1][bounce_lo] = dl_bound[i][bounce_lo] + 0.005

        bounce_ur = (to_upper < 0.0) #& (population[:,i + 3] > 0)
        population[:,i + 3][bounce_ur] = -abs(population[:,i + 3][bounce_ur])
        population[:,i + 1][bounce_ur] = ur_bound[i][bounce_ur] - 0.005

        # Repelling force
        wall_force[:,i] += np.maximum((-1 / wall_buffer**1 + 1 / to_lower**1), 0)
        wall_force[:,i] -= np.maximum((-1 / wall_buffer**1 + 1 / to_upper**1), 0)

        # Repelling force
        # wall_force[:,i] += np.maximum((1 / to_lower), 0)
        # wall_force[:,i] -= np.maximum((1 / to_upper), 0)

        # wall_force[:,i][to_lower < wall_buffer] += abs(1 / ((to_lower[to_lower < wall_buffer])**2))
        # wall_force[:,i][to_upper < wall_buffer] -= abs(1 / ((to_upper[to_upper < wall_buffer])**2))

        population[:,i + 15] += wall_force[:,i]

    # Update forces
    return population


def pairwise_comp(A): # Using NumPy broadcasting    
    a = np.asarray(A) # Convert to array if not already so
    mask = a - a[:,None]
    return mask

def update_repulsive_forces(population, social_distance_factor):

    '''calculated repulsive forces between individuals

    Function that calculates repulsion forces between individuals during social distancing.
    
    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    xbounds, ybounds : list or tuple
        contains the lower and upper bounds of the world [min, max]
    '''

    x = population[:,1]
    y = population[:,2]
    
    dist = cd.cdist(population[:,1:3],population[:,1:3])
    dist += np.eye(len(population[:,1]))

    to_point_x = pairwise_comp(x)
    to_point_y = pairwise_comp(y)

    repulsion_force_x = - social_distance_factor * np.sum( (to_point_x) / (dist**5.0) , axis = 1)
    repulsion_force_y = - social_distance_factor * np.sum( (to_point_y) / (dist**5.0) , axis = 1)

    population[:,15] += repulsion_force_x
    population[:,16] += repulsion_force_y

    # Update forces
    return population

def update_gravity_forces(population, wander_step_size, time, gravity_strength, 
                          wander_step_duration, last_step_change):

    # Gravity
    if wander_step_size != 0:
        if (time - last_step_change) > wander_step_duration:

            vect_un = np.random.uniform(low = -1, 
                                        high = 1,
                                        size = (len(population[:,1]),2))

            vect = vect_un / np.linalg.norm(vect_un,axis=1)[:,np.newaxis]

            gravity_well = population[:,1:3] + wander_step_size * vect
            last_step_change = time

            to_well = (gravity_well - population[:,1:3])
            dist = np.linalg.norm(to_well,axis=1)
            
            population[:,15][dist != 0] += gravity_strength * to_well[:,0][dist != 0] / (dist[dist != 0]**3)
            population[:,16][dist != 0] += gravity_strength * to_well[:,1][dist != 0] / (dist[dist != 0]**3)

    return population, last_step_change

def get_motion_parameters(xmin, ymin, xmax, ymax):
    '''gets destination center and wander ranges

    Function that returns geometric parameters of the destination
    that the population members have set.

    Keyword arguments:
    ------------------
        xmin, ymin, xmax, ymax : int or float
        lower and upper bounds of the destination area set.

    '''

    x_center = xmin + ((xmax - xmin) / 2)
    y_center = ymin + ((ymax - ymin) / 2)

    x_wander = (xmax - xmin) / 2
    y_wander = (ymax - ymin) / 2

    return x_center, y_center, x_wander, y_wander
