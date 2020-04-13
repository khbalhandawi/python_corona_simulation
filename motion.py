'''
file that contains all function related to population mobility
and related computations
'''

import numpy as np
import scipy.spatial.distance as cd
from utils import pairwise_dist, pairwise_comp
from random import gauss

def update_positions(population, dt = 0.01):
    '''update positions of all people

    Uses heading and speed to update all positions for
    the next time step

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information
        
     dt : float 
        Time increment used for incrementing velocity due to forces
    '''

    #update positions
    #x
    population[:,1] += population[:,3] * dt
    #y
    population[:,2] += population[:,4] * dt

    return population

def update_velocities(population, max_speed = 0.3, dt = 0.01):
    '''update positions of all people

    Uses heading and speed to update all positions for
    the next time step

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    max_speed : float
        Maximum speed cap for individuals

     dt : float 
        Time increment used for incrementing velocity due to forces
    '''

    for i in range(2):
        # Apply force
        population[:,i + 3] += population[:,i + 15] * dt

        # Limit speed
        speed = np.linalg.norm(population[:,3:5],axis = 1)
        population[speed > max_speed,i + 3] *= max_speed / speed[speed > max_speed]

        # Limit force
        population[:,i + 15] = 0.0

    return population

def update_wall_forces(population, xbounds, ybounds, wall_buffer = 0.01, bounce_buffer = 0.005):

    '''checks which people are about to go out of bounds and corrects

    Function that calculates wall repulsion forces on individuals that are about to 
    go outside of the world boundaries.
    
    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    xbounds, ybounds : list or tuple
        contains the lower and upper bounds of the world [min, max]

    wall_buffer, bounce_buffer : float
        buffer used for wall force calculation and returning 
        individuals within bounds
    '''

    dl_bound = [ xbounds[:,0], ybounds[:,0] ]
    ur_bound = [ xbounds[:,1], ybounds[:,1] ]
    
    # Avoid walls
    wall_force = np.zeros_like(population[:,0:2])
    
    for i in range(2):
        
        to_lower = population[:,i + 1] - dl_bound[i]
        to_upper = ur_bound[i] - population[:,i + 1]
        
        # Bounce individuals within the world
        bounce_lo = (to_lower > -bounce_buffer) & (to_lower < 0.0) #& (population[:,i + 3] < 0)
        population[:,i + 3][bounce_lo] = abs(population[:,i + 3][bounce_lo])
        population[:,i + 1][bounce_lo] = dl_bound[i][bounce_lo] + bounce_buffer

        bounce_ur = (to_upper > -bounce_buffer) & (to_upper < 0.0) #& (population[:,i + 3] > 0)
        population[:,i + 3][bounce_ur] = -abs(population[:,i + 3][bounce_ur])
        population[:,i + 1][bounce_ur] = ur_bound[i][bounce_ur] - bounce_buffer

        # Attract outside individuals returning to the world
        lo_outside = (to_lower < 0.0) #& (population[:,i + 3] < 0)
        wall_force[:,i][lo_outside] += abs(1 / ((to_lower[lo_outside])**1))
        # wall_force[:,i][lo_outside] -= abs(1 / ((to_upper[lo_outside])**1))

        ur_outside = (to_upper < 0.0) #& (population[:,i + 3] > 0)
        wall_force[:,i][ur_outside] += abs(1 / ((to_lower[ur_outside])**1))
        # wall_force[:,i][ur_outside] -= abs(1 / ((to_upper[ur_outside])**1))

        # # Repelling force
        # wall_force[:,i] += np.maximum((-1 / wall_buffer**1 + 1 / to_lower**1), 0)
        # wall_force[:,i] -= np.maximum((-1 / wall_buffer**1 + 1 / to_upper**1), 0)

        # Repelling force
        # wall_force[:,i] += np.maximum((1 / to_lower), 0)
        # wall_force[:,i] -= np.maximum((1 / to_upper), 0)

        inside_wall_lower = (to_lower < wall_buffer) & (to_lower > -bounce_buffer)
        inside_wall_upper = (to_upper < wall_buffer) & (to_lower > -bounce_buffer)
        wall_force[:,i][inside_wall_lower] += abs(1 / ((to_lower[inside_wall_lower])**2))
        wall_force[:,i][inside_wall_upper] -= abs(1 / ((to_upper[inside_wall_upper])**2))

        population[:,i + 15] += wall_force[:,i]

    # Update forces
    return population

def update_repulsive_forces(population, social_distance_factor):

    '''calculated repulsive forces between individuals

    Function that calculates repulsion forces between individuals during social distancing.
    
    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    social_distance_factor : float
        Amplitude of repulsive force used to enforce social distancing
    '''

    x = population[:,1]
    y = population[:,2]
    
    dist = cd.cdist(population[:,1:3],population[:,1:3])
    # dist = pairwise_dist(population[:,1:3])
    dist += np.eye(len(population[:,1]))

    to_point_x = pairwise_comp(x)
    to_point_y = pairwise_comp(y)

    repulsion_force_x = - social_distance_factor * np.sum( (to_point_x) / (dist**5) , axis = 1)
    repulsion_force_y = - social_distance_factor * np.sum( (to_point_y) / (dist**5) , axis = 1)

    population[:,15] += repulsion_force_x
    population[:,16] += repulsion_force_y

    # Update forces
    return population

def update_gravity_forces(population, time, last_step_change, wander_step_size = 0.01, 
                          gravity_strength = 0.1, wander_step_duration = 0.01):

    '''updates random perturbation in forces near individuals to cause random motion

    Function that returns geometric parameters of the destination
    that the population members have set.

    Keyword arguments:
    ------------------
    population : ndarray
        the array containing all the population information

    time : float
        current simulation time

    last_step_change : float
        last time value at which a random perturbation was introduced
    
    wander_step_size, gravity_strength, wander_step_duration : float
        proximity of perturbation to individuals, 
        strength of attracion to perturbation,
        length of time perturbation is present
    '''
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
