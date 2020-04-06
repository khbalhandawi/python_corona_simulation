'''
file that contains all function related to population mobility
and related computations
'''

import numpy as np

def update_positions(population):
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
    population[:,1] += population[:,3]
    #y
    population[:,2] += population [:,4]

    return population

def update_velocities(population):
    '''update positions of all people

    Uses heading and speed to update all positions for
    the next time step

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information
    '''

    max_speed = 0.005; dt = 0.001

    for i in range(2):
        # Apply force
        population[:,i + 3] += population[:,i + 15] * dt

        # Limit speed
        speed = np.linalg.norm(population[:,3:5],axis = 1)
        population[speed > max_speed,i + 3] *= max_speed / speed[speed > max_speed]
        # population[speed < min_speed,v_i[i]] *= min_speed / speed[speed < min_speed]

    return population

def out_of_bounds(population, xbounds, ybounds):
    '''checks which people are about to go out of bounds and corrects

    Function that updates headings of individuals that are about to 
    go outside of the world boundaries.
    
    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    xbounds, ybounds : list or tuple
        contains the lower and upper bounds of the world [min, max]
    '''
    #update headings and positions where out of bounds
    #update x heading
    #determine number of elements that need to be updated

    shp = population[:,3][(population[:,1] <= xbounds[:,0]) &
                            (population[:,3] < 0)].shape
    population[:,3][(population[:,1] <= xbounds[:,0]) &
                    (population[:,3] < 0)] = np.clip(np.random.normal(loc = 0.5, 
                                                                        scale = 0.5/3,
                                                                        size = shp),
                                                        a_min = 0.05, a_max = 1)

    shp = population[:,3][(population[:,1] >= xbounds[:,1]) &
                            (population[:,3] > 0)].shape
    population[:,3][(population[:,1] >= xbounds[:,1]) &
                    (population[:,3] > 0)] = np.clip(-np.random.normal(loc = 0.5, 
                                                                        scale = 0.5/3,
                                                                        size = shp),
                                                        a_min = -1, a_max = -0.05)

    #update y heading
    shp = population[:,4][(population[:,2] <= ybounds[:,0]) &
                            (population[:,4] < 0)].shape
    population[:,4][(population[:,2] <= ybounds[:,0]) &
                    (population[:,4] < 0)] = np.clip(np.random.normal(loc = 0.5, 
                                                                        scale = 0.5/3,
                                                                        size = shp),
                                                        a_min = 0.05, a_max = 1)

    shp = population[:,4][(population[:,2] >= ybounds[:,1]) &
                            (population[:,4] > 0)].shape
    population[:,4][(population[:,2] >= ybounds[:,1]) &
                    (population[:,4] > 0)] = np.clip(-np.random.normal(loc = 0.5, 
                                                                        scale = 0.5/3,
                                                                        size = shp),
                                                        a_min = -1, a_max = -0.05)

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
    wall_buffer = 0.08
    wall_force = np.zeros_like(population[:,0:2])
    
    for i in range(2):
        to_lower = population[:,i + 1] - dl_bound[i]
        to_upper = ur_bound[i] - population[:,i + 1]

        # Bounce
        bounce_lo = (to_lower <= 0.0) & (population[:,i + 3] < 0)
        population[:,i + 3][bounce_lo] = abs(population[:,i + 3][bounce_lo])

        bounce_ur = (to_upper <= 0.0) & (population[:,i + 3] > 0)
        population[:,i + 3][bounce_ur] = -abs(population[:,i + 3][bounce_ur])

        # Repelling force
        wall_force[:,i] += np.maximum((-1 / wall_buffer + 1 / to_lower), 0)
        wall_force[:,i] -= np.maximum((-1 / wall_buffer + 1 / to_upper), 0)

        population[:,i + 15] += wall_force[:,i]

    # Update position
    return population

def update_randoms(population, pop_size, speed=0.01, heading_update_chance=0.02, 
                   speed_update_chance=0.02, heading_multiplication=1,
                   speed_multiplication=1):
    '''updates random states such as heading and speed
    
    Function that randomized the headings and speeds for population members
    with settable odds.

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information
    
    pop_size : int
        the size of the population

    heading_update_chance : float
        the odds of updating the heading of each member, each time step

    speed_update_chance : float
        the oodds of updating the speed of each member, each time step

    heading_multiplication : int or float
        factor to multiply heading with (default headings are between -1 and 1)

    speed_multiplication : int or float
        factor to multiply speed with (default speeds are between 0.0001 and 0.05

    speed : int or float
        mean speed of population members, speeds will be taken from gaussian distribution
        with mean 'speed' and sd 'speed / 3'
    '''

    #randomly update heading
    #x
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:,3][update <= heading_update_chance] = np.random.normal(loc = 0, 
                                                        scale = 1/3,
                                                        size = shp) * heading_multiplication
    #y
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:,4][update <= heading_update_chance] = np.random.normal(loc = 0, 
                                                        scale = 1/3,
                                                        size = shp) * heading_multiplication

    return population


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
